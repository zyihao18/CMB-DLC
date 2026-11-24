"""Checkpoint-compatible v4 multiscale inference, without evaluation imports.

Adapted from Brain-CMD eval_v4_all_patients.py and analyze_candidate_distribution.py,
local source revision dc1b552ea5238eb93170e8c182ee515167b025b9 plus source hashes
described in the repository README.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
from pathlib import Path
from typing import Callable

import nibabel as nib
import numpy as np
from scipy import ndimage

from .imaging import content_hash, file_hash

PREPROCESSING_VERSION = "v4-native-three-scale-gaussian-complete-coverage-v1"


@dataclass(frozen=True)
class DetectorConfig:
    patch_size: int = 64
    num_slices: int = 5
    scales: tuple[int, ...] = (32, 64, 128)
    stride: int = 16
    batch_size: int = 16
    proposal_threshold: float = 0.3
    min_voxels: int = 2
    max_voxels: int = 500

    def __post_init__(self):
        if (self.patch_size != 64 or self.num_slices != 5 or self.scales != (32, 64, 128)
                or self.stride != 16):
            raise ValueError("The detector requires the v4 preprocessing configuration")
        if not 1 <= self.batch_size <= 128:
            raise ValueError("Batch size must be between 1 and 128")
        if not 0 < self.proposal_threshold <= 1 or not 1 <= self.min_voxels <= self.max_voxels:
            raise ValueError("Invalid component proposal settings")


def normalize_intensity(data: np.ndarray) -> np.ndarray:
    if data.ndim != 3 or not np.isfinite(data).all():
        raise ValueError("A finite three-dimensional SWI image is required")
    ceiling = np.percentile(data, 99.5)
    if ceiling <= 0:
        raise ValueError("SWI contains no usable positive intensities")
    return (np.clip(data, 0, ceiling) / ceiling).astype(np.float32)


def crop(volume: np.ndarray, center: tuple[int, int, int], shape: tuple[int, int, int]) -> np.ndarray:
    """Zero-padded native-axis crop, matching the reference row/column convention."""
    start = [int(c) - s // 2 for c, s in zip(center, shape)]
    result = np.zeros(shape, dtype=np.float32)
    source, destination = [], []
    for origin, size, limit in zip(start, shape, volume.shape):
        lo, hi = max(0, origin), min(limit, origin + size)
        if hi <= lo:
            return result
        source.append(slice(lo, hi))
        destination.append(slice(lo - origin, hi - origin))
    result[tuple(destination)] = volume[tuple(source)]
    return result


def load_checkpoint(path: Path, device: str, config: DetectorConfig):
    import torch
    from .model import CompactUNet25D

    if device != "cpu" and not (device.startswith("cuda:") and device[5:].isdigit()):
        raise ValueError("Device must be cpu or an explicit cuda:N")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected = {"in_channels": 3, "num_slices": config.num_slices,
                "crop_size": config.patch_size, "scale_crop_sizes": list(config.scales),
                "inference_stride": config.stride}
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Checkpoint has incompatible {key}")
    model = CompactUNet25D(in_channels=3, num_slices=5, out_channels=1)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), {"checkpoint_sha256": file_hash(path),
                                    "checkpoint_contract": expected}


def window_positions(shape: tuple[int, int, int], config: DetectorConfig):
    if min(shape[:2]) < config.patch_size or shape[2] < config.num_slices:
        raise ValueError("v4 requires at least 64 x 64 x 5 native voxels")
    axes = []
    for extent in shape[:2]:
        positions = list(range(0, extent - config.patch_size + 1, config.stride))
        if positions[-1] != extent - config.patch_size:
            positions.append(extent - config.patch_size)
        axes.append(positions)
    half = config.num_slices // 2
    return itertools.product(range(half, shape[2] - half), *axes)


def sliding_window(model, swi: np.ndarray, device: str, config: DetectorConfig,
                   progress: Callable[[int, int], None] = lambda done, total: None) -> np.ndarray:
    """Whole native volume, no lesion/brain-mask/reference-based crop selection.

    Appending the final window fixes uncovered remainders in non-multiple-of-16
    source dimensions. On the usual grid this preserves reference window order.
    """
    import torch

    # Validate dimensions before allocating intermediates.
    window_positions(swi.shape, config)
    swi = normalize_intensity(swi)
    scaled = [(config.patch_size / size, ndimage.zoom(
        swi, (config.patch_size / size, config.patch_size / size, 1), order=1)
        if size != config.patch_size else swi) for size in config.scales]
    prediction = np.zeros(swi.shape, dtype=np.float64)
    weights = np.zeros(swi.shape, dtype=np.float64)
    axis = np.arange(config.patch_size) - (config.patch_size - 1) / 2
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    gaussian = np.exp(-(xx ** 2 + yy ** 2) / (2 * (config.patch_size * 0.125) ** 2))
    gaussian = (gaussian / gaussian.max())[:, :, None]
    positions = window_positions(swi.shape, config)
    total = sum(1 for _ in window_positions(swi.shape, config))
    done = 0
    with torch.inference_mode():
        while batch := list(itertools.islice(positions, config.batch_size)):
            tensors = []
            for k, i, j in batch:
                channels = []
                for factor, volume in scaled:
                    block = crop(volume, (round((i + 32) * factor),
                                          round((j + 32) * factor), k), (64, 64, 5))
                    block = (block - block.mean()) / (block.std() + 1e-8)
                    channels.append(block.transpose(2, 0, 1))
                tensors.append(np.stack(channels))
            inputs = torch.from_numpy(np.stack(tensors)).float().to(device)
            outputs = model(inputs).cpu().numpy()
            if outputs.shape != (len(batch), 1, 5, 64, 64) or not np.isfinite(outputs).all():
                raise ValueError("Detector returned invalid probability tensors")
            if outputs.min() < 0 or outputs.max() > 1:
                raise ValueError("Detector probabilities are outside [0,1]")
            for index, (k, i, j) in enumerate(batch):
                region = np.s_[i:i + 64, j:j + 64, k - 2:k + 3]
                prediction[region] += outputs[index, 0].transpose(1, 2, 0) * gaussian
                weights[region] += gaussian
            done += len(batch)
            progress(done, total)  # Caller may raise to cancel between bounded batches.
    if np.any(weights == 0):
        raise ValueError("Whole-volume coverage check failed")
    return (prediction / weights).astype(np.float32)


def extract_candidates(probability: np.ndarray, affine: np.ndarray, study_fingerprint: str,
                       config: DetectorConfig) -> list[dict]:
    if (probability.ndim != 3 or not np.isfinite(probability).all()
            or probability.min() < 0 or probability.max() > 1):
        raise ValueError("Invalid whole-volume probability map")
    labels, _ = ndimage.label(probability >= config.proposal_threshold)
    candidates = []
    for label_id, bounds in enumerate(ndimage.find_objects(labels), start=1):
        if bounds is None:
            continue
        mask = labels[bounds] == label_id
        count = int(mask.sum())
        if not config.min_voxels <= count <= config.max_voxels:
            continue
        values = probability[bounds]
        coordinates = np.argwhere(mask) + np.array([s.start for s in bounds])
        center = np.average(coordinates, axis=0, weights=values[mask])
        signature = {"study": study_fingerprint, "component_voxels": coordinates.tolist(),
                     "proposal": asdict(config), "version": PREPROCESSING_VERSION}
        # Batch size affects scheduling only, so it must not affect candidate identity.
        signature["proposal"].pop("batch_size")
        candidates.append({"schema_version": 1, "candidate_id": "c_" + content_hash(signature)[:24],
                           "native_voxel_ijk": center.tolist(),
                           "world_ras_mm": nib.affines.apply_affine(affine, center).tolist(),
                           "detector_score": float(values[mask].max()), "voxel_count": count,
                           "bbox_min_ijk": coordinates.min(axis=0).tolist(),
                           "bbox_max_ijk": coordinates.max(axis=0).tolist()})
    return sorted(candidates, key=lambda c: (-c["detector_score"], c["native_voxel_ijk"][2],
                                            c["native_voxel_ijk"][0], c["native_voxel_ijk"][1]))
