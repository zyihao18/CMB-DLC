"""Locate candidates in a supplied MARS label image registered to native SWI."""

from collections import Counter
from colorsys import hsv_to_rgb
from pathlib import Path
import json

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np

from .imaging import Volume
from .vlm import REGION_GROUPS

SIDES = ("left", "right", "midline_or_uncertain")


class RegisteredAtlas:
    def __init__(self, labels: Path, label_map: Path, swi: Volume):
        image = nib.load(labels)
        if (image.shape != swi.image.shape
                or not np.allclose(image.affine, swi.image.affine, atol=1e-4, rtol=0)
                or image.header.get_xyzt_units()[0] != "mm"):
            raise ValueError("The MARS label image must match the native SWI grid and millimetre units")
        values = np.asanyarray(image.dataobj)
        if (not np.isfinite(values).all() or np.any(values < 0)
                or np.any(values != np.rint(values)) or values.max() > np.iinfo(np.int32).max):
            raise ValueError("Atlas voxels must contain nonnegative integer label IDs")
        self.labels = values.astype(np.int32, copy=False)
        self.image = nib.Nifti1Image(self.labels, swi.image.affine)
        self.label_map = {0: {"mars_region": "unassigned", "hemisphere": "midline_or_uncertain"}}
        supplied = json.loads(label_map.read_text())
        if not isinstance(supplied, dict) or not supplied:
            raise ValueError("Provide a JSON object mapping positive atlas IDs to region and side")
        for key, entry in supplied.items():
            if (not key.isascii() or not key.isdecimal() or key.startswith("0")
                    or not isinstance(entry, dict) or set(entry) != {"mars_region", "hemisphere"}
                    or entry["mars_region"] not in REGION_GROUPS or entry["hemisphere"] not in SIDES):
                raise ValueError("Invalid MARS label mapping")
            self.label_map[int(key)] = entry
        if not set(np.unique(self.labels)) <= self.label_map.keys():
            raise ValueError("The atlas contains a label missing from the supplied map")

    def neighborhood(self, native_voxel_ijk: list[float]) -> dict:
        point = np.asarray(native_voxel_ijk, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("A finite native voxel coordinate is required")
        center = np.rint(point).astype(int)
        shape = np.asarray(self.labels.shape)
        if np.any(center < 0) or np.any(center >= shape):
            raise ValueError("Candidate lies outside the native SWI grid")
        linear = self.image.affine[:3, :3]
        # Bound a 2 mm physical sphere in array indices, including oblique grids.
        radius = 2.0
        reach = np.ceil(radius * np.linalg.norm(np.linalg.inv(linear), axis=1)).astype(int)
        lo = np.maximum(center - reach, 0)
        hi = np.minimum(center + reach + 1, shape)
        points = np.stack(np.meshgrid(*(np.arange(a, b) for a, b in zip(lo, hi)), indexing="ij"), -1)
        mask = np.linalg.norm((points - center) @ linear.T, axis=-1) <= radius + 1e-8
        voxels = self.labels[tuple(points[mask].T)]
        counts = Counter(self.label_map[int(label)]["mars_region"] for label in voxels)
        return {
            **self.label_map[int(self.labels[tuple(center)])],
            "region_fractions_2mm": {region: n / voxels.size for region, n in sorted(counts.items())},
        }

    def overlay(self, shape: tuple[int, int, int], affine: np.ndarray) -> tuple[np.ndarray, dict]:
        values = np.asarray(resample_from_to(self.image, (shape, affine), order=0).dataobj)
        pixels = np.zeros((*shape, 3), dtype=np.uint8)
        colors = {
            region: [round(255 * v) for v in hsv_to_rgb(i * .61803398875 % 1, .66, 1)]
            for i, region in enumerate(REGION_GROUPS)
        }
        legend = {}
        for label, entry in self.label_map.items():
            if label == 0:
                continue
            region = entry["mars_region"]
            pixels[values == label] = colors[region]
            legend[region] = colors[region]
        return pixels, legend
