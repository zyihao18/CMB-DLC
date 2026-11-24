"""Native NIfTI geometry and explicit, annotation-free study manifests.

Coordinates are always array-axis indices (i, j, k), mapped to RAS millimetres
by the source affine. Display resampling never changes detector input geometry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_from_to, resample_to_output
import numpy as np

MODALITIES = ("swi", "phase", "magnitude", "mip")
GEOMETRY_VERSION = "native-ijk-cardinal-ras-v1"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


@dataclass
class Volume:
    image: nib.Nifti1Image = field(repr=False)
    sha256: str

    def summary(self) -> dict:
        affine = self.image.affine
        return {"sha256": self.sha256, "shape": list(self.image.shape),
                "affine": affine.tolist(),
                "spacing_mm": nib.affines.voxel_sizes(affine).tolist(),
                "orientation": list(nib.aff2axcodes(affine)),
                "obliquity_degrees": np.rad2deg(nib.affines.obliquity(affine)).tolist()}

    def data(self) -> np.ndarray:
        data = self.image.get_fdata(dtype=np.float32, caching="unchanged")
        if not np.isfinite(data).all():
            raise ValueError("Image contains non-finite intensities")
        return data

    def world(self, voxel: list[float]) -> list[float]:
        return nib.affines.apply_affine(self.image.affine, voxel).tolist()


@dataclass
class Study:
    volumes: dict[str, Volume]

    @property
    def fingerprint(self) -> str:
        return content_hash({name: volume.sha256 for name, volume in self.volumes.items()})

    @property
    def id(self) -> str:
        return "s_" + self.fingerprint[:24]

    def summary(self) -> dict:
        missing = [name for name in MODALITIES if name not in self.volumes]
        return {"schema_version": 1, "study_id": self.id,
                "input_fingerprint": self.fingerprint, "geometry_version": GEOMETRY_VERSION,
                "modalities": {k: v.summary() for k, v in self.volumes.items()},
                "missing_modalities": missing,
                "input_readiness": {
                    "detection": "available",
                    "seven_assessments": "available" if not ({"phase", "mip"} & set(missing))
                    else "missing_required_modality",
                    "paired_localization": "available" if "magnitude" not in missing
                    else "missing_required_modality"},
                "mip_policy": "Separate scanner slab grid; sample through its own affine. "
                              "Equal SWI and minIP indices are not assumed."}


def load_study(manifest_path: Path, max_voxels: int = 128_000_000) -> Study:
    """Load only explicit image paths; never discover samples or annotation files."""
    manifest_path = manifest_path.resolve()
    if manifest_path.stat().st_size > 65536:
        raise ValueError("Study manifest exceeds 64 KiB")
    payload = json.loads(manifest_path.read_text())
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "modalities"}:
        raise ValueError("Manifest requires only schema_version and modalities")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("Unsupported study schema version")
    paths = payload["modalities"]
    if not isinstance(paths, dict) or "swi" not in paths or not set(paths) <= set(MODALITIES):
        raise ValueError("Supply SWI and optional phase, magnitude, mip paths only")
    volumes = {}
    for name, value in sorted(paths.items()):
        if not isinstance(value, str) or not value:
            raise ValueError("Each modality requires a nonempty NIfTI path")
        path = Path(value)
        if not path.is_absolute():
            path = manifest_path.parent / path
        if not (path.name.endswith(".nii") or path.name.endswith(".nii.gz")):
            raise ValueError("Only NIfTI .nii and .nii.gz images are supported")
        if not path.is_file():
            raise ValueError(f"Missing {name} image")
        try:
            image = nib.load(path)
        except Exception:
            raise ValueError(f"Unreadable {name} NIfTI image") from None
        if len(image.shape) != 3 or min(image.shape) < 1 or np.prod(image.shape) > max_voxels:
            raise ValueError(f"Invalid or oversized {name} three-dimensional image")
        if image.header.get_xyzt_units()[0] != "mm":
            raise ValueError(f"{name} must explicitly declare millimetre spatial units")
        affine = image.affine
        if (not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-10
                or not np.allclose(affine[3], [0, 0, 0, 1])):
            raise ValueError(f"Invalid {name} affine")
        volumes[name] = Volume(image, file_hash(path))
    # Catch disjoint modalities before any spatially misleading combination.
    reference = volumes["swi"].image
    for name, volume in volumes.items():
        if name == "swi":
            continue
        bounds = []
        for image in (reference, volume.image):
            corners = np.array(np.meshgrid(*[(0, n - 1) for n in image.shape],
                                          indexing="ij")).reshape(3, -1).T
            world = nib.affines.apply_affine(image.affine, corners)
            bounds.append((world.min(axis=0), world.max(axis=0)))
        if np.any(np.maximum(bounds[0][0], bounds[1][0]) >
                  np.minimum(bounds[0][1], bounds[1][1])):
            raise ValueError(f"{name} has no physical overlap with SWI")
    return Study(volumes)


def display_images(study: Study) -> dict[str, nib.Nifti1Image]:
    """True-cardinal RAS images on one shared physical grid, including oblique input.

    Scanner minIP retains its slab meaning; resampling is for display only and
    cannot establish which source slices contributed to a scanner projection.
    """
    swi = study.volumes["swi"].image
    target = resample_to_output(swi, voxel_sizes=nib.affines.voxel_sizes(swi.affine), order=1)
    return {name: target if name == "swi" else resample_from_to(
        volume.image, (target.shape, target.affine), order=1)
        for name, volume in study.volumes.items()}
