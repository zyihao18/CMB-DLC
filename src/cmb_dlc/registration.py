"""Image-only MARS atlas transfer by the two registration routes in the paper."""

import os
from pathlib import Path
import subprocess
import tempfile

import nibabel as nib
import numpy as np

from .atlas import SIDES
from .imaging import Volume, file_hash
from .runtime import atomic_json, output_outside_checkout
from .vlm import REGION_GROUPS


def use_t1(image) -> bool:
    return image is not None and len(image.shape) == 3 and bool(
        np.all(nib.affines.voxel_sizes(image.affine) <= 2)
    )


def same_grid(first, second) -> bool:
    return first.shape == second.shape and np.allclose(first.affine, second.affine, rtol=0, atol=1e-4)


def brain_image(source: Path, image, prepared: Path | None, directory: Path, name: str) -> Path:
    if prepared is None:
        prepared = directory / f"{name}_brain.nii.gz"
        try:
            subprocess.run(
                ["mri_synthstrip", "--threads", "2", "-i", str(source), "-o", str(prepared)],
                check=True, capture_output=True, timeout=900,
            )
        except FileNotFoundError:
            raise ValueError("Install SynthStrip or supply a brain-extracted image on the native grid") from None
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            raise ValueError("SynthStrip failed or timed out") from None
    brain = nib.load(prepared)
    if not same_grid(image, brain):
        raise ValueError("The brain-extracted image must have the original image grid")
    values = brain.get_fdata(dtype=np.float32)
    retained = values != 0
    if (not retained.any() or not np.isfinite(values).all()
            or not np.allclose(values[retained], image.get_fdata(dtype=np.float32)[retained],
                               atol=1e-3, rtol=0)):
        raise ValueError("Brain extraction must retain the original in-brain image intensities")
    return prepared


def register_atlas(swi: Volume, template_path: Path, atlas_path: Path, t1_path: Path | None,
                   output: Path, swi_brain: Path | None = None, t1_brain: Path | None = None) -> dict:
    """Fit the transforms on intensity images and transfer the labels afterward."""
    os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "2")
    import ants

    output = output_outside_checkout(output)
    if output.exists():
        raise ValueError("The registration output directory must be new")
    template = nib.load(template_path)
    atlas = nib.load(atlas_path)
    if len(template.shape) != 3 or not same_grid(template, atlas):
        raise ValueError("The MNI template and published MARS atlas must share one grid")
    values = atlas.get_fdata()
    if not np.isfinite(values).all() or np.any(values < 0) or values.max() > 169.003:
        raise ValueError("Invalid published MARS atlas values")
    region = np.rint(np.sqrt(values)).astype(np.uint8)
    if not np.allclose(values, region.astype(float)**2, atol=.003, rtol=0):
        raise ValueError("Expected the published MARS atlas with squared label IDs")
    x = np.ogrid[:region.shape[0], :region.shape[1], :region.shape[2]]
    ras_x = sum(axis * atlas.affine[0, i] for i, axis in enumerate(x)) + atlas.affine[0, 3]
    sides = np.where(ras_x < -.5, 1, np.where(ras_x > .5, 2, 3)).astype(np.uint8)
    sides[region == 0] = 0
    t1 = nib.load(t1_path) if t1_path is not None else None
    via_t1 = use_t1(t1)
    sources = {"swi": Path(swi.image.get_filename()), "template": template_path, "atlas": atlas_path}
    if swi_brain is not None:
        sources["swi_brain"] = swi_brain
    if via_t1:
        sources["t1"] = t1_path
        if t1_brain is not None:
            sources["t1_brain"] = t1_brain
    hashes = {name: file_hash(path) for name, path in sources.items()}
    if hashes["swi"] != swi.sha256:
        raise ValueError("The native SWI changed after it was loaded")
    settings = {
        "random_seed": 20261102,
        "aff_iterations": (1000, 500, 250, 100),
        "aff_shrink_factors": (8, 4, 2, 1),
        "aff_smoothing_sigmas": (3, 2, 1, 0),
    }

    with tempfile.TemporaryDirectory(prefix="cmb-atlas-") as scratch:
        p = Path(scratch)
        fixed = ants.image_read(str(brain_image(Path(swi.image.get_filename()), swi.image, swi_brain, p, "swi")))
        if via_t1:
            bridge = ants.image_read(str(brain_image(t1_path, t1, t1_brain, p, "t1")))
        else:
            bridge = fixed
        moving = ants.image_read(str(template_path))
        first = ants.registration(fixed=bridge, moving=moving, type_of_transform="Affine",
                                  outprefix=str(p / "mni_"), **settings)
        chain = first["fwdtransforms"]
        if via_t1:
            second = ants.registration(fixed=fixed, moving=bridge, type_of_transform="Rigid",
                                       outprefix=str(p / "t1_"), **settings)
            chain = second["fwdtransforms"] + chain
        if not chain or any(
            np.linalg.det(np.asarray(ants.read_transform(f).parameters[:9]).reshape(3, 3)) <= 0
            for f in chain
        ):
            raise ValueError("Atlas registration returned an invalid or reflected transform")

        def warp(data):
            path = p / "mni_labels.nii.gz"
            nib.save(nib.Nifti1Image(data, atlas.affine, atlas.header), path)
            image = ants.apply_transforms(fixed=fixed, moving=ants.image_read(str(path)),
                                          transformlist=chain, interpolator="genericLabel")
            return np.rint(image.numpy()).astype(np.int32)

        native_region, native_side = warp(region), warp(sides)
        if (native_region.shape != swi.image.shape or native_side.shape != swi.image.shape
                or not set(np.unique(native_region)) <= set(range(14))
                or not set(np.unique(native_side)) <= set(range(4)) or not np.any(native_region)):
            raise ValueError("Registered atlas labels are invalid on the native SWI grid")
    if any(file_hash(path) != hashes[name] for name, path in sources.items()):
        raise ValueError("A registration input changed during execution")
    packed = np.where(native_region > 0, 4 * native_region + native_side, 0).astype(np.uint8)
    label_map = {}
    regions = tuple(REGION_GROUPS)
    for code in np.unique(packed):
        if code:
            region_id, side_id = divmod(int(code), 4)
            label_map[str(code)] = {"mars_region": regions[region_id - 1],
                                    "hemisphere": SIDES[side_id - 1] if side_id else SIDES[2]}

    output.mkdir(mode=0o700)
    result = nib.Nifti1Image(packed, swi.image.affine)
    result.header.set_xyzt_units("mm")
    nib.save(result, output / "mars_swi.nii.gz")
    atomic_json(output / "mars_labels.json", label_map)
    manifest = {"schema_version": 1, "status": "complete", "route": "t1_bridge" if via_t1 else "direct_swi",
                "input_sha256": hashes,
                "label_interpolation": "genericLabel", "settings": settings}
    atomic_json(output / "registration.json", manifest)
    return manifest
