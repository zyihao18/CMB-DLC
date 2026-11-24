"""Image-only evidence for the seven assessment tasks and localization.

Layouts follow Brain-CMD's seven merged operators and MARS true-cardinal RAS
panels. This renderer has its own version: no historical image cache is reused.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from PIL import Image, ImageDraw

from .imaging import Study, display_images

METHOD = json.loads(Path(__file__).with_name("method.json").read_text())


def gray(values: np.ndarray, low: float | None = None, high: float | None = None) -> np.ndarray:
    finite = values[np.isfinite(values) & (values != 0)]
    if low is None or high is None:
        low, high = np.percentile(finite, [1, 99]) if finite.size else (0., 1.)
    return np.rint(np.clip((np.nan_to_num(values) - low) / max(float(high - low), 1e-8), 0, 1) * 255).astype(np.uint8)


def plane(values: np.ndarray, direction: str, center: np.ndarray):
    i, j, k = np.rint(center).astype(int)
    if direction == "axial":
        return values[:, :, k].swapaxes(0, 1)[::-1], (float(center[0]), values.shape[1] - 1 - float(center[1])), ("L", "R", "A", "P"), (0, 1)
    if direction == "coronal":
        return values[:, j, :].swapaxes(0, 1)[::-1], (float(center[0]), values.shape[2] - 1 - float(center[2])), ("L", "R", "S", "I"), (0, 2)
    if direction == "sagittal":
        return values[i, :, :].swapaxes(0, 1)[::-1], (float(center[1]), values.shape[2] - 1 - float(center[2])), ("P", "A", "S", "I"), (1, 2)
    raise ValueError("Unknown plane")


def mark(draw: ImageDraw.ImageDraw, x: float, y: float, color="#ff5e62", radius=7):
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
    for a, b in [((x - 17, y), (x - 10, y)), ((x + 10, y), (x + 17, y)),
                 ((x, y - 17), (x, y - 10)), ((x, y + 10), (x, y + 17))]:
        draw.line((a, b), fill=color, width=1)


class Evidence:
    def __init__(self, study: Study, probability_path: Path | None = None, candidates: list[dict] | None = None):
        # Validate finite source data before any interpolation.
        for volume in study.volumes.values():
            volume.data()
        images = display_images(study)
        self.affine = images["swi"].affine
        self.shape = images["swi"].shape
        self.spacing = nib.affines.voxel_sizes(self.affine)
        self.data = {name: image.get_fdata(dtype=np.float32, caching="unchanged") for name, image in images.items()}
        self.projection = np.broadcast_to(self.data["swi"].min(axis=2, keepdims=True), self.shape)
        self.probability = None
        if probability_path:
            image = nib.load(probability_path)
            self.probability = resample_from_to(image, (self.shape, self.affine), order=1).get_fdata(dtype=np.float32)
        self.points = {c["candidate_id"]: self.voxel(c["world_ras_mm"]) for c in candidates or []}

    def voxel(self, world) -> np.ndarray:
        return nib.affines.apply_affine(np.linalg.inv(self.affine), world)

    def tile(self, values: np.ndarray, center: np.ndarray, direction: str, title: str,
             crop_size: int | None = None, heatmap: np.ndarray | None = None,
             other_points: bool = False, all_points: bool = False,
             atlas_overlay: np.ndarray | None = None) -> Image.Image:
        array, marker, labels, axes = plane(values, direction, center)
        x, y = marker
        x0 = y0 = 0
        if crop_size:
            x0, y0 = max(0, round(x) - crop_size // 2), max(0, round(y) - crop_size // 2)
            array = array[y0:y0 + crop_size, x0:x0 + crop_size]
        pixels = np.repeat(gray(array)[:, :, None], 3, axis=2)
        if heatmap is not None:
            response, _, _, _ = plane(heatmap, direction, center)
            response = response[y0:y0 + array.shape[0], x0:x0 + array.shape[1]]
            alpha = np.clip(response, 0, 1)[:, :, None] * .7
            pixels = np.rint(pixels * (1 - alpha) + np.array([255, 80, 40]) * alpha).astype(np.uint8)
        if atlas_overlay is not None:
            labels_here, _, _, _ = plane(atlas_overlay, direction, center)
            labels_here = labels_here[y0:y0 + array.shape[0], x0:x0 + array.shape[1]]
            visible = np.any(labels_here != 0, axis=2)
            pixels[visible] = np.rint(.63 * pixels[visible] + .37 * labels_here[visible]).astype(np.uint8)
        physical = np.array([array.shape[1], array.shape[0]]) * self.spacing[list(axes)]
        dimensions = np.maximum(1, np.rint(physical * min(216 / physical))).astype(int)
        image = Image.new("RGB", (240, 264), "#101823")
        offset = ((240 - int(dimensions[0])) // 2, 31 + (216 - int(dimensions[1])) // 2)
        image.paste(Image.fromarray(pixels).resize(tuple(dimensions), Image.Resampling.BILINEAR), offset)
        draw = ImageDraw.Draw(image)
        draw.text((7, 5), title, fill="white")
        for letter, position in zip(labels, [(3, 132), (230, 132), (116, 20), (116, 249)]):
            draw.text(position, letter, fill="#8cd5e5")
        def project(px, py):
            return offset[0] + (px - x0 + .5) * dimensions[0] / array.shape[1], offset[1] + (py - y0 + .5) * dimensions[1] / array.shape[0]
        if other_points and direction == "axial":
            for point in self.points.values():
                if all_points or abs(point[2] - center[2]) <= 1:
                    px, py = point[0], self.shape[1] - 1 - point[1]
                    if x0 <= px < x0 + array.shape[1] and y0 <= py < y0 + array.shape[0]:
                        a, b = project(px, py)
                        draw.ellipse((a - 3, b - 3, a + 3, b + 3), outline="#65d2fa", width=1)
        mark(draw, *project(x, y))
        return image

    def target(self, candidate: dict) -> np.ndarray:
        point = self.voxel(candidate["world_ras_mm"])
        rounded = np.rint(point)
        if not np.isfinite(point).all() or np.any(rounded < 0) or np.any(rounded >= self.shape):
            raise ValueError("Candidate lies outside the display image")
        return point

    def assessment(self, candidate: dict, role: dict) -> bytes:
        if self.probability is None:
            raise ValueError("Assessment requires the whole-volume detector map")
        center = self.target(candidate)
        count = role["slices"]
        canvas = Image.new("RGB", (240 * count, 264 * len(role["rows"])), "#101823")
        for row_index, row in enumerate(role["rows"]):
            for column, delta in enumerate(range(-(count // 2), count // 2 + 1)):
                point = center.copy()
                point[2] = np.clip(round(center[2]) + delta, 0, self.shape[2] - 1)
                modality = "phase" if row.startswith("phase") else "mip" if row.startswith("mip") else "swi"
                context = row in ("mip_context", "full_slice_context", "patient_candidate_projection")
                data = self.data[modality]
                if row == "patient_candidate_projection":
                    # A detector-only whole-volume projection; all marked points remain hypotheses.
                    data = self.projection
                tile = self.tile(data, point, "axial", row.replace("_", " "),
                                 None if context else role["crop_size"],
                                 self.probability if row == "probability_heatmap" else None,
                                 row in ("local_candidate_map", "mip_local_candidate_map", "patient_candidate_projection"),
                                 row == "patient_candidate_projection")
                canvas.paste(tile, (column * 240, row_index * 264))
        stream = io.BytesIO()
        canvas.save(stream, format="PNG")
        return stream.getvalue()

    def localization(self, candidate: dict, atlas_overlay: np.ndarray | None = None) -> bytes:
        if "magnitude" not in self.data:
            raise ValueError("Default paired localization requires magnitude")
        point = self.target(candidate)
        canvas = Image.new("RGB", (720, 528), "#101823")
        for row, modality in enumerate(("swi", "magnitude")):
            for column, direction in enumerate(("axial", "coronal", "sagittal")):
                title = f"{modality} {direction}" + (" + atlas" if atlas_overlay is not None else "")
                tile = self.tile(self.data[modality], point, direction, title, atlas_overlay=atlas_overlay)
                canvas.paste(tile, (column * 240, row * 264))
        stream = io.BytesIO()
        canvas.save(stream, format="PNG")
        return stream.getvalue()
