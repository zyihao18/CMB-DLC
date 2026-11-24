import io
import json
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
from PIL import Image

from cmb_dlc.atlas import RegisteredAtlas
from cmb_dlc.cli import prepare_evidence, prepare_localization
from cmb_dlc.detection import DetectorConfig, extract_candidates, window_positions
from cmb_dlc.imaging import file_hash, load_study
from cmb_dlc.scoring import WEIGHTS, fuse_scores, summarize
from cmb_dlc.runtime import output_outside_checkout


def save_volume(path: Path, array: np.ndarray, affine: np.ndarray):
    vol = nib.Nifti1Image(array, affine)
    vol.header.set_xyzt_units("mm")
    nib.save(vol, path)


class CoreTest(unittest.TestCase):
    def test_fixed_fusion_renormalizes_available_roles(self):
        a, b, *_ = WEIGHTS
        actual = fuse_scores(.7, {a: .2, b: .9})
        expected = .2 * .7 + .8 * (.25 * .2 + .15 * .9) / .4
        self.assertAlmostEqual(actual, expected)
        with self.assertRaises(ValueError):
            fuse_scores(.7, {"not-a-role": .5})

    def test_pending_response_is_not_an_exclusion(self):
        role = next(iter(WEIGHTS))
        result = summarize({"schema_version": 1, "available_roles": [role],
                            "candidates": [{"candidate_id": "s1", "detector_score": .8,
                                            "assessments": {}}]}, threshold=.5)
        self.assertEqual(result["candidates"][0]["status"], "pending_assessment")
        self.assertIsNone(result["candidates"][0]["score"])
        self.assertEqual(len(result["regional_counts"]), 13)

    def test_visibility_does_not_change_the_fixed_fusion(self):
        role = next(iter(WEIGHTS))
        response = {"target_visible": "uncertain", "classification": "possible_lesion",
                    "real_score": .6, "confidence": .3, "mimic_risk": "uncertain",
                    "evidence": ["Synthetic response"], "concise_rationale": "Visibility is limited"}
        payload = {"schema_version": 1, "available_roles": [role],
                   "candidates": [{"candidate_id": "s1", "detector_score": .7,
                                   "assessments": {role: response}}]}
        candidate = summarize(payload, .6)["candidates"][0]
        self.assertAlmostEqual(candidate["score"], .62)
        self.assertEqual(candidate["status"], "retained")
        self.assertEqual(candidate["limited_visibility_roles"], [role])

    def test_generated_outputs_are_rejected_inside_source_tree(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            output_outside_checkout(Path(__file__).resolve().parents[1] / "generated")

    def test_native_edge_coverage_and_affine(self):
        config = DetectorConfig()
        windows = list(window_positions((71, 69, 6), config))
        self.assertIn((3, 7, 5), windows)
        probability = np.zeros((71, 69, 6), dtype=np.float32)
        probability[10:12, 11, 3] = (.6, .9)
        affine = np.array([[0., -2., 0., 30.], [1.2, 0., 0., -10.],
                           [0., 0., 3., -6.], [0., 0., 0., 1.]])
        findings = extract_candidates(probability, affine, "synthetic", config)
        self.assertEqual(len(findings), 1)
        native = findings[0]["native_voxel_ijk"]
        np.testing.assert_allclose(native, [10.6, 11, 3], atol=1e-5)
        np.testing.assert_allclose(findings[0]["world_ras_mm"], nib.affines.apply_affine(affine, native))

    def test_evidence_and_registered_atlas(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shape = (21, 18, 7)
            affine = np.diag([1.2, 1.2, 2.4, 1.])
            example = np.indices(shape).sum(axis=0).astype(np.float32)
            paths = {key: f"{key}.nii.gz" for key in ("swi", "phase", "magnitude", "mip")}
            for filename in paths.values():
                save_volume(root / filename, example, affine)
            study_path = root / "study.json"
            study_path.write_text(json.dumps({"schema_version": 1, "modalities": paths}))
            study = load_study(study_path)

            run = root / "run"
            run.mkdir()
            probs = np.zeros(shape, np.float32)
            probs[10:12, 9, 3] = .8
            prob_path = run / "probability.nii.gz"
            save_volume(prob_path, probs, affine)
            candidates = extract_candidates(probs, study.volumes["swi"].image.affine, study.fingerprint,
                                            DetectorConfig())
            candidate_file = run / "candidates.json"
            candidate_file.write_text(json.dumps({"candidates": candidates}))
            (run / "manifest.json").write_text(json.dumps({
                "status": "complete", "provenance": {"input_fingerprint": study.fingerprint},
                "artifact_sha256": {p.name: file_hash(p) for p in (prob_path, candidate_file)},
            }))

            labels = np.zeros(shape, np.int32)
            labels[9:13, 8:12, 3] = 1
            atlas_path = root / "atlas.nii.gz"
            save_volume(atlas_path, labels, affine)
            map_path = root / "labels.json"
            map_path.write_text(json.dumps({"1": {"mars_region": "frontal", "hemisphere": "left"}}))
            atlas = RegisteredAtlas(atlas_path, map_path, study.volumes["swi"])
            hint = atlas.neighborhood(candidates[0]["native_voxel_ijk"])
            self.assertEqual(hint["hemisphere"], "left")
            self.assertAlmostEqual(sum(hint["region_fractions_2mm"].values()), 1.0)

            dest = root / "evidence"
            key = candidates[0]["candidate_id"]
            result = prepare_evidence(study, run, key, dest)
            self.assertEqual(result["task_count"], 7)
            tasks = json.loads((dest / "evidence.json").read_text())["tasks"]
            self.assertEqual(set(WEIGHTS), {t["role"] for t in tasks})

            scores = root / "scores.json"
            scores.write_text(json.dumps({"schema_version": 1, "candidates": [
                {"candidate_id": key, "status": "excluded"}]}))
            with self.assertRaisesRegex(ValueError, "retained candidate"):
                prepare_localization(study, run, key, scores, root / "location", atlas_path, map_path)
            scores.write_text(json.dumps({"schema_version": 1, "candidates": [
                {"candidate_id": key, "status": "retained"}]}))
            out = root / "location"
            prepare_localization(study, run, key, scores, out, atlas_path, map_path)
            task = json.loads((out / "evidence.json").read_text())["tasks"][0]
            self.assertEqual(len(task["images"]), 2)
            self.assertIn("region_fractions_2mm", task["user"])
            self.assertNotIn("detector_score", task["user"])
            panels = [(out / name).read_bytes() for name in task["images"]]
            self.assertNotEqual(*panels)
            with Image.open(io.BytesIO(panels[-1])) as image:
                self.assertEqual(image.size, (720, 528))

            paths.pop("phase")
            paths.pop("mip")
            study_path.write_text(json.dumps({"schema_version": 1, "modalities": paths}))
            reduced = load_study(study_path)
            manifest = json.loads((run / "manifest.json").read_text())
            manifest["provenance"]["input_fingerprint"] = reduced.fingerprint
            (run / "manifest.json").write_text(json.dumps(manifest))
            subset = prepare_evidence(reduced, run, key, root / "limited")["available_roles"]
            self.assertEqual(len(subset), 4)
            self.assertNotIn("op_phase_focality", subset)
            self.assertNotIn("op_mip_focal_dot", subset)
            candidate_file.write_text("{}")
            with self.assertRaisesRegex(ValueError, "integrity"):
                prepare_evidence(reduced, run, key, root / "invalid")
            self.assertFalse((root / "invalid").exists())


if __name__ == "__main__":
    unittest.main()
