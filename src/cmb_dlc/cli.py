"""Local commands for detection, evidence preparation and score integration."""

import argparse
import json
from pathlib import Path
import sys

from .detection import DetectorConfig
from .imaging import file_hash, load_study
from .rendering import Evidence, METHOD
from .runtime import atomic_json, detect_study, output_outside_checkout
from .scoring import summarize
from .vlm import prompts


REQUIRED_CONTRAST = {"op_phase_focality": "phase", "op_mip_focal_dot": "mip",
                     "op_mip_vessel_guard": "mip"}


def detected_candidate(study, run_directory: Path, candidate_id: str):
    run = json.loads((run_directory / "manifest.json").read_text())
    if run["status"] != "complete" or run["provenance"]["input_fingerprint"] != study.fingerprint:
        raise ValueError("Expected a completed detection run for these exact study inputs")
    probability = run_directory / "probability.nii.gz"
    candidate_file = run_directory / "candidates.json"
    for path in (probability, candidate_file):
        if file_hash(path) != run["artifact_sha256"][path.name]:
            raise ValueError("Detection artifact integrity check failed")
    candidates = json.loads(candidate_file.read_text())["candidates"]
    matches = [candidate for candidate in candidates if candidate["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise ValueError("Candidate ID must match exactly one detector candidate")
    return matches[0], candidates, probability


def save_evidence(output: Path, images: dict[str, bytes], record: dict) -> None:
    output = output_outside_checkout(output)
    output.mkdir(mode=0o700)
    for name, image in images.items():
        (output / name).write_bytes(image)
    atomic_json(output / "evidence.json", record)


def prepare_evidence(study, run_directory: Path, candidate_id: str, output: Path) -> dict:
    candidate, candidates, probability = detected_candidate(study, run_directory, candidate_id)
    evidence = Evidence(study, probability, candidates)
    images, tasks = {}, []
    omitted = []
    missing = {"phase", "mip"} - study.volumes.keys()
    for role in METHOD["roles"]:
        role_id = role["id"]
        if REQUIRED_CONTRAST.get(role_id) in missing:
            omitted.append(role_id)
            continue
        selected = {**role, "rows": [row for row in role["rows"]
                                    if not any(row.startswith(modality) for modality in missing)]}
        name = role_id + ".png"
        images[name] = evidence.assessment(candidate, selected)
        system, user = prompts("assessment", selected)
        if missing:
            user += ("\nUnavailable contrasts: " + ", ".join(sorted(missing))
                     + ". Assess only the rows present. Do not infer evidence from unavailable contrasts.")
        tasks.append({"kind": "assessment", "role": role_id, "images": [name],
                      "system": system, "user": user})
    roles = [task["role"] for task in tasks]
    save_evidence(output, images, {"schema_version": 1, "candidate_id": candidate_id,
                                   "detector_score": candidate["detector_score"], "method": METHOD["version"],
                                   "available_roles": roles, "omitted_roles": omitted, "tasks": tasks})
    return {"status": "complete", "candidate_id": candidate_id, "task_count": len(tasks),
            "available_roles": roles, "omitted_roles": omitted}


def prepare_localization(study, run_directory: Path, candidate_id: str, score_path: Path,
                         output: Path, atlas_path: Path, label_map: Path) -> dict:
    from .atlas import RegisteredAtlas

    candidate, _, _ = detected_candidate(study, run_directory, candidate_id)
    scored = json.loads(score_path.read_text())
    decisions = [row for row in scored["candidates"] if row["candidate_id"] == candidate_id]
    if scored.get("schema_version") != 1 or len(decisions) != 1 or decisions[0]["status"] != "retained":
        raise ValueError("Atlas-assisted localization requires a retained candidate from score")
    if "magnitude" not in study.volumes:
        raise ValueError("The default SWI plus magnitude localization requires magnitude")
    evidence = Evidence(study)
    atlas = RegisteredAtlas(atlas_path, label_map, study.volumes["swi"])
    colors, legend = atlas.overlay(evidence.shape, evidence.affine)
    images = {"localization.png": evidence.localization(candidate),
              "localization_atlas.png": evidence.localization(candidate, colors)}
    system, user = prompts("localization")
    context = {"atlas_suggestion": atlas.neighborhood(candidate["native_voxel_ijk"]), "region_rgb": legend}
    user += ("\nThe second composite shows matching planes with a registered MARS overlay. "
             "Use the supplied atlas location and 2 mm neighborhood as suggestions. "
             "Check the visible brain structures and leave ambiguous locations unresolved."
             "\nAtlas context: " + json.dumps(context))
    task = {"kind": "localization", "images": list(images), "system": system, "user": user}
    save_evidence(output, images, {"schema_version": 1, "candidate_id": candidate_id, "tasks": [task]})
    return {"status": "complete", "candidate_id": candidate_id, "task_count": 1}

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cmb-dlc", description="CMB-DLC research inference components")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Check input NIfTI files and spatial metadata")
    validate.add_argument("--study", type=Path, required=True)
    detect = commands.add_parser("detect", help="Run the v4 detector on the full native SWI volume")
    detect.add_argument("--study", type=Path, required=True)
    detect.add_argument("--checkpoint", type=Path, required=True)
    detect.add_argument("--runtime", type=Path, required=True)
    detect.add_argument("--device", required=True, help="cpu or explicit cuda:N")
    detect.add_argument("--batch-size", type=int, default=16)
    detect.add_argument("--retry-incomplete", action="store_true")
    evidence = commands.add_parser("evidence", help="Render images and prompts for one detector candidate")
    evidence.add_argument("--study", type=Path, required=True)
    evidence.add_argument("--run", type=Path, required=True)
    evidence.add_argument("--candidate-id", required=True)
    evidence.add_argument("--output", type=Path, required=True, help="A new private output directory")
    register = commands.add_parser("register", help="Register the published MARS atlas to native SWI")
    register.add_argument("--study", type=Path, required=True)
    register.add_argument("--mni-template", type=Path, required=True)
    register.add_argument("--mni-atlas", type=Path, required=True)
    register.add_argument("--t1", type=Path)
    register.add_argument("--output", type=Path, required=True)
    register.add_argument("--swi-brain", type=Path, help="Existing SynthStrip SWI image on the native grid")
    register.add_argument("--t1-brain", type=Path, help="Existing SynthStrip T1 image on the native grid")
    localize = commands.add_parser("localize", help="Prepare atlas and MRI evidence for a retained finding")
    localize.add_argument("--study", type=Path, required=True)
    localize.add_argument("--run", type=Path, required=True)
    localize.add_argument("--candidate-id", required=True)
    localize.add_argument("--scores", type=Path, required=True)
    localize.add_argument("--output", type=Path, required=True)
    localize.add_argument("--atlas", type=Path, required=True)
    localize.add_argument("--atlas-labels", type=Path, required=True)
    score = commands.add_parser("score", help="Validate external responses and tally retained findings")
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "score":
            result = summarize(json.loads(args.input.read_text()), args.threshold)
        else:
            study = load_study(args.study)
            if args.command == "validate":
                result = study.summary()
            elif args.command == "detect":
                result = detect_study(study, args.checkpoint, args.runtime, args.device,
                                      DetectorConfig(batch_size=args.batch_size), args.retry_incomplete)
            elif args.command == "evidence":
                result = prepare_evidence(study, args.run, args.candidate_id, args.output)
            elif args.command == "register":
                from .registration import register_atlas
                result = register_atlas(study.volumes["swi"], args.mni_template,
                                        args.mni_atlas, args.t1, args.output, args.swi_brain, args.t1_brain)
            else:
                result = prepare_localization(study, args.run, args.candidate_id, args.scores,
                                              args.output, args.atlas, args.atlas_labels)
    except (KeyError, TypeError):
        print("CMB-DLC: invalid input object; see the README for the expected fields", file=sys.stderr)
        return 2
    except ValueError as error:
        print(f"CMB-DLC: {error}", file=sys.stderr)
        return 2
    except ImportError:
        print("CMB-DLC: missing dependency; see the README for detector and registration extras", file=sys.stderr)
        return 2
    except (OSError, RuntimeError) as error:
        print(f"CMB-DLC: {type(error).__name__}; check input availability and output permissions", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
