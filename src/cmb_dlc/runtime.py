"""Save detection results with a single writer and verify matching cached artifacts."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np
import scipy

from . import __version__
from .detection import (DetectorConfig, PREPROCESSING_VERSION, extract_candidates,
                        load_checkpoint, sliding_window)
from .imaging import Study, content_hash, file_hash


def atomic_json(path: Path, value: dict) -> None:
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".write-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def output_outside_checkout(path: Path) -> Path:
    resolved = path.resolve()
    for parent in (resolved, *resolved.parents):
        if (parent / "pyproject.toml").is_file() and (parent / "src/cmb_dlc/method.json").is_file():
            raise ValueError("Use an output directory outside the CMB-DLC source checkout")
    return resolved


def detect_study(study: Study, checkpoint: Path, runtime: Path, device: str,
                 config: DetectorConfig, retry_incomplete: bool = False) -> dict:
    import torch
    torch.set_num_threads(2)
    runtime = output_outside_checkout(runtime)
    runtime.mkdir(mode=0o700, parents=True, exist_ok=True)
    if runtime.stat().st_mode & 0o077:
        raise ValueError("Runtime directory must have owner-only permissions (chmod 700)")
    signature = {"schema_version": 1, "package_version": __version__,
                 "input_fingerprint": study.fingerprint,
                 "checkpoint_sha256": file_hash(checkpoint),
                 "preprocessing_version": PREPROCESSING_VERSION, "configuration": asdict(config),
                 "code_sha256": {name: file_hash(Path(__file__).with_name(name + ".py"))
                                 for name in ("imaging", "model", "detection", "runtime")},
                 "software": {"torch": torch.__version__, "nibabel": nib.__version__,
                              "numpy": np.__version__, "scipy": scipy.__version__},
                 "device": device}
    run_id = "r_" + content_hash(signature)
    directory = runtime / run_id
    directory.mkdir(mode=0o700, exist_ok=True)
    with (directory / "writer.lock").open("a") as lock:
        os.chmod(directory / "writer.lock", 0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("This exact detection run already has a live writer") from None
        status_path = directory / "manifest.json"
        if status_path.exists():
            saved = json.loads(status_path.read_text())
            if saved["status"] == "complete":
                for name, digest in saved["artifact_sha256"].items():
                    if file_hash(directory / name) != digest:
                        raise ValueError("Saved run artifact integrity check failed")
                return {**saved, "execution_mode": "compatible_cache_reuse"}
            if not retry_incomplete:
                raise ValueError("Incomplete run requires explicit --retry-incomplete; "
                                 "local CNN computation restarts, no VLM calls are involved")
        status = {"schema_version": 1, "run_id": run_id, "stage": "detection",
                  "status": "running", "execution_mode": "live_inference",
                  "provenance": signature, "study": study.summary(),
                  "started_at": datetime.now(timezone.utc).isoformat(),
                  "device": device, "windows_completed": 0, "windows_total": None,
                  "downstream_status": "not_run"}
        atomic_json(status_path, status)

        def progress(done: int, total: int):
            if (directory / "cancel.request").exists():
                raise InterruptedError("Cancellation requested")
            status.update(windows_completed=done, windows_total=total)
            atomic_json(status_path, status)

        try:
            if (directory / "cancel.request").exists():
                raise InterruptedError("Cancellation requested")
            model, checkpoint_info = load_checkpoint(checkpoint, device, config)
            if checkpoint_info["checkpoint_sha256"] != signature["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed during run preparation")
            swi = study.volumes["swi"]
            probability = sliding_window(model, swi.data(), device, config, progress)
            candidates = extract_candidates(probability, swi.image.affine, study.fingerprint, config)
            # Ensure opened image content still matches the run's frozen input fingerprints.
            for volume in study.volumes.values():
                if file_hash(Path(volume.image.get_filename())) != volume.sha256:
                    raise ValueError("Input image changed during execution")
            probability_path = directory / "probability.nii.gz"
            temporary = directory / ".probability.nii.gz"
            output_image = nib.Nifti1Image(probability, swi.image.affine)
            output_image.header.set_xyzt_units("mm")
            nib.save(output_image, temporary)
            os.chmod(temporary, 0o600)
            os.replace(temporary, probability_path)
            atomic_json(directory / "candidates.json", {"schema_version": 1,
                        "study_id": study.id, "candidates": candidates})
            status.update(status="complete", candidate_count=len(candidates),
                          completed_at=datetime.now(timezone.utc).isoformat(),
                          artifact_sha256={name: file_hash(directory / name)
                                           for name in ("probability.nii.gz", "candidates.json")})
            atomic_json(status_path, status)
            return status
        except BaseException as error:
            status.update(status="cancelled" if isinstance(error, (KeyboardInterrupt, InterruptedError))
                          else "failed", error_code=type(error).__name__,
                          ended_at=datetime.now(timezone.utc).isoformat())
            atomic_json(status_path, status)
            raise
