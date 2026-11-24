"""Write fabricated responses for the scoring example. No scan or model is used."""

import json
from pathlib import Path
import sys

from cmb_dlc.scoring import WEIGHTS


def assessment(score: float) -> dict:
    return {
        "target_visible": "yes",
        "classification": "real_lesion" if score >= .5 else "not_lesion",
        "real_score": score,
        "confidence": .8,
        "mimic_risk": "none",
        "evidence": ["Synthetic example"],
        "concise_rationale": "Generated for the offline demonstration",
    }


def example() -> dict:
    candidates = [
        {"candidate_id": f"synthetic-{i}", "detector_score": detector,
         "assessments": {role: assessment(score) for role in WEIGHTS}}
        for i, (detector, score) in enumerate(((.9, .85), (.8, .7), (.7, .15)), 1)
    ]
    candidates[0]["localization"] = {
        "marker_visible": "yes", "localization_status": "localized", "hemisphere": "left",
        "mars_region": "frontal", "broad_compartment": "lobar", "region_confidence": .9,
        "alternative_region": "none", "landmark_evidence": ["Synthetic example"],
        "concise_rationale": "Generated for the offline demonstration",
    }
    return {"schema_version": 1, "available_roles": list(WEIGHTS), "candidates": candidates}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python examples/synthetic_scores.py OUTPUT.json")
    with Path(sys.argv[1]).open("x") as dest:
        json.dump(example(), dest, indent=2)
        dest.write("\n")
