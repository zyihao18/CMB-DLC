"""Image-only prompts and model-response validation."""
from __future__ import annotations

import json
import math

from .rendering import METHOD

REGION_GROUPS = {"brainstem": "infratentorial", "cerebellum": "infratentorial",
                 "basal_ganglia": "deep", "thalamus": "deep", "internal_capsule": "deep",
                 "external_capsule": "deep", "corpus_callosum": "deep", "deep_periventricular_white_matter": "deep",
                 "frontal": "lobar", "parietal": "lobar", "temporal": "lobar", "occipital": "lobar", "insula": "lobar"}


def prompts(kind: str, role: dict | None = None) -> tuple[str, str]:
    if kind == "assessment":
        system = "You review one detector-proposed cerebral microbleed target using only the supplied de-identified images. A marker and detector response are hypotheses, not proof. Assess only the requested evidence. Preserve visibility limits and uncertainty. Return one JSON object; no hidden reasoning or diagnostic/treatment advice."
        user = f"""Prompt version: {METHOD['prompt_version']}
Focused assessment: {role['id']}
Question: {role['purpose']}
Rows: {', '.join(role['rows'])}. Columns: {role['slices']} adjacent true-cardinal RAS slices centered on the target. Whole-volume detector projections are the same projection across columns; all overlaid peaks are unconfirmed detector hypotheses. The hollow red marker identifies the target; blue points identify other detector proposals. Orientation letters refer to the patient. Scanner minIP has its own slab extent; images are aligned by physical coordinates, not equal original slice indices.
Check: {json.dumps(role['focus'])}
Return JSON with exactly these fields:
{{"target_visible":"yes|no|uncertain","classification":"real_lesion|possible_lesion|not_lesion","real_score":0.0,"confidence":0.0,"mimic_risk":"none|vessel|edge_artifact|phase_artifact|distribution_artifact|other|uncertain","evidence":["brief visible observation"],"concise_rationale":"short evidence note"}}
Scores must be numbers in [0,1], not strings. Do not infer patient identity, labels, counts, or clinical certainty from presentation. real_score is compatibility, not calibrated probability. If the target is not assessable, preserve no/uncertain visibility and state the limitation in the brief evidence note."""
    elif kind == "localization":
        system = "You perform anatomical localization only, using the marked target and image anatomy. Do not decide lesion identity, receive prior recognition outputs, or estimate patient burden. Use MARS taxonomy and preserve uncertain region/side. Return one JSON object only with concise visible anatomical evidence."
        user = f"""Prompt version: image_only_mars_v1. Input variant: triplanar_swi_magnitude.
Top row: SWI axial/coronal/sagittal. Bottom row: magnitude axial/coronal/sagittal. Images are true-cardinal neurological RAS with physical aspect ratios and patient orientation letters. The hollow red marker identifies the target in each panel. Use image landmarks and orientation letters, not assumptions from scanner array indices.
Allowed MARS regions and compartments: {json.dumps(REGION_GROUPS)}.
Return JSON exactly:
{{"marker_visible":"yes|no|uncertain","localization_status":"localized|uncertain|not_assessable","hemisphere":"left|right|midline_or_uncertain","mars_region":"{'|'.join(REGION_GROUPS)}|unassigned","broad_compartment":"infratentorial|deep|lobar|unassigned","region_confidence":0.0,"alternative_region":"{'|'.join(REGION_GROUPS)}|none","landmark_evidence":["short anatomical observation"],"concise_rationale":"short location-only note"}}
localized requires a named region and matching compartment. not_assessable requires unassigned region and compartment. Preserve midline_or_uncertain instead of forcing a side. Confidence is a number in [0,1]. Do not mention recognition classifications, human counts, source names or patient identity."""
    else:
        raise ValueError("Unknown model task")
    return system, user


def validate_response(kind: str, value: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError("Response must be an object")
    if kind == "assessment":
        expected = {"target_visible", "classification", "real_score", "confidence", "mimic_risk", "evidence", "concise_rationale"}
        if set(value) != expected or value["target_visible"] not in ("yes", "no", "uncertain"):
            raise ValueError("Invalid assessment fields or visibility")
        if value["classification"] not in ("real_lesion", "possible_lesion", "not_lesion"):
            raise ValueError("Invalid assessment class")
        if value["mimic_risk"] not in ("none", "vessel", "edge_artifact", "phase_artifact", "distribution_artifact", "other", "uncertain"):
            raise ValueError("Invalid mimic risk")
        numeric, evidence = ("real_score", "confidence"), "evidence"
    elif kind == "localization":
        expected = {"marker_visible", "localization_status", "hemisphere", "mars_region", "broad_compartment", "region_confidence", "alternative_region", "landmark_evidence", "concise_rationale"}
        if set(value) != expected or value["marker_visible"] not in ("yes", "no", "uncertain"):
            raise ValueError("Invalid localization fields or visibility")
        if value["localization_status"] not in ("localized", "uncertain", "not_assessable") or value["hemisphere"] not in ("left", "right", "midline_or_uncertain"):
            raise ValueError("Invalid localization status or side")
        region, compartment = value["mars_region"], value["broad_compartment"]
        if region not in (*REGION_GROUPS, "unassigned") or value["alternative_region"] not in (*REGION_GROUPS, "none"):
            raise ValueError("Invalid MARS region")
        if compartment != REGION_GROUPS.get(region, "unassigned"):
            raise ValueError("MARS region and compartment disagree")
        if value["localization_status"] == "localized" and (region == "unassigned" or value["marker_visible"] != "yes"):
            raise ValueError("Localized target requires visible named anatomy")
        if value["localization_status"] == "not_assessable" and region != "unassigned":
            raise ValueError("Unassessable target cannot have assigned anatomy")
        numeric, evidence = ("region_confidence",), "landmark_evidence"
    else:
        raise ValueError("Unknown model task")
    for key in numeric:
        if type(value[key]) not in (int, float) or not math.isfinite(value[key]) or not 0 <= value[key] <= 1:
            raise ValueError("Invalid finite unit score")
    if not isinstance(value[evidence], list) or not 1 <= len(value[evidence]) <= 6 or any(not isinstance(x, str) or not x.strip() or len(x) > 600 for x in value[evidence]):
        raise ValueError("Invalid concise evidence list")
    if not isinstance(value["concise_rationale"], str) or not value["concise_rationale"].strip() or len(value["concise_rationale"]) > 1200:
        raise ValueError("Invalid concise rationale")
    return value
