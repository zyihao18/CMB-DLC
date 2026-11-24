"""Fixed CNN and VLM score fusion and MARS counts by region and side."""

import math

from .rendering import METHOD
from .vlm import REGION_GROUPS, validate_response

WEIGHTS = {role["id"]: role["weight"] for role in METHOD["roles"]}


def unit_score(value: float, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return float(value)


def fuse_scores(detector_score: float, role_scores: dict[str, float]) -> float:
    """Combine the supplied, available roles. A missing response is handled before fusion."""
    if not role_scores or not role_scores.keys() <= WEIGHTS.keys():
        raise ValueError("Supply scores for at least one recognized assessment role")
    detector = unit_score(detector_score, "detector score")
    weight = math.fsum(WEIGHTS[r] for r in role_scores)
    review = math.fsum(WEIGHTS[r] * unit_score(v, "role score") for r, v in role_scores.items()) / weight
    alpha = METHOD["cnn_weight"]
    return alpha * detector + (1 - alpha) * review


def summarize(payload: dict, threshold: float) -> dict:
    """Validate model responses, retain findings at the threshold and tally resolved locations."""
    threshold = unit_score(threshold, "threshold")
    if (not isinstance(payload, dict) or set(payload) != {"schema_version", "available_roles", "candidates"}
            or type(payload["schema_version"]) is not int or payload["schema_version"] != 1):
        raise ValueError("Expected a version 1 scoring input with available_roles and candidates")
    roles = payload["available_roles"]
    if (not isinstance(roles, list) or not roles or any(r not in WEIGHTS for r in roles)
            or len(set(roles)) != len(roles)):
        raise ValueError("available_roles must list distinct configured assessment roles")
    if not isinstance(payload["candidates"], list):
        raise ValueError("candidates must be a list")

    counts = {region: {"left": 0, "right": 0} for region in REGION_GROUPS}
    findings = []
    seen = set()
    for item in payload["candidates"]:
        required = {"candidate_id", "detector_score", "assessments"}
        if (not isinstance(item, dict) or not required <= item.keys()
                or not item.keys() <= required | {"localization"}):
            raise ValueError("A candidate requires an ID, a detector score and assessment responses")
        key = item["candidate_id"]
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError("Candidate IDs must be nonempty and unique")
        seen.add(key)
        detector = unit_score(item["detector_score"], "detector score")
        assessments = item["assessments"]
        if not isinstance(assessments, dict) or not assessments.keys() <= set(roles):
            raise ValueError("Candidate assessment roles must come from available_roles")
        responses = {role: validate_response("assessment", value) for role, value in assessments.items()}
        loc = item.get("localization")
        if loc is not None:
            loc = validate_response("localization", loc)

        result = {"candidate_id": key, "score": None, "counted": False,
                  "limited_visibility_roles": [role for role, value in responses.items()
                                               if value["target_visible"] != "yes"]}
        if responses.keys() != set(roles):
            result["status"] = "pending_assessment"
        else:
            score = fuse_scores(detector, {k: v["real_score"] for k, v in responses.items()})
            result.update(score=score, status="retained" if score >= threshold else "excluded")
            if result["status"] == "retained" and loc is not None:
                region, side = loc["mars_region"], loc["hemisphere"]
                result.update(mars_region=region, hemisphere=side)
                if loc["localization_status"] == "localized" and side in ("left", "right"):
                    counts[region][side] += 1
                    result["counted"] = True
        findings.append(result)
    retained = sum(row["status"] == "retained" for row in findings)
    counted = sum(row["counted"] for row in findings)
    return {
        "schema_version": 1,
        "threshold": threshold,
        "available_roles": roles,
        "retained": retained,
        "counted": counted,
        "retained_with_unresolved_location": retained - counted,
        "regional_counts": counts,
        "candidates": findings,
    }
