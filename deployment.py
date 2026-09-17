#!/usr/bin/env python3
"""The configuration to run the model on real, unlabelled boards.

Benchmarks and deployment want different operating points, and until now only
the benchmark one was reachable. `fit_thresholds.py` searched for a better set
on 2026-09-05 and found one that survived a held-out half of validation
(+0.0105 F1), but nothing outside `reevaluate.py` ever applied it, so every
prediction run still used values inherited from Model V5.

Measured on 16 real boards where the model was reported as failing:

    score band     shipped   this profile
    0.2 - 0.4          581             13
    0.6 - 1.0          187            194
    total            1,074            299

54% of all detections sat in the 0.2-0.4 band, and they were duplicates --
a spurious low-confidence `Rectangle` on top of pads already found correctly.
The class mix read 80.4% Rectangle against 39.7% in the training data. This
profile removes 98% of that band while leaving the confident detections alone
(they go slightly *up*, because instances no longer fragment against each
other).

**Why this is not simply the better configuration.** On the labelled test split
these thresholds score marginally *worse* -- mask mAP50-95 0.7710 -> 0.7694 --
because they buy precision with recall, and mask AP is rank-aware: a weak
duplicate that ranks last costs a benchmark almost nothing. Unlabelled
deployment has no ranking to hide behind; every retained instance is drawn and
counted. Same model, genuinely different right answer. Keep the shipped values
for anything whose number gets published, and this for anything that looks at
a real board.

    from deployment import apply_deployment_profile
    applied = apply_deployment_profile(model_module)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
# V7 only. The V6.2 fit lives beside this as threshold_fit_v6_legacy.json and is
# deliberately NOT loaded: two of its five values (center_confidence,
# center_nms_radius) set attributes V7's decoder never reads, and the other three
# were fitted against the centre/offset grouping rule, not this one. Applying it
# would succeed silently and calibrate nothing -- the worst failure mode, because
# it looks like a configured deployment.
THRESHOLD_FIT = HERE / "results" / "threshold_fit_v7.json"
LEGACY_FIT = HERE / "results" / "threshold_fit_v6_legacy.json"

# results/threshold_fit_v7.json -> model_v7 module attribute. These are the six
# controls V7's decoder actually reads; `fit_thresholds.py` searches exactly
# this set.
THRESHOLD_ATTRIBUTES = {
    "semantic_confidence": "SEMANTIC_CONFIDENCE_THRESHOLD",
    "inner_distance_core_threshold": "INNER_DISTANCE_CORE_THRESHOLD",
    "minimum_core_area": "MIN_INNER_DISTANCE_CORE_AREA",
    "minimum_instance_area": "MIN_INSTANCE_AREA",
    "boundary_confidence": "BOUNDARY_CONFIDENCE_THRESHOLD",
    "minimum_confidence": "DEPLOYMENT_MIN_CONFIDENCE",
}

# Instances recovered without a centre peak score FALLBACK_INSTANCE_SCORE
# (0.05) and survive the fitted thresholds, because those act on centre
# evidence this path has none of. On the same 16 boards they were 24% of what
# remained. 0.50 clears them without touching the 0.6-1.0 mass where every
# correct detection sits; the measured gap between the two populations is wide
# enough that the exact value is not delicate.
DEFAULT_MIN_CONFIDENCE = 0.50


def apply_deployment_profile(module, *, minimum_confidence=None, verbose=True):
    """Point `module` at the deployment operating point. Returns what changed.

    Set PCB_DEPLOYMENT_PROFILE=0 to opt out and keep the shipped thresholds,
    or PCB_MIN_CONFIDENCE to override the floor.
    """
    if os.environ.get("PCB_DEPLOYMENT_PROFILE", "1").strip().lower() in (
        "0", "false", "no",
    ):
        if verbose:
            print("thresholds : shipped (deployment profile disabled)")
        return None

    if minimum_confidence is None:
        minimum_confidence = float(
            os.environ.get("PCB_MIN_CONFIDENCE", DEFAULT_MIN_CONFIDENCE)
        )

    applied: dict[str, object] = {}
    if THRESHOLD_FIT.is_file():
        values = json.loads(THRESHOLD_FIT.read_text())["fitted"]["values"]
        for key, attribute in THRESHOLD_ATTRIBUTES.items():
            if key in values:
                setattr(module, attribute, values[key])
                applied[key] = values[key]
    elif verbose:
        # Not fatal, but say it loudly. A V7 deployment with no V7 fit is running
        # the shipped defaults, and on this project a quiet run has been mistaken
        # for a configured one before.
        print(f"WARNING: {THRESHOLD_FIT.name} missing -- V7 thresholds NOT "
              "applied; only the confidence floor is active.")
        if LEGACY_FIT.is_file():
            print("         (a V6.2 fit is present and is deliberately ignored; "
                  "run ./run.sh fit-thresholds against a V7 checkpoint)")

    module.DEPLOYMENT_MIN_CONFIDENCE = float(minimum_confidence)
    applied["minimum_confidence"] = float(minimum_confidence)

    if verbose:
        print(f"profile    : deployment -> {json.dumps(applied)}")
    return applied
