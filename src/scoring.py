"""
Scientific validation (Layer 6): turn the vetting tests and Layer-4 features
into a physical-plausibility verdict and a single auditable confidence
score, so a candidate with a low-confidence or implausible "planet" label
gets flagged for human review rather than reported as a discovery.

This is a transparent, rule-based scorer, not a trained model -- no labeled
training set is available yet (see README "Planned Enhancement"). Every
term below is one inspectable number so the score stays explainable, and
the function signature (candidate + features + classification in, a score
+ breakdown out) is what a learned uncertainty head (Layer 5) would later
replace without touching main.py.
"""

import numpy as np

# Duration/period ratio above which a transit geometry is no longer
# physically sensible for any plausible impact parameter or stellar radius.
MAX_DUTY_CYCLE = 0.2
MAX_PLAUSIBLE_DEPTH_PPM = 1e5  # >10% depth: almost certainly EB/blend, not a planet
MIN_CLEAN_TRANSITS = 2


def physical_plausibility(candidate, features):
    """Sanity-check recovered parameters against basic transit physics."""
    period_days = candidate["period_days"]
    duration_hours = candidate["duration_hours"]
    depth_ppm = candidate["depth_ppm"]

    flags = []

    duty_cycle = (duration_hours / 24) / period_days
    if duty_cycle > MAX_DUTY_CYCLE:
        flags.append(f"duration is {duty_cycle:.0%} of the period (unphysically long transit)")

    if depth_ppm <= 0:
        flags.append("non-positive transit depth")
    elif depth_ppm > MAX_PLAUSIBLE_DEPTH_PPM:
        flags.append("depth > 10% of stellar flux -- likely an eclipsing binary or blend, not a planet")

    consistency = features.get("depth_consistency_frac")
    if consistency is not None and consistency > 0.5:
        flags.append(f"per-transit depths vary by {consistency:.0%} -- inconsistent with a stable transit")

    if features.get("n_transits_measured", 0) < MIN_CLEAN_TRANSITS:
        flags.append("fewer than 2 clean transits observed -- period/ephemeris unconfirmed")

    return {"plausible": len(flags) == 0, "flags": flags}


def confidence_score(candidate, features, classification, plausibility):
    """
    Combine detection significance, vetting flags, and plausibility checks
    into a 0-1 confidence score plus a human-readable breakdown.
    """
    snr = candidate["detection_snr"]
    # SNR 7 is the classical "likely real" detection threshold used by the
    # Kepler/TESS pipelines; scale so SNR<=7 stays below 0.5 and it
    # saturates towards 1 well above that.
    snr_term = float(np.clip(snr / 20, 0.0, 1.0))

    consistency = features.get("depth_consistency_frac")
    consistency_term = float(np.clip(1.0 - consistency, 0.0, 1.0)) if consistency is not None else 0.5

    vetting_penalty = 0.15 * len(classification["flags"])
    plausibility_penalty = 0.2 * len(plausibility["flags"])

    raw_score = 0.6 * snr_term + 0.4 * consistency_term
    score = float(np.clip(raw_score - vetting_penalty - plausibility_penalty, 0.0, 1.0))

    # Implausibility already costs `plausibility_penalty` points above; it
    # shouldn't *also* gate every tier below the top one, or a single edge
    # case (e.g. noisy per-transit depths on an otherwise strong detection)
    # gets dumped straight into "false positive" instead of "needs review".
    if score >= 0.7 and plausibility["plausible"] and classification["label"] == "candidate_planetary_transit":
        verdict = "high_confidence_candidate"
    elif score >= 0.35:
        verdict = "requires_human_review"
    else:
        verdict = "low_confidence_likely_false_positive"

    return {
        "score": score,
        "verdict": verdict,
        "breakdown": {
            "snr_term": snr_term,
            "depth_consistency_term": consistency_term,
            "vetting_flag_penalty": vetting_penalty,
            "plausibility_flag_penalty": plausibility_penalty,
        },
    }
