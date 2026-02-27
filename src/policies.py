import numpy as np
from config import (
    LATENCY_GLOBAL_MIN, LATENCY_GLOBAL_MAX,
    CARBON_GLOBAL_MIN, CARBON_GLOBAL_MAX
)


def latency_first(lats, cis, **kwargs):
    """Policy 1: Route to lowest-latency region. Ignores carbon."""
    return np.argmin(lats)


def carbon_first(lats, cis, **kwargs):
    """Policy 2: Route to lowest carbon intensity region. Ignores latency."""
    return np.argmin(cis)


def hybrid_policy(lats, cis, alpha, **kwargs):
    """
    Policy 3: Weighted combination of latency and carbon.
    Uses GLOBAL min-max normalization so alpha is stable across all requests.
    alpha=1.0 → pure latency-first
    alpha=0.0 → pure carbon-first
    """
    norm_l = (lats - LATENCY_GLOBAL_MIN) / (LATENCY_GLOBAL_MAX - LATENCY_GLOBAL_MIN)
    norm_c = (cis - CARBON_GLOBAL_MIN) / (CARBON_GLOBAL_MAX - CARBON_GLOBAL_MIN)

    # Clip to [0, 1] to handle edge cases
    norm_l = np.clip(norm_l, 0, 1)
    norm_c = np.clip(norm_c, 0, 1)

    scores = alpha * norm_l + (1 - alpha) * norm_c
    return np.argmin(scores)


def constrained_hybrid(lats, cis, slo_threshold, inference_ms, jitter_buffer=9, **kwargs):
    """
    Policy 4: SLO-constrained carbon optimization.
    Filters to regions where (RTT + inference + jitter_buffer) <= SLO,
    then picks the lowest carbon among eligible regions.
    Falls back to minimum latency if no region qualifies.
    jitter_buffer=9ms = 3 x std_dev(3ms) for 3-sigma jitter protection.
    """
    total_lats = lats + inference_ms + jitter_buffer
    eligible = total_lats <= slo_threshold

    if eligible.any():
        masked_ci = np.where(eligible, cis, 1e9)
        return np.argmin(masked_ci)
    else:
        return np.argmin(lats)
