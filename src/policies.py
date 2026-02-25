"""
Routing Policies for Carbon-Aware Simulation.
Researchers can define custom scheduling heuristics here and add them 
to the `POLICIES` registry at the bottom of the file to be evaluated.
"""

import numpy as np
from typing import Callable, Dict, Any, Optional

def latency_first(latencies: list | np.ndarray, carbons: list | np.ndarray, **kwargs: Any) -> int:
    """
    Baseline policy: Choose the region with strictly the minimum network latency.
    Ignores carbon intensity entirely.
    """
    return int(np.argmin(latencies))

def carbon_first(latencies: list | np.ndarray, carbons: list | np.ndarray, **kwargs: Any) -> int:
    """
    Environmental baseline: Choose the region with the minimum carbon intensity.
    Ignores network latency entirely.
    """
    return int(np.argmin(carbons))

def hybrid_policy(
    latencies: list | np.ndarray, 
    carbons: list | np.ndarray, 
    alpha: float = 0.5, 
    **kwargs: Any
) -> int:
    """
    Combines normalized latency and carbon using a weighted average.
    `alpha` controls the trade-off (1.0 = purely latency, 0.0 = purely carbon).
    """
    lats, carbs = np.array(latencies), np.array(carbons)
    
    # Min-max normalize to bring both arrays to a [0, 1] scale
    min_lat, max_lat = np.min(lats), np.max(lats)
    norm_lat = (lats - min_lat) / (max_lat - min_lat + 1e-9)
    
    min_carb, max_carb = np.min(carbs), np.max(carbs)
    norm_carb = (carbs - min_carb) / (max_carb - min_carb + 1e-9)
    
    # Compute combined scores and return region index with the lowest score
    scores = alpha * norm_lat + (1 - alpha) * norm_carb
    return int(np.argmin(scores))

def constrained_hybrid(
    latencies: list | np.ndarray, 
    carbons: list | np.ndarray, 
    slos: Optional[float] = None, 
    mean_inference: float = 0.0, 
    **kwargs: Any
) -> int:
    """
    Filters out regions that violate the expected SLO threshold.
    Among the remaining valid regions, picks the greenest one.
    If no regions meet the SLO, falls back to Latency-First.
    """
    if slos is None:
        return int(np.argmin(latencies))
        
    total_expected_lat = np.array(latencies) + mean_inference
    meets_slo = total_expected_lat <= slos
    
    if np.any(meets_slo):
        # Pick greenest region amongst those satisfying the condition
        valid_carbons = np.where(meets_slo, carbons, np.inf)
        return int(np.argmin(valid_carbons))
    else:
        # Fallback to absolute lowest latency
        return int(np.argmin(latencies))

# =====================================================================
# POLICY REGISTRY
# Researchers: Register new algorithmic policies here to evaluate them.
# The keys define column names mapping directly to the outputs matrix.
# =====================================================================
POLICIES: Dict[str, Callable] = {
    'Latency-First': latency_first,
    'Carbon-First': carbon_first,
    'Hybrid': lambda latencies, carbons, **k: hybrid_policy(latencies, carbons, alpha=0.5, **k),
    'Constrained-Hybrid': constrained_hybrid
}
