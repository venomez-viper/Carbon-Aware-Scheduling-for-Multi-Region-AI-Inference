"""
Configuration settings for the Carbon-Aware AI Inference Simulator.
Researchers can modify these global variables to test different hypotheses,
add new regions, or simulate alternative workloads.
"""
from typing import Dict, List, Any

# Region configuration with base carbon intensity (gCO2/kWh).
# Researchers: Add new regions here and update LATENCY_MATRIX accordingly.
REGIONS: Dict[str, Dict[str, float]] = {
    'us-east': {'base_carbon': 400.0},
    'us-west': {'base_carbon': 250.0},
    'europe': {'base_carbon': 150.0},
    'asia': {'base_carbon': 500.0}
}
REGION_NAMES: List[str] = list(REGIONS.keys())
NUM_REGIONS: int = len(REGIONS)

# User location distributions simulating the origin of requests.
USER_LOCATIONS: List[str] = ['North_America', 'Europe', 'Asia']
USER_PROBS: List[float] = [0.5, 0.3, 0.2]  # Must sum to 1.0

# Latency matrix (measured in ms) from each user location to each region.
# Rows map to USER_LOCATIONS, columns map to REGION_NAMES.
LATENCY_MATRIX: Dict[str, List[int]] = {
    'North_America': [20, 60, 100, 200],
    'Europe': [100, 150, 20, 150],
    'Asia': [200, 150, 150, 20]
}

# Workload configurations for varying ML model characteristics.
# means and stds are in ms; slo is the Service Level Objective latency threshold (ms).
WORKLOADS: Dict[str, Dict[str, Any]] = {
    'BERT-base': {'mean_lat': 50, 'std_lat': 10, 'slo': 100, 'mix_prob': 0.6},
    'BERT-large': {'mean_lat': 120, 'std_lat': 20, 'slo': 200, 'mix_prob': 0.3},
    'ResNet-50': {'mean_lat': 30, 'std_lat': 5, 'slo': 80, 'mix_prob': 0.1}
}
WORKLOAD_NAMES: List[str] = list(WORKLOADS.keys())
WORKLOAD_PROBS: List[float] = [WORKLOADS[w]['mix_prob'] for w in WORKLOAD_NAMES]

# Global Simulation Settings
SIM_HOURS: int = 168        # Duration of simulation (e.g., 168 hours = 7 days)
REQS_PER_HOUR: int = 1000   # Traffic volume generated each hour
SEED: int = 42              # Random seed for determinism and reproducible research
JITTER: float = 2.0         # Small positive network jitter (ms) applied to total latency
