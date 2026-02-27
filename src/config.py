import numpy as np
import pandas as pd

WORKLOADS = {
    "bert_base": {
        "name": "BERT-base Text Classification",
        "model": "BERT-base",
        "inference_mean_ms": 18,
        "inference_std_ms": 4,
        "slo_threshold_ms": 100,
        "probability": 0.60,
        "description": "Real-time text classification",
    },
    "bert_large": {
        "name": "BERT-large Question Answering",
        "model": "BERT-large",
        "inference_mean_ms": 60,
        "inference_std_ms": 12,
        "slo_threshold_ms": 150,
        "probability": 0.30,
        "description": "Q&A system",
    },
    "resnet50": {
        "name": "ResNet-50 Image Embedding",
        "model": "ResNet-50",
        "inference_mean_ms": 12,
        "inference_std_ms": 3,
        "slo_threshold_ms": 80,
        "probability": 0.10,
        "description": "Image similarity search",
    },
}

_prob_sum = sum(w["probability"] for w in WORKLOADS.values())
assert abs(_prob_sum - 1.0) < 0.001, f"Workload prob must sum to 1.0"

REGIONS = ['US-East', 'US-West', 'EU-West', 'EU-North', 'Singapore']

BASE_CARBON_INTENSITY = {
    'US-East': 323,
    'US-West': 79,
    'EU-West': 276,
    'EU-North': 25,
    'Singapore': 367,
}

USER_LOCATIONS = ['US-East', 'US-West', 'EU', 'Asia']

LATENCY_DATA = {
    'US-East':    [5,   65,  85,  230],
    'US-West':    [65,  5,   145, 170],
    'EU-West':    [85,  145, 10,  165],
    'EU-North':   [95,  155, 25,  175],
    'Singapore':  [230, 170, 165, 5],
}

LATENCY_MATRIX = pd.DataFrame(LATENCY_DATA, index=USER_LOCATIONS)

SIMULATION_HOURS = 168
REQUESTS_PER_HOUR = 200

USER_DISTRIBUTION = {
    'US-East': 0.40,
    'US-West': 0.20,
    'EU': 0.25,
    'Asia': 0.15,
}

NETWORK_JITTER_MEAN = 0
NETWORK_JITTER_STD = 3
RANDOM_SEED = 42

CARBON_DIURNAL_AMPLITUDE = 0.2
CARBON_RANDOM_NOISE_RANGE = 0.1

HYBRID_ALPHA_VALUES = [0.2, 0.3, 0.5, 0.7]

# Global normalization bounds for hybrid policy scoring
# Used to ensure alpha is a stable, consistent weight across all requests
LATENCY_GLOBAL_MIN = 5      # Minimum RTT in latency matrix (intra-region)
LATENCY_GLOBAL_MAX = 230    # Maximum RTT in latency matrix (US-East to Singapore)
CARBON_GLOBAL_MIN = 5       # Floor value set in generate_carbon_traces()
CARBON_GLOBAL_MAX = 450     # Approx max: Singapore(367) x 1.2 diurnal amplitude


def get_workload_list():
    return list(WORKLOADS.keys())

def get_workload_probabilities():
    return [WORKLOADS[w]["probability"] for w in get_workload_list()]

def sample_inference_time(workload_id, rng=None):
    if rng is None: rng = np.random
    workload = WORKLOADS[workload_id]
    sample = rng.normal(workload["inference_mean_ms"], workload["inference_std_ms"])
    return max(1.0, sample)

def get_slo_threshold(workload_id):
    return WORKLOADS[workload_id]["slo_threshold_ms"]

def print_config_summary():
    print("=" * 60)
    print("SIMULATION CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Duration : {SIMULATION_HOURS} hours ({SIMULATION_HOURS//24} days)")
    print(f"Requests : {REQUESTS_PER_HOUR} req/hr → {SIMULATION_HOURS * REQUESTS_PER_HOUR} total")
    print(f"Regions  : {REGIONS}")
    print(f"Alpha    : {HYBRID_ALPHA_VALUES}")
    print(f"Seed     : {RANDOM_SEED}")
    print("-" * 60)
    for wid, w in WORKLOADS.items():
        print(f"  {wid:12s} | Inference N({w['inference_mean_ms']},{w['inference_std_ms']})ms "
              f"| SLO {w['slo_threshold_ms']}ms | {int(w['probability']*100)}% traffic")
    print("=" * 60)

