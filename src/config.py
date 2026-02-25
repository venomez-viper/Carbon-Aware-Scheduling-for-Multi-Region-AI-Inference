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
    pass
