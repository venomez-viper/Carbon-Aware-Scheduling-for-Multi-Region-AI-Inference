import numpy as np
import pandas as pd
from config import *
import os

def latency_first(lats, cis, **kwargs):
    return np.argmin(lats)

def carbon_first(lats, cis, **kwargs):
    return np.argmin(cis)

def hybrid_policy(lats, cis, alpha, **kwargs):
    lr = lats.max() - lats.min()
    lr = lr if lr > 0 else 1
    cr = cis.max() - cis.min()
    cr = cr if cr > 0 else 1
    
    norm_l = (lats - lats.min()) / lr
    norm_c = (cis - cis.min()) / cr
    
    scores = alpha * norm_l + (1 - alpha) * norm_c
    return np.argmin(scores)

def constrained_hybrid(lats, cis, slo_threshold, inference_ms, **kwargs):
    total_lats = lats + inference_ms
    eligible = total_lats <= slo_threshold
    
    if eligible.any():
        masked_ci = np.where(eligible, cis, 1e9)
        return np.argmin(masked_ci)
    else:
        return np.argmin(lats)
