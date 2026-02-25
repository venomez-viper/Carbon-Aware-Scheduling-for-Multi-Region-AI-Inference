import os
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from config import (
    REGIONS, REGION_NAMES, USER_LOCATIONS, USER_PROBS,
    LATENCY_MATRIX, WORKLOADS, WORKLOAD_NAMES, WORKLOAD_PROBS,
    SIM_HOURS, REQS_PER_HOUR, SEED, JITTER
)
from policies import POLICIES

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CarbonSimulator")

def generate_carbon_traces(seed: int = SEED, sim_hours: int = SIM_HOURS) -> pd.DataFrame:
    """
    Generates synthetic hourly carbon intensity traces for each region.
    Uses a diurnal sine wave + random noise to simulate daily fluctuations.
    """
    logger.info("Generating carbon intensity traces...")
    np.random.seed(seed)
    
    traces = {}
    time_steps = np.arange(sim_hours)
    
    for region in REGION_NAMES:
        base = REGIONS[region]['base_carbon']
        # Simulated diurnal cycle: amplitude is 20% of base carbon, 24h period
        diurnal_cycle = 0.2 * base * np.sin(2 * np.pi * time_steps / 24)
        noise = np.random.normal(0, 0.05 * base, sim_hours)
        
        # Ensure carbon intensity is non-negative
        trace = np.clip(base + diurnal_cycle + noise, a_min=0, a_max=None)
        traces[region] = trace
        
    df = pd.DataFrame(traces)
    df.index.name = 'hour'
    
    out_path = Path('../data/carbon/carbon_intensity_traces.csv')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    
    return df

def export_latency_matrix() -> Dict[str, List[int]]:
    """Exports the configuration latency matrix to CSV for reference."""
    logger.info("Exporting latency matrix...")
    df = pd.DataFrame(LATENCY_MATRIX, index=REGION_NAMES).T
    df.index.name = 'user_location'
    
    out_path = Path('../data/latency/latency_matrix.csv')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    
    return LATENCY_MATRIX

def run_policy_simulation(
    policy_name: str, 
    policy_fn: callable, 
    carbon_traces: pd.DataFrame, 
    user_loc_indices: np.ndarray, 
    workload_indices: np.ndarray,
    sim_hours: int = SIM_HOURS,
    reqs_per_hour: int = REQS_PER_HOUR
) -> tuple:
    """
    Simulates a given routing policy over the synthesized workload requests.
    Returns global metrics dict and a list of workload-specific metrics dicts.
    """
    logger.info(f"Evaluating policy: {policy_name}")
    
    latencies_log = []
    slo_violations = 0
    total_carbon = 0.0
    region_counts = {r: 0 for r in REGION_NAMES}
    total_reqs = len(user_loc_indices)
    
    # Per-workload tracking
    wl_stats = {w: {'latencies': [], 'slo_violations': 0, 'count': 0} for w in WORKLOAD_NAMES}
    
    req_idx = 0
    for hour in range(sim_hours):
        current_carbons = carbon_traces.iloc[hour].values
        
        for _ in range(reqs_per_hour):
            u_idx = user_loc_indices[req_idx]
            w_idx = workload_indices[req_idx]
            req_idx += 1
            
            user_loc = USER_LOCATIONS[u_idx]
            workload_name = WORKLOAD_NAMES[w_idx]
            workload_cfg = WORKLOADS[workload_name]
            
            # Retrieve latency options for this user location
            net_lats = LATENCY_MATRIX[user_loc]
            
            # Compute policy choice
            chosen_region_idx = policy_fn(
                latencies=net_lats,
                carbons=current_carbons,
                slos=workload_cfg['slo'],
                mean_inference=workload_cfg['mean_lat']
            )
            
            chosen_region = REGION_NAMES[chosen_region_idx]
            region_counts[chosen_region] += 1
            
            # Compute request timing
            inf_time = np.random.normal(workload_cfg['mean_lat'], workload_cfg['std_lat'])
            inf_time = max(0.0, inf_time)
            
            total_latency = net_lats[chosen_region_idx] + inf_time + JITTER
            latencies_log.append(total_latency)
            wl_stats[workload_name]['latencies'].append(total_latency)
            wl_stats[workload_name]['count'] += 1
            
            if total_latency > workload_cfg['slo']:
                slo_violations += 1
                wl_stats[workload_name]['slo_violations'] += 1
                
            total_carbon += current_carbons[chosen_region_idx]
            
    # Compile global performance metrics
    metrics = {
        'Policy': policy_name,
        'Avg_Latency_ms': np.mean(latencies_log),
        'P95_Latency_ms': np.percentile(latencies_log, 95),
        'SLO_Violation_Rate_%': (slo_violations / total_reqs) * 100,
        'Avg_Carbon_gCO2': total_carbon / total_reqs,
    }
    
    # Track regional distribution
    for r in REGION_NAMES:
        metrics[f'Region_{r}_%'] = (region_counts[r] / total_reqs) * 100
        
    # Compile per-workload performance metrics
    workload_metrics = []
    for w_name in WORKLOAD_NAMES:
        count = wl_stats[w_name]['count']
        lats = wl_stats[w_name]['latencies']
        viols = wl_stats[w_name]['slo_violations']
        workload_metrics.append({
            'Policy': policy_name,
            'Workload': w_name,
            'SLO_Target_ms': WORKLOADS[w_name]['slo'],
            'Avg_Latency_ms': np.mean(lats) if count > 0 else 0,
            'P95_Latency_ms': np.percentile(lats, 95) if count > 0 else 0,
            'SLO_Violation_Rate_%': (viols / count * 100) if count > 0 else 0
        })
        
    return metrics, workload_metrics

def main():
    parser = argparse.ArgumentParser(description="Carbon-aware Multi-region AI Inference Simulator")
    parser.add_argument('--sim-hours', type=int, default=SIM_HOURS, 
                        help="Duration of the simulation in hours.")
    parser.add_argument('--reqs-per-hour', type=int, default=REQS_PER_HOUR, 
                        help="Number of inference requests per hour.")
    parser.add_argument('--seed', type=int, default=SEED, 
                        help="Random seed for reproducibility.")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    # 1. Setup environment and trace data
    carbon_traces = generate_carbon_traces(seed=args.seed, sim_hours=args.sim_hours)
    export_latency_matrix()
    
    # 2. Pre-generate all incoming traffic (to ensure fair policy comparison)
    total_reqs = args.sim_hours * args.reqs_per_hour
    user_loc_indices = np.random.choice(len(USER_LOCATIONS), size=total_reqs, p=USER_PROBS)
    workload_indices = np.random.choice(len(WORKLOAD_NAMES), size=total_reqs, p=WORKLOAD_PROBS)
    
    logger.info(f"Generated {total_reqs:,} requests over {args.sim_hours} hours.")
    
    # 3. Execution loop across policies
    global_results = []
    all_workload_results = []
    baseline_carbon = None
    
    for name, fn in POLICIES.items():
        metrics, workload_metrics = run_policy_simulation(
            policy_name=name, 
            policy_fn=fn, 
            carbon_traces=carbon_traces, 
            user_loc_indices=user_loc_indices, 
            workload_indices=workload_indices,
            sim_hours=args.sim_hours,
            reqs_per_hour=args.reqs_per_hour
        )
        
        if name == 'Latency-First':
            baseline_carbon = metrics['Avg_Carbon_gCO2']
            metrics['Carbon_Reduction_%'] = 0.0
        else:
            reduction = ((baseline_carbon - metrics['Avg_Carbon_gCO2']) / baseline_carbon) * 100
            metrics['Carbon_Reduction_%'] = reduction
            
        global_results.append(metrics)
        all_workload_results.extend(workload_metrics)
        
    # 4. Save and report outputs
    df_global = pd.DataFrame(global_results)
    df_workload = pd.DataFrame(all_workload_results)
    
    out_dir = Path('../outputs/tables')
    out_dir.mkdir(parents=True, exist_ok=True)
    df_global.to_csv(out_dir / 'simulation_results.csv', index=False)
    df_workload.to_csv(out_dir / 'workload_results.csv', index=False)
    
    logger.info("Simulation complete. Global Results:\n")
    print(df_global[['Policy', 'Avg_Latency_ms', 'P95_Latency_ms', 
                      'SLO_Violation_Rate_%', 'Avg_Carbon_gCO2', 'Carbon_Reduction_%']].to_string(index=False))

if __name__ == '__main__':
    main()
