import numpy as np
import pandas as pd
import os
import argparse
from config import *
from policies import latency_first, carbon_first, hybrid_policy, constrained_hybrid

def generate_carbon_traces(hours, seed=RANDOM_SEED):
    np.random.seed(seed)
    carbon = {}
    
    for region in REGIONS:
        base = BASE_CARBON_INTENSITY[region]
        vals = []
        for h in range(hours):
            hour_of_day = h % 24
            diurnal = 1 + CARBON_DIURNAL_AMPLITUDE * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
            noise = np.random.uniform(1 - CARBON_RANDOM_NOISE_RANGE, 1 + CARBON_RANDOM_NOISE_RANGE)
            vals.append(max(5, base * diurnal * noise))
        carbon[region] = np.array(vals)
    return pd.DataFrame(carbon)

def generate_requests(hours, rph, seed=RANDOM_SEED):
    np.random.seed(seed)
    total = rph * hours
    req_hours = np.repeat(np.arange(hours), rph)
    req_users = np.random.choice(
        list(USER_DISTRIBUTION.keys()), size=total, p=list(USER_DISTRIBUTION.values())
    )
    req_workloads = np.random.choice(
        get_workload_list(), size=total, p=get_workload_probabilities()
    )
    return req_hours, req_users, req_workloads

def run_simulation(output_dir='../outputs', hours=SIMULATION_HOURS, rph=REQUESTS_PER_HOUR, seed=RANDOM_SEED):
    os.makedirs(f'{output_dir}/tables', exist_ok=True)
    os.makedirs(f'{output_dir}/data', exist_ok=True)
    
    carbon_df = generate_carbon_traces(hours, seed=seed)
    ci_arr = carbon_df.values
    
    req_hours, req_users, req_workloads = generate_requests(hours, rph, seed=seed)
    total_requests = len(req_hours)
    
    lat_lookup = {}
    for ul in USER_LOCATIONS:
        lat_lookup[ul] = np.array([LATENCY_MATRIX.loc[ul, r] for r in REGIONS])
        
    policy_configs = [
        ('Latency-First', 'latency_first', None),
        ('Carbon-First', 'carbon_first', None),
    ]
    for alpha in HYBRID_ALPHA_VALUES:
        policy_configs.append((f'Hybrid (\u03b1={alpha})', 'hybrid', alpha))
    policy_configs.append(('Constrained Hybrid', 'constrained', None))
    
    results = {}
    detailed_results = {}
    rng = np.random.default_rng(seed)
    
    for label, ptype, alpha in policy_configs:
        latencies = np.zeros(total_requests)
        carbons_out = np.zeros(total_requests)
        inference_times = np.zeros(total_requests)
        region_selections = np.zeros(total_requests, dtype=int)
        
        workload_stats = {wid: {'count': 0, 'latencies': [], 'slo_violations': 0} for wid in get_workload_list()}
        total_slo_violations = 0
        region_counts = np.zeros(len(REGIONS), dtype=int)
        
        for i in range(total_requests):
            h = req_hours[i]
            ul = req_users[i]
            wid = req_workloads[i]
            
            lats = lat_lookup[ul]
            cis = ci_arr[h]
            
            inference_ms = sample_inference_time(wid, rng=rng)
            inference_times[i] = inference_ms
            slo_threshold = get_slo_threshold(wid)
            
            if ptype == 'latency_first':
                idx = latency_first(lats, cis)
            elif ptype == 'carbon_first':
                idx = carbon_first(lats, cis)
            elif ptype == 'hybrid':
                idx = hybrid_policy(lats, cis, alpha)
            elif ptype == 'constrained':
                idx = constrained_hybrid(lats, cis, slo_threshold, inference_ms)
            
            region_selections[i] = idx
            
            net_lat = lats[idx]
            jitter = max(0, rng.normal(NETWORK_JITTER_MEAN, NETWORK_JITTER_STD))
            total_lat = max(1.0, net_lat + inference_ms + jitter)
            
            latencies[i] = total_lat
            carbons_out[i] = cis[idx]
            region_counts[idx] += 1
            
            if total_lat > slo_threshold:
                total_slo_violations += 1
                workload_stats[wid]['slo_violations'] += 1
                
            workload_stats[wid]['count'] += 1
            workload_stats[wid]['latencies'].append(total_lat)
            
        results[label] = {
            'avg_latency': round(np.mean(latencies), 1),
            'p95_latency': round(np.percentile(latencies, 95), 1),
            'slo_violation_pct': round(100 * total_slo_violations / total_requests, 2),
            'avg_carbon': round(np.mean(carbons_out), 1),
            'avg_inference_time': round(np.mean(inference_times), 1),
            'region_dist': {REGIONS[j]: int(region_counts[j]) for j in range(len(REGIONS))},
        }

        detailed_results[label] = {}
        for wid, stats in workload_stats.items():
            if stats['count'] > 0:
                detailed_results[label][wid] = {
                    'count': stats['count'],
                    'avg_latency': np.mean(stats['latencies']),
                    'p95_latency': np.percentile(stats['latencies'], 95),
                    'slo_violation_pct': 100 * stats['slo_violations'] / stats['count'],
                    'slo_threshold': get_slo_threshold(wid),
                }

    baseline_carbon = results['Latency-First']['avg_carbon']
    for label, res in results.items():
        if label == 'Latency-First':
            res['carbon_reduction'] = 0.0
        else:
            red = round(100 * (1 - res['avg_carbon'] / baseline_carbon), 1)
            res['carbon_reduction'] = red
            
    rows = []
    for label, res in results.items():
        rows.append({
            'Policy': label,
            'Avg Latency (ms)': res['avg_latency'],
            'P95 Latency (ms)': res['p95_latency'],
            'SLO Violation Rate (%)': res['slo_violation_pct'],
            'Avg Carbon (gCO2eq/kWh)': res['avg_carbon'],
            'Carbon Reduction': res['carbon_reduction'],
        })
        
    results_df = pd.DataFrame(rows)
    results_df.to_csv(f'{output_dir}/tables/simulation_results.csv', index=False)
    carbon_df.to_csv(f'{output_dir}/data/carbon_intensity_traces.csv', index_label='hour')
    LATENCY_MATRIX.to_csv(f'{output_dir}/data/latency_matrix.csv')
    
    workload_rows = []
    for policy, wl_data in detailed_results.items():
        for wid, stats in wl_data.items():
            workload_rows.append({
                'Policy': policy,
                'Workload': WORKLOADS[wid]['name'],
                'Workload_ID': wid,
                'Request_Count': stats['count'],
                'Avg_Latency_ms': round(stats['avg_latency'], 1),
                'P95_Latency_ms': round(stats['p95_latency'], 1),
                'SLO_Threshold_ms': stats['slo_threshold'],
                'SLO_Violation_Rate_%': round(stats['slo_violation_pct'], 2),
            })
            
    workload_df = pd.DataFrame(workload_rows)
    workload_df.to_csv(f'{output_dir}/tables/per_workload_results.csv', index=False)
    return results_df, carbon_df, detailed_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim-hours', type=int, default=SIMULATION_HOURS)
    parser.add_argument('--reqs-per-hour', type=int, default=REQUESTS_PER_HOUR)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    
    run_simulation(hours=args.sim_hours, rph=args.reqs_per_hour, seed=args.seed)
