import numpy as np
import pandas as pd
import os
import argparse
from pathlib import Path
from config import *
from policies import latency_first, carbon_first, hybrid_policy, constrained_hybrid, adaptive_hybrid


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


def run_simulation(output_dir=None, hours=SIMULATION_HOURS, rph=REQUESTS_PER_HOUR, seed=RANDOM_SEED):
    if output_dir is None:
        output_dir = str(Path(__file__).parent.parent / 'outputs')
    os.makedirs(f'{output_dir}/tables', exist_ok=True)
    os.makedirs(f'{output_dir}/data', exist_ok=True)

    carbon_df = generate_carbon_traces(hours, seed=seed)
    ci_arr = carbon_df.values

    req_hours, req_users, req_workloads = generate_requests(hours, rph, seed=seed)
    total_requests = len(req_hours)

    lat_lookup = {}
    for ul in USER_LOCATIONS:
        lat_lookup[ul] = np.array([LATENCY_MATRIX.loc[ul, r] for r in REGIONS])

    # --- Policy configuration ---
    policy_configs = [
        ('Latency-First', 'latency_first', None),
        ('Carbon-First',  'carbon_first',  None),
    ]
    for alpha in HYBRID_ALPHA_VALUES:
        policy_configs.append((f'Hybrid (\u03b1={alpha})', 'hybrid', alpha))
    policy_configs.append(('Constrained Hybrid', 'constrained', None))
    policy_configs.append(('Adaptive Hybrid',    'adaptive',    None))

    results          = {}
    detailed_results = {}

    # --- Adaptive Hybrid controller constants ---
    # Sliding window size per workload for P95 estimation
    WINDOW_SIZE     = 200
    # How much alpha shifts per controller update
    ALPHA_STEP      = 0.02
    # EMA smoothing factor toward neutral (0.5) in comfortable zone
    EMA_FACTOR      = 0.05
    # Alpha bounds — never go fully extreme in either direction
    ALPHA_MIN       = 0.10
    ALPHA_MAX       = 0.90
    # Headroom thresholds that trigger alpha adjustment
    # headroom = (SLO - P95_observed) / SLO
    HEADROOM_RELAX  = 0.30   # > 30% headroom → shift carbon-aware (lower alpha)
    HEADROOM_TIGHT  = 0.10   # < 10% headroom → shift latency-safe (raise alpha)
    # Minimum observations before controller activates
    MIN_OBS         = 50

    for label, ptype, alpha in policy_configs:
        # Reset RNG per policy — identical inference/jitter samples for fair comparison
        rng = np.random.default_rng(seed)

        latencies         = np.zeros(total_requests)
        carbons_out       = np.zeros(total_requests)
        inference_times   = np.zeros(total_requests)
        region_selections = np.zeros(total_requests, dtype=int)
        workload_stats    = {
            wid: {'count': 0, 'latencies': [], 'slo_violations': 0}
            for wid in get_workload_list()
        }
        total_slo_violations = 0
        region_counts = np.zeros(len(REGIONS), dtype=int)

        # Per-workload adaptive state — reset fresh for each policy run
        adaptive_alpha  = {wid: 0.5 for wid in get_workload_list()}
        latency_windows = {wid: [] for wid in get_workload_list()}

        for i in range(total_requests):
            h   = req_hours[i]
            ul  = req_users[i]
            wid = req_workloads[i]
            lats = lat_lookup[ul]
            cis  = ci_arr[h]

            inference_ms  = sample_inference_time(wid, rng=rng)
            inference_times[i] = inference_ms
            slo_threshold = get_slo_threshold(wid)

            # --- Route request ---
            if ptype == 'latency_first':
                idx = latency_first(lats, cis)
            elif ptype == 'carbon_first':
                idx = carbon_first(lats, cis)
            elif ptype == 'hybrid':
                idx = hybrid_policy(lats, cis, alpha)
            elif ptype == 'constrained':
                idx = constrained_hybrid(lats, cis, slo_threshold, inference_ms)
            elif ptype == 'adaptive':
                # Use the current dynamically adjusted alpha for this workload
                idx = adaptive_hybrid(lats, cis, adaptive_alpha[wid])
            else:
                raise ValueError(f"Unknown policy type: {ptype}")

            region_selections[i] = idx
            net_lat   = lats[idx]
            jitter    = max(0, rng.normal(NETWORK_JITTER_MEAN, NETWORK_JITTER_STD))
            total_lat = max(1.0, net_lat + inference_ms + jitter)

            latencies[i]   = total_lat
            carbons_out[i] = cis[idx]
            region_counts[idx] += 1

            if total_lat > slo_threshold:
                total_slo_violations += 1
                workload_stats[wid]['slo_violations'] += 1
            workload_stats[wid]['count'] += 1
            workload_stats[wid]['latencies'].append(total_lat)

            # --- Adaptive controller update ---
            # Only runs when policy is adaptive, but the structure sits here
            # so it updates alpha AFTER the routing decision (causal)
            if ptype == 'adaptive':
                window = latency_windows[wid]
                window.append(total_lat)
                if len(window) > WINDOW_SIZE:
                    window.pop(0)
                if len(window) >= MIN_OBS:
                    p95_obs  = np.percentile(window, 95)
                    headroom = (slo_threshold - p95_obs) / slo_threshold
                    a = adaptive_alpha[wid]
                    if headroom > HEADROOM_RELAX:
                        # Lots of SLO budget — route more carbon-aware
                        a = max(ALPHA_MIN, a - ALPHA_STEP)
                    elif headroom < HEADROOM_TIGHT:
                        # SLO is tight — protect latency
                        a = min(ALPHA_MAX, a + ALPHA_STEP)
                    else:
                        # Comfortable zone — soft EMA pull toward neutral 0.5
                        a = (1 - EMA_FACTOR) * a + EMA_FACTOR * 0.5
                    adaptive_alpha[wid] = a

        # --- Aggregate results ---
        results[label] = {
            'avg_latency':       round(np.mean(latencies), 1),
            'p95_latency':       round(np.percentile(latencies, 95), 1),
            'slo_violation_pct': round(100 * total_slo_violations / total_requests, 2),
            'avg_carbon':        round(np.mean(carbons_out), 1),
            'avg_inference_time': round(np.mean(inference_times), 1),
            'region_dist':       {REGIONS[j]: int(region_counts[j]) for j in range(len(REGIONS))},
        }

        # Save final converged alpha values for Adaptive Hybrid (useful for paper)
        if ptype == 'adaptive':
            results[label]['final_alpha'] = {
                wid: round(adaptive_alpha[wid], 3) for wid in get_workload_list()
            }

        detailed_results[label] = {}
        for wid, stats in workload_stats.items():
            if stats['count'] > 0:
                detailed_results[label][wid] = {
                    'count':             stats['count'],
                    'avg_latency':       np.mean(stats['latencies']),
                    'p95_latency':       np.percentile(stats['latencies'], 95),
                    'slo_violation_pct': 100 * stats['slo_violations'] / stats['count'],
                    'slo_threshold':     get_slo_threshold(wid),
                }

    # --- Carbon reduction vs Latency-First baseline ---
    baseline_carbon = results['Latency-First']['avg_carbon']
    for label, res in results.items():
        if label == 'Latency-First':
            res['carbon_reduction'] = 0.0
        else:
            res['carbon_reduction'] = round(100 * (1 - res['avg_carbon'] / baseline_carbon), 1)

    # --- Print final adaptive alpha convergence values ---
    if 'Adaptive Hybrid' in results and 'final_alpha' in results['Adaptive Hybrid']:
        print("\n[Adaptive Hybrid] Converged alpha values per workload:")
        for wid, val in results['Adaptive Hybrid']['final_alpha'].items():
            print(f"  {wid:12s} → α = {val}")
    print()

    # --- Build and save results CSV ---
    rows = []
    for label, res in results.items():
        row = {
            'Policy':                  label,
            'Avg Latency (ms)':        res['avg_latency'],
            'P95 Latency (ms)':        res['p95_latency'],
            'SLO Violation Rate (%)':  res['slo_violation_pct'],
            'Avg Carbon (gCO2eq/kWh)': res['avg_carbon'],
            'Carbon Reduction':        res['carbon_reduction'],
        }
        for region, count in res['region_dist'].items():
            row[region] = count
        rows.append(row)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(
        f'{output_dir}/tables/simulation_results.csv',
        index=False, encoding='utf-8'
    )
    carbon_df.to_csv(f'{output_dir}/data/carbon_intensity_traces.csv', index_label='hour')
    LATENCY_MATRIX.to_csv(f'{output_dir}/data/latency_matrix.csv')

    workload_rows = []
    for policy, wl_data in detailed_results.items():
        for wid, stats in wl_data.items():
            workload_rows.append({
                'Policy':               policy,
                'Workload':             WORKLOADS[wid]['name'],
                'Workload_ID':          wid,
                'Request_Count':        stats['count'],
                'Avg_Latency_ms':       round(stats['avg_latency'], 1),
                'P95_Latency_ms':       round(stats['p95_latency'], 1),
                'SLO_Threshold_ms':     stats['slo_threshold'],
                'SLO_Violation_Rate_%': round(stats['slo_violation_pct'], 2),
            })
    workload_df = pd.DataFrame(workload_rows)
    workload_df.to_csv(
        f'{output_dir}/tables/per_workload_results.csv',
        index=False, encoding='utf-8'
    )

    return results_df, carbon_df, detailed_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim-hours',      type=int, default=SIMULATION_HOURS)
    parser.add_argument('--reqs-per-hour',  type=int, default=REQUESTS_PER_HOUR)
    parser.add_argument('--seed',           type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    run_simulation(hours=args.sim_hours, rph=args.reqs_per_hour, seed=args.seed)
