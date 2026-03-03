# Carbon-Aware Scheduling for Multi-Region AI Inference

## Overview

This project implements a discrete-event simulation framework for evaluating carbon-aware request routing strategies across five cloud regions. It quantifies the trade-offs between network latency, SLO compliance, and grid carbon intensity under six scheduling policies — Latency-First, Carbon-First, Hybrid (α-sweep), Constrained Hybrid, and **Adaptive Hybrid** — across three representative AI inference workloads (BERT-base, BERT-large, ResNet-50), each with distinct SLO thresholds and inference time profiles.

**Team:** Akash Anipakalu Giridhar · Yogith Ramanan · Brandon Youngkrantz · Alexandre Corret 

---

## 📊 Latest Simulation Results

> Last updated: March 2, 2026 · 7-day simulation · 33,600 total requests · Seed 42

### Policy Comparison (Aggregate)

| Policy | Avg Latency (ms) | P95 Latency (ms) | SLO Violation % | Avg Carbon (gCO₂eq/kWh) | Carbon Reduction |
|--------|:---------------:|:---------------:|:---------------:|:------------------------:|:----------------:|
| Latency-First *(baseline)* | 37.5 | 79.5 | 0.00% | 268.7 | 0% |
| Carbon-First | 132.7 | 225.3 | 71.31% | 24.9 | **90.7%** |
| Hybrid α=0.2 | 102.9 | 200.8 | 51.45% | 35.6 | 86.8% |
| Hybrid α=0.3 | 102.6 | 200.6 | 51.25% | 35.7 | 86.7% |
| Hybrid α=0.5 | 82.1 | 193.2 | 20.31% | 70.5 | 73.8% |
| **Hybrid α=0.7** | **45.9** | **92.6** | **0.32%** | 182.5 | **32.1%** |
| **Constrained Hybrid** ✅ | **62.3** | **129.3** | **0.00%** | 121.5 | **54.8%** |
| **Adaptive Hybrid** 🔁 | **50.6** | **122.9** | **4.53%** | 189.0 | **29.7%** |

> ✅ **Recommendation:** The **Constrained Hybrid** policy delivers the best production-viable outcome — **54.8% carbon reduction with 0.0% SLO violations**. It achieves this by filtering candidate regions to those satisfying the per-workload SLO budget (RTT + inference + 9 ms jitter buffer ≤ SLO threshold) before selecting the lowest-carbon option. Hybrid α=0.7 is a strong alternative, offering a slightly lower average latency (45.9 ms) at a marginal 0.32% SLO violation rate.
>
> 🔁 **Adaptive Hybrid** uses a closed-loop controller that dynamically adjusts α per workload based on observed P95 latency headroom against SLO thresholds, converging to a workload-specific balance without manual tuning.

### Per-Workload SLO Breakdown

| Policy | BERT-base (SLO: 100 ms) | BERT-large (SLO: 150 ms) | ResNet-50 (SLO: 80 ms) |
|--------|:-----------------------:|:------------------------:|:----------------------:|
| Latency-First | 0.00% | 0.00% | 0.00% |
| **Constrained Hybrid** | **0.00%** | **0.00%** | **0.00%** |
| Hybrid α=0.7 | 0.00% | 0.24% | 2.50% |
| Hybrid α=0.5 | 20.61% | 17.20% | 28.68% |
| Adaptive Hybrid | — | — | — |
| Carbon-First | 74.87% | 62.48% | 74.76% |

> ResNet-50 (80 ms SLO) is the most latency-sensitive workload. Hybrid α=0.7 generates 2.5% violations on ResNet-50 due to occasional cross-region routing, while Constrained Hybrid's SLO-gating eliminates violations entirely across all three workloads.

---

## 📈 Generated Figures

### Figure 1 — Regional Carbon Intensity & Latency
![Figure 1: Regional Carbon and Latency](outputs/graphs/regional_carbon_latency.png)
*EU-North (25 gCO₂eq/kWh) and US-West (79 gCO₂eq/kWh) are the lowest-carbon regions, but impose 90–155 ms additional network latency for US-East users — the core spatial tension this work addresses.*

---

### Figure 2 — Carbon–Latency Trade-off Curve (Pareto)
![Figure 2: Pareto Trade-off Curve](outputs/graphs/tradeoff_curve.png)
*Only Constrained Hybrid and Hybrid α=0.7 occupy the viable zone (SLO violations < 5%) while delivering meaningful carbon savings. Carbon-First and low-α Hybrid variants achieve higher carbon reduction but at operationally unacceptable SLO violation rates.*

---

### Figure 3 — Request Routing Distribution by Policy
![Figure 3: Routing Distribution](outputs/graphs/routing_distribution.png)
*Latency-First concentrates traffic in nearby high-carbon regions (US-East, US-West). Carbon-First routes 100% of requests to EU-North. Constrained Hybrid achieves a geographically balanced distribution, routing primarily to US-West (53%) and EU-North (27%) — leveraging their lower carbon profiles where latency budgets permit.*

---

### Figure 4 — Per-Workload SLO Violation Rates
![Figure 4: Per-Workload SLO Violations](outputs/graphs/workload_slo_violations.png)
*Grouped bar chart showing SLO violation rates per workload across all policies. Constrained Hybrid achieves 0.0% violations across all three workload types. ResNet-50's tight 80 ms SLO is the binding constraint that differentiates viable policies from impractical ones.*

---

### Figure 5 — Hourly Carbon Intensity Traces (7-Day Simulation)
![Figure 5: Carbon Traces](outputs/graphs/carbon_traces.png)
*EU-North and US-West maintain consistently low carbon profiles throughout the simulation window. Singapore and US-East remain persistently high. The 20% diurnal amplitude is visible across all regions, confirming the modeled temporal variation.*

---

### Prior Work Comparison
![Prior Work Comparison Table](outputs/graphs/prior_work_comparison.png)
*Positioning of this work relative to CASPER (2023), CASA (2024), Microsoft Carbon-Aware Computing (2023), and Google CICS (2021). This work uniquely targets live AI inference with per-workload SLO enforcement and an adaptive closed-loop controller across a 5-region spatial routing setup.*

---

## ✨ Premium Research Figures

### Carbon Intensity Heatmap (Region × Hour-of-Day)
![Carbon Heatmap](outputs/graphs/premium/carbon_heatmap.png)
*2D heatmap of average carbon intensity by region and hour of day (24-hour aggregated over 7 days). Confirms that EU-North and US-West are the cleanest options throughout the entire day. Singapore and US-East show no low-carbon window, making temporal shifting ineffective for those regions.*

---

### Multi-Metric Policy Radar Chart
![Radar Policy Comparison](outputs/graphs/premium/radar_policy_comparison.png)
*Spider chart comparing four representative policies — Latency-First, Hybrid α=0.7, Constrained Hybrid, and Adaptive Hybrid — across five normalized performance dimensions: Latency Score, P95 Score, SLO Compliance, Carbon Reduction, and Carbon Efficiency. Constrained Hybrid dominates across all five axes simultaneously.*

---

### Latency CDF by Policy
![Latency CDF](outputs/graphs/premium/latency_cdf.png)
*Empirical CDF of end-to-end request latency per policy. Vertical lines mark the per-workload SLO thresholds (80 ms ResNet-50, 100 ms BERT-base, 150 ms BERT-large). Latency-First and Constrained Hybrid cross the 95th-percentile threshold well within all SLO bounds. Adaptive Hybrid tracks close to Constrained Hybrid due to its dynamic α convergence.*

---

### 3-Way Trade-off Bubble Chart
![Bubble Tradeoff](outputs/graphs/premium/bubble_tradeoff.png)
*Three dimensions visualized simultaneously: X = Carbon Reduction, Y = SLO Violation Rate, bubble size = Average Latency. Constrained Hybrid occupies the bottom-right sweet spot — high carbon reduction, zero SLO violations, moderate latency.*

---

### Editorial Dual Bar: Carbon Savings vs SLO Violations
![Carbon Savings Bar](outputs/graphs/premium/carbon_savings_bar.png)
*Side-by-side horizontal bar chart for immediate readability. Left panel shows carbon reduction %, right panel shows SLO violation rate % with a 5% viability threshold marked. Designed for presentation and executive audiences.*

---

## 🚀 Quick Start

### 1. Clone the repository:
```bash
git clone https://github.com/venomez-viper/Carbon-Aware-Scheduling-for-Multi-Region-AI-Inference.git
cd Carbon-Aware-Scheduling-for-Multi-Region-AI-Inference
```
### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the simulation:
```bash
cd src
python simulation.py
```

### Optional parameter overrides:
```bash
python simulation.py --sim-hours 336 --reqs-per-hour 500 --seed 42
```

### 4. Generate all standard figures:
```bash
python metrics.py
```

### 5. Generate premium research figures:
```bash
python premium_figures.py
```
---
## All outputs are written to:

- outputs/tables/ — simulation_results.csv, per_workload_results.csv
- outputs/data/ — carbon_intensity_traces.csv, latency_matrix.csv
- outputs/graphs/ — Figures 1–5 + prior work comparison table (PNG)
- outputs/graphs/premium/ — 5 additional research-grade figures

---

🗂️ Repository Layout:
```bash
Carbon-Aware-Scheduling-for-Multi-Region-AI-Inference/
├── src/
│   ├── config.py            # Region definitions, workload profiles, simulation parameters
│   ├── policies.py          # Scheduling policy implementations (4 policies)
│   ├── simulation.py        # Main simulation driver — generates traces, routes requests, exports CSVs
│   ├── metrics.py           # Standard figure generation (Figures 1–5 + prior work table)
│   └── premium_figures.py   # Premium research figures (heatmap, radar, CDF, bubble, dual-bar)
├── outputs/
│   ├── graphs/              # All generated figures
│   │   └── premium/         # Premium research figures
│   └── tables/              # simulation_results.csv, per_workload_results.csv
├── README.md
└── requirements.txt
```
---

## ⚙️ Scheduling Policies

| Policy | Logic | Best For |
|--------|-------|----------|
| Latency-First | Routes every request to the minimum-RTT region | Baseline; latency-critical workloads |
| Carbon-First | Routes every request to the minimum-carbon region | Deferrable or batch workloads |
| Hybrid (α) | Weighted score: α·norm_latency + (1−α)·norm_carbon | Tunable trade-off; α=0.7 recommended |
| Constrained Hybrid | SLO-filter first, then pick lowest carbon among eligible regions | Production inference; hard SLO guarantees |
| Adaptive Hybrid | Closed-loop controller adjusts α per workload based on observed P95 vs SLO headroom | Self-tuning; no manual α selection needed |

Global min-max normalization ensures α is a stable, consistent weight across all requests regardless of instantaneous carbon or latency values.

---

## 🎛️ Adaptive Hybrid — Controller Details
The Adaptive Hybrid controller maintains a separate α per workload and updates it after every request using a sliding window P95 estimate:

- Headroom > 30% → shift carbon-aware (lower α) — lots of SLO budget available
- Headroom < 10% → shift latency-safe (raise α) — SLO is tight
- Comfortable zone → soft EMA pull toward neutral α=0.5
Controller constants: window=200 requests, α step=0.02, α bounds=[0.10, 0.90], EMA factor=0.05, minimum observations=50.

---

## 🔧 Advanced Configuration

- Add custom policies: Implement a routing function in `src/policies.py` and register it in the `policy_configs` list in `src/simulation.py`.
- Add new workloads or regions: Update `WORKLOADS` or `REGIONS` in `src/config.py` — the simulation adapts automatically.
- Adjust SLO thresholds: Modify `slo_threshold_ms` per workload in `config.py` to model stricter or more relaxed SLO regimes.
- Extend the α sweep: Add values to `HYBRID_ALPHA_VALUES` in `config.py` for a finer-grained trade-off curve.
- Tune the Adaptive controller: Modify`WINDOW_SIZE`, `ALPHA_STEP`, `HEADROOM_RELAX`, and `HEADROOM_TIGHT` constants in `simulation.py`.
  
---

## 🔁 Reproducibility

All code, configuration, random seeds, and output data are committed to this repository. Running `src/simulation.py` followed by `src/metrics.py` and `src/premium_figures.py` with the default seed (42) reproduces every figure and table exactly as reported.

```bash
cd src
python simulation.py   # regenerates outputs/tables/*.csv
python metrics.py      # regenerates outputs/graphs/*.png
python premium_figures.py   # regenerates outputs/graphs/premium/*.png
```
***
