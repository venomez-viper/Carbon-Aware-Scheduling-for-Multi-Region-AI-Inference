# Carbon-Aware Scheduling for Multi-Region AI Inference

## Overview
This project implements a simulation framework for routing AI inference requests across multiple cloud regions. It evaluates the trade-offs between network latency, SLO violations, and carbon intensity by testing various scheduling policies (Latency-First, Carbon-First, and Hybrid). The framework supports multiple AI workloads (BERT-base, BERT-large, ResNet-50), each with specific SLOs and performance profiles.

**Team:** Akash Anipakalu Giridhar · Brandon Youngkrantz · Alexandre Corret · Yogith Ramanan

---

## 📊 Latest Simulation Results

> Last updated: February 24, 2026 · 7-day simulation · 33,600 total requests

### Policy Comparison (Aggregate)

| Policy | Avg Latency (ms) | P95 Latency (ms) | SLO Violation % | Avg Carbon (gCO₂eq/kWh) | Carbon Reduction |
|--------|-----------------|-----------------|-----------------|-------------------------|-----------------|
| Latency-First *(baseline)* | 37.5 | 79.5 | 0.0% | 268.7 | 0% |
| Carbon-First | 132.7 | 225.9 | 71.1% | 24.9 | **90.7%** |
| Hybrid α=0.2 | 102.6 | 201.1 | 51.2% | 35.7 | 86.7% |
| Hybrid α=0.3 | 102.6 | 201.2 | 51.0% | 35.7 | 86.7% |
| Hybrid α=0.5 | 101.7 | 201.1 | 48.4% | 37.1 | 86.2% |
| **Hybrid α=0.7** | **62.8** | **127.5** | **1.3%** | 117.7 | **56.2%** |
| **Constrained Hybrid** ✅ | **65.9** | **137.6** | **1.0%** | 108.6 | **59.6%** |

> ✅ **Recommendation:** The **Constrained Hybrid** policy delivers the best balance — 59.6% carbon reduction with only 1.0% SLO violations. Hybrid α=0.7 is nearly equivalent.

### Per-Workload Breakdown

| Policy | BERT-base SLO Violation | BERT-large SLO Violation | ResNet-50 SLO Violation |
|--------|------------------------|-------------------------|------------------------|
| Latency-First | 0.0% | 0.0% | 0.0% |
| Constrained Hybrid | 0.02% | 1.38% | 5.88% |
| Hybrid α=0.7 | 0.02% | 0.98% | 10.47% |
| Carbon-First | 74.9% | 62.5% | 74.8% |

---

## 📈 Generated Figures

### Figure 1 — Regional Carbon Intensity & Latency
![Figure 1: Regional Carbon and Latency](outputs/graphs/regional_carbon_latency.png)

*EU-North (25 gCO₂eq/kWh) and US-West (79 gCO₂eq/kWh) are the lowest-carbon options but add 90–170 ms latency for US-based users.*

---

### Figure 2 — Carbon–Latency Trade-off Curve (Pareto)
![Figure 2: Pareto Trade-off Curve](outputs/graphs/tradeoff_curve.png)

*Hybrid α=0.7 and Constrained Hybrid sit in the viable zone (SLO violations < 5%) while delivering >56% carbon reduction.*

---

### Figure 3 — Request Routing Distribution by Policy
![Figure 3: Routing Distribution](outputs/graphs/routing_distribution.png)

*Latency-First routes most requests to nearby high-carbon regions. Carbon-First shifts majority to EU-North. Hybrid and Constrained Hybrid achieve a balanced distribution.*

---

### Figure 4 — Per-Workload SLO Violations
![Figure 4: Per-Workload SLO Violations](outputs/graphs/workload_slo_violations.png)

*ResNet-50 (80 ms SLO) is the most sensitive workload. Constrained Hybrid keeps all workloads below 6% violation.*

---

### Figure 5 — Hourly Carbon Intensity Traces (7-Day Simulation)
![Figure 5: Carbon Traces](outputs/graphs/carbon_traces.png)

*EU-North and US-West show consistent low-carbon profiles. Singapore and US-East remain high throughout.*

---

### Prior Work Comparison
![Prior Work Comparison Table](outputs/graphs/prior_work_comparison.png)

---

## ✨ Premium Research Figures

### Carbon Intensity Heatmap (Region × Hour-of-Day)
![Carbon Heatmap](outputs/graphs/premium/carbon_heatmap.png)

*Shows exactly which regions are cleanest at each hour — the core driver behind why carbon-aware routing works.*

---

### Multi-Metric Policy Radar Chart
![Radar Policy Comparison](outputs/graphs/premium/radar_policy_comparison.png)

*Spider chart comparing Latency Score, P95, SLO Compliance, Carbon Reduction, and Carbon Efficiency simultaneously. Constrained Hybrid and Hybrid α=0.7 dominate the viable region.*

---

### Latency CDF by Policy
![Latency CDF](outputs/graphs/premium/latency_cdf.png)

*Standard academic CDF showing latency distributions. Curves to the left = faster. SLO thresholds shown as vertical lines. Latency-First and Constrained Hybrid stay far left of the 95th-percentile line.*

---

### 3-Way Trade-off Bubble Chart
![Bubble Tradeoff](outputs/graphs/premium/bubble_tradeoff.png)

*X=Carbon Reduction, Y=SLO Violations, bubble size=Avg Latency — three dimensions in one figure. Constrained Hybrid sits in the bottom-right sweet spot.*

---

### Editorial Dual Bar: Carbon Savings vs SLO Violations
![Carbon Savings Bar](outputs/graphs/premium/carbon_savings_bar.png)

*Side-by-side horizontal bars make the trade-off immediately readable for any audience.*

---

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone https://github.com/venomez-viper/Carbon-Aware-Scheduling-for-Multi-Region-AI-Inference.git
cd Carbon-Aware-Scheduling-for-Multi-Region-AI-Inference
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the simulation:
```bash
cd src
python simulation.py
```

Optional overrides:
```bash
python simulation.py --sim-hours 336 --reqs-per-hour 5000 --seed 42
```

4. Generate all figures:
```bash
python metrics.py
```

Outputs go to `outputs/graphs/` (figures) and `outputs/tables/` (CSVs).

---

## 🗂️ Repository Layout

```text
Carbon-Aware-Scheduling-for-Multi-Region-AI-Inference/
├── src/
│   ├── config.py          # Multi-workload configuration
│   ├── policies.py        # Scheduling algorithms
│   ├── simulation.py      # Main simulation driver
│   └── metrics.py         # Figure generation (6 plots)
├── outputs/
│   ├── graphs/            # All generated figures (Figs 1–5 + comparison table)
│   └── tables/            # simulation_results.csv, per_workload_results.csv
├── README.md
└── requirements.txt
```

---

## Advanced Configuration

- **Add custom policies:** Write a routing function in `src/policies.py` and register it in `src/simulation.py`.
- **New workloads or regions:** Update `WORKLOADS` or `REGIONS` in `src/config.py` — the simulation adapts automatically.

## Reproducibility

All code, configuration, and output data are in this repository. Running `src/simulation.py` followed by `src/metrics.py` reproduces every figure and table exactly.
