# Carbon-Aware Scheduling for Multi-Region AI Inference

## Overview
This project implements a simulation framework for routing AI inference requests across multiple cloud regions. It evaluates the trade-offs between network latency, SLO violations, and carbon intensity by testing various scheduling policies (Latency-First, Carbon-First, and Hybrid). The framework also supports multiple AI workloads (e.g., BERT-base, BERT-large, ResNet-50), each with specific SLOs and performance profiles.

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/venomez-viper/Carbon-Aware-Scheduling-for-Multi-Region-AI-Inference.git
cd Carbon-Aware-Scheduling-for-Multi-Region-AI-Inference
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the baseline simulation:
```bash
cd src
python simulation.py
```

*Note: Researchers can override core constants cleanly via the command line interface without altering source files:*
```bash
python simulation.py --sim-hours 336 --reqs-per-hour 5000 --seed 42
```

This generates summary tables and CSV files inside `outputs/tables/` and `data/` metrics.

4. Generate figures:
```bash
python metrics.py
```
This produces graphs representing carbon traces and trade-off curves in `outputs/graphs/` natively ready for academic publishing.

## Advanced Configuration for Researchers

### Adding Custom Routing Policies
Researchers can evaluate their own heuristic or machine-learning-based schedulers by writing a new routing function and registering it in the `POLICIES` dictionary found at the bottom of `src/policies.py`. 

### Defining New AI Workloads or Cloud Regions
The framework is fully modularized. If your study requires different cloud region models, varying latency topologies, or new generative AI payloads (e.g., Llama-3, GPT-4), you can define these natively within `src/config.py`. Update the `WORKLOADS` or `REGIONS` definitions, and the simulator matrices will dynamically adjust.

## Repository Layout

```text
Carbon-Aware-Scheduling-for-Multi-Region-AI-Inference/
├── src/
│   ├── config.py          # Configuration for simulation
│   ├── policies.py        # Scheduling algorithms
│   ├── simulation.py      # Main simulation driver
│   └── metrics.py         # Visualization scripts
├── data/
│   ├── carbon/            # Carbon intensity traces (Generated)
│   └── latency/           # User-to-region latency matrix (Generated)
├── outputs/
│   ├── tables/            # Metrics summary tables (Generated)
│   └── graphs/            # Trade-off curves and plots (Generated)
├── README.md
└── requirements.txt
```

## Reproducibility Statement

This repository contains all code and configuration needed to reproduce the results in our paper on carbon-aware scheduling for multi-region AI inference. Simulation parameters, region definitions, and workloads are directly specified in `src/config.py`. Running `src/simulation.py` followed by `src/metrics.py` will regenerate the exact evaluation tables and plot figures mirroring the manuscript.
