"""
premium_figures.py — High-impact supplementary figures for the carbon-aware
scheduling research paper.

Run from src/:  python premium_figures.py
Outputs saved to ../outputs/graphs/premium/
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path

# ── shared style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "figure.dpi":     150,
    "savefig.dpi":    300,
    "savefig.bbox":   "tight",
    "axes.titlepad":  14,
    "axes.labelpad":  6,
})

POLICY_COLORS = {
    "Latency-First":     "#E63946",
    "Carbon-First":      "#2A9D8F",
    "Hybrid (α=0.2)":   "#8338EC",
    "Hybrid (α=0.3)":   "#3A86FF",
    "Hybrid (α=0.5)":   "#FB8500",
    "Hybrid (α=0.7)":   "#06D6A0",
    "Constrained Hybrid": "#FFB703",
}

REGION_COLORS = {
    "US-East":   "#E63946",
    "US-West":   "#2A9D8F",
    "EU-West":   "#3A86FF",
    "EU-North":  "#06D6A0",
    "Singapore": "#FB8500",
}

OUT = Path("../outputs/graphs/premium")
OUT.mkdir(parents=True, exist_ok=True)

# ── helpers ─────────────────────────────────────────────────────────────────
def load_results():
    return pd.read_csv("../outputs/tables/simulation_results.csv")

def load_traces():
    return pd.read_csv("../outputs/data/carbon_intensity_traces.csv", index_col="hour")

def load_workloads():
    return pd.read_csv("../outputs/tables/per_workload_results.csv")


# ── Figure A: Carbon Intensity Heatmap ──────────────────────────────────────
def fig_carbon_heatmap():
    """
    2-D heatmap: Region × Hour-of-day (24 h average), showing diurnal
    carbon patterns.  Visually striking and explains WHY temporal variation
    exists.
    """
    traces = load_traces()
    regions = list(traces.columns)

    # Build 24-h average matrix
    hours_of_day = np.arange(24)
    matrix = np.zeros((len(regions), 24))
    for h in hours_of_day:
        rows = [i for i in traces.index if i % 24 == h]
        matrix[:, h] = traces.iloc[rows][regions].mean().values

    fig, ax = plt.subplots(figsize=(13, 4.5))
    cmap = plt.cm.RdYlGn_r          # red = high carbon, green = low

    im = ax.imshow(matrix, aspect="auto", cmap=cmap,
                   vmin=0, vmax=matrix.max(),
                   interpolation="bilinear")

    # colour bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.12)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label("gCO₂eq/kWh", fontsize=10)

    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(regions)))
    ax.set_yticklabels(regions, fontsize=10)
    ax.set_xlabel("Hour of Day (local UTC)", fontsize=11)
    ax.set_title("Carbon Intensity by Region & Hour-of-Day\n"
                 "(darker red = higher carbon; green = cleaner grid)",
                 fontsize=12, fontweight="bold")

    # annotate mean inside each cell
    for ri, reg in enumerate(regions):
        for h in range(24):
            val = matrix[ri, h]
            txt_col = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(h, ri, f"{val:.0f}", ha="center", va="center",
                    fontsize=6.5, color=txt_col, fontweight="bold")

    plt.tight_layout()
    path = OUT / "carbon_heatmap.png"
    plt.savefig(path)
    plt.close()
    print(f"[OK] Saved {path}")


# ── Figure B: Radar / Spider Chart ──────────────────────────────────────────
def fig_radar_chart():
    """
    Radar chart comparing 4 key policies on 5 metrics simultaneously.
    The most visually impactful single-figure for a research poster.
    """
    df = load_results()

    # Select representative policies
    policies = ["Latency-First", "Hybrid (α=0.5)", "Hybrid (α=0.7)", "Constrained Hybrid"]
    df = df[df["Policy"].isin(policies)].set_index("Policy")

    # Metrics — each normalised 0→1 where 1 = best
    # Latency score: lower avg latency = better  → invert
    # P95 score: lower = better → invert
    # SLO compliance: lower violation = better → invert
    # Carbon reduction: higher = better
    # Carbon efficiency: lower avg carbon = better → invert
    max_lat   = df["Avg Latency (ms)"].max()
    max_p95   = df["P95 Latency (ms)"].max()
    max_viol  = df["SLO Violation Rate (%)"].max() or 1
    max_carb  = df["Avg Carbon (gCO2eq/kWh)"].max()
    max_red   = df["Carbon Reduction"].max() or 1

    metrics = [
        "Latency Score",
        "P95 Score",
        "SLO Compliance",
        "Carbon Reduction",
        "Carbon Efficiency",
    ]
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # close polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=10.5)

    # ring grid lines
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7.5, color="grey")
    ax.grid(color="grey", linestyle="--", linewidth=0.6, alpha=0.5)

    for policy in policies:
        row = df.loc[policy]
        scores = [
            1 - row["Avg Latency (ms)"] / max_lat,
            1 - row["P95 Latency (ms)"] / max_p95,
            1 - row["SLO Violation Rate (%)"] / max_viol,
            row["Carbon Reduction"] / max_red,
            1 - row["Avg Carbon (gCO2eq/kWh)"] / max_carb,
        ]
        scores += scores[:1]
        color = POLICY_COLORS[policy]
        ax.plot(angles, scores, "o-", linewidth=2.2, color=color,
                markersize=5, label=policy)
        ax.fill(angles, scores, alpha=0.10, color=color)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              fontsize=9.5, framealpha=0.85, title="Policy", title_fontsize=10)
    ax.set_title("Multi-Metric Policy Comparison\n(outer edge = best performance)",
                 fontsize=12, fontweight="bold", pad=26)

    plt.tight_layout()
    path = OUT / "radar_policy_comparison.png"
    plt.savefig(path)
    plt.close()
    print(f"[OK] Saved {path}")


# ── Figure C: Latency CDF per Policy ────────────────────────────────────────
def fig_latency_cdf():
    """
    Empirical CDF of end-to-end latency for each policy — the standard
    academic way to show latency distributions and SLO compliance.
    """
    # Regenerate per-request latencies from simulation constants
    from config import (REGIONS, LATENCY_MATRIX, BASE_CARBON_INTENSITY,
                        SIMULATION_HOURS, REQUESTS_PER_HOUR, RANDOM_SEED,
                        USER_DISTRIBUTION, get_workload_list,
                        get_workload_probabilities, sample_inference_time,
                        get_slo_threshold, HYBRID_ALPHA_VALUES,
                        CARBON_DIURNAL_AMPLITUDE, CARBON_RANDOM_NOISE_RANGE,
                        NETWORK_JITTER_MEAN, NETWORK_JITTER_STD)
    from policies import latency_first, carbon_first, hybrid_policy, constrained_hybrid

    np.random.seed(RANDOM_SEED)
    hours = SIMULATION_HOURS
    rph   = REQUESTS_PER_HOUR

    # Build carbon traces
    carbon = {}
    for region in REGIONS:
        base = BASE_CARBON_INTENSITY[region]
        vals = []
        for h in range(hours):
            hod = h % 24
            d = 1 + CARBON_DIURNAL_AMPLITUDE * np.sin(2*np.pi*(hod-6)/24)
            n = np.random.uniform(1-CARBON_RANDOM_NOISE_RANGE,
                                  1+CARBON_RANDOM_NOISE_RANGE)
            vals.append(max(5, base*d*n))
        carbon[region] = np.array(vals)
    ci_arr = np.column_stack([carbon[r] for r in REGIONS])

    total = rph * hours
    req_hours = np.repeat(np.arange(hours), rph)
    USER_LOCS = list(USER_DISTRIBUTION.keys())
    req_users = np.random.choice(USER_LOCS, size=total,
                                  p=list(USER_DISTRIBUTION.values()))
    req_wls   = np.random.choice(get_workload_list(), size=total,
                                  p=get_workload_probabilities())

    lat_lookup = {ul: np.array([LATENCY_MATRIX.loc[ul, r] for r in REGIONS])
                  for ul in USER_LOCS}

    policy_configs = [
        ("Latency-First",    "latency_first", None),
        ("Carbon-First",     "carbon_first",  None),
    ]
    for a in HYBRID_ALPHA_VALUES:
        policy_configs.append((f"Hybrid (α={a})", "hybrid", a))
    policy_configs.append(("Constrained Hybrid", "constrained", None))

    rng = np.random.default_rng(RANDOM_SEED)
    all_lats = {}

    for label, ptype, alpha in policy_configs:
        lats = np.zeros(total)
        for i in range(total):
            h  = req_hours[i]
            ul = req_users[i]
            wid = req_wls[i]
            net = lat_lookup[ul]
            cis = ci_arr[h]
            inf = sample_inference_time(wid, rng=rng)
            slo = get_slo_threshold(wid)
            if   ptype == "latency_first": idx = latency_first(net, cis)
            elif ptype == "carbon_first":  idx = carbon_first(net, cis)
            elif ptype == "hybrid":        idx = hybrid_policy(net, cis, alpha)
            else:                          idx = constrained_hybrid(net, cis, slo, inf)
            jit = max(0, rng.normal(NETWORK_JITTER_MEAN, NETWORK_JITTER_STD))
            lats[i] = max(1.0, net[idx] + inf + jit)
        all_lats[label] = np.sort(lats)

    fig, ax = plt.subplots(figsize=(11, 6))
    cdf_y = np.linspace(0, 1, total)

    for label, lats in all_lats.items():
        color = POLICY_COLORS[label]
        ls = "--" if "Carbon-First" in label else \
             ":"  if "α=0.2" in label or "α=0.3" in label else "-"
        ax.plot(lats, cdf_y, linewidth=2, color=color, linestyle=ls, label=label)

    # SLO reference lines
    for slo, wl in [(100, "BERT-base SLO"), (150, "BERT-large SLO"), (80, "ResNet-50 SLO")]:
        ax.axvline(slo, color="black", linestyle=":", linewidth=1.0, alpha=0.5)
        ax.text(slo+2, 0.05, wl, fontsize=8, rotation=90, color="black", alpha=0.6)

    ax.axhline(0.95, color="black", linestyle="--", linewidth=0.9, alpha=0.4)
    ax.text(2, 0.96, "95th percentile (SLO target)", fontsize=8.5, color="grey")

    ax.set_xlabel("End-to-End Request Latency (ms)", fontsize=12)
    ax.set_ylabel("Empirical CDF (fraction of requests)", fontsize=12)
    ax.set_title("Latency CDF by Scheduling Policy\n"
                 "Curves closer to the left = faster; crossing the 95% line = SLO met",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, 350)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.88)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = OUT / "latency_cdf.png"
    plt.savefig(path)
    plt.close()
    print(f"[OK] Saved {path}")


# ── Figure D: 3-Way Bubble Chart ────────────────────────────────────────────
def fig_bubble_tradeoff():
    """
    Bubble chart: X=Carbon Reduction, Y=SLO Violation Rate,
    bubble size=Avg Latency.  Shows 3 dimensions in one publication figure.
    """
    df = load_results()

    fig, ax = plt.subplots(figsize=(10, 7))

    # shaded viable region
    ax.axhspan(0, 5, alpha=0.07, color="#06D6A0", zorder=0)
    ax.text(2, 1.2, "✅  Viable zone: SLO violations < 5%",
            fontsize=9.5, color="#06D6A0", fontstyle="italic", fontweight="bold")

    size_scale = 60   # bubble scale factor
    for _, row in df.iterrows():
        policy = row["Policy"]
        cr = row["Carbon Reduction"]
        sv = row["SLO Violation Rate (%)"]
        al = row["Avg Latency (ms)"]
        color = POLICY_COLORS.get(policy, "#888888")
        marker = "D" if "Constrained" in policy else "o"

        ax.scatter(cr, sv, s=al * size_scale, c=color,
                   alpha=0.78, edgecolors="white", linewidths=1.5,
                   marker=marker, zorder=5)
        ax.annotate(policy, (cr, sv),
                    textcoords="offset points",
                    xytext=(9, 5), fontsize=9,
                    fontweight="bold", color=color,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.8))

    # Bubble size legend
    for lat_val in [50, 100, 150]:
        ax.scatter([], [], s=lat_val * size_scale, c="grey", alpha=0.5,
                   label=f"Avg Latency = {lat_val} ms")

    ax.set_xlabel("Carbon Reduction vs. Latency-First Baseline (%)", fontsize=12)
    ax.set_ylabel("SLO Violation Rate (%)", fontsize=12)
    ax.set_title("3-Way Trade-off: Carbon Reduction × SLO Violations × Latency\n"
                 "(bubble size = average end-to-end latency)",
                 fontsize=12, fontweight="bold")
    ax.legend(title="Bubble Size Key", fontsize=9, loc="upper left",
              framealpha=0.85)
    ax.set_xlim(-5, 100)
    ax.set_ylim(-2, 80)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    path = OUT / "bubble_tradeoff.png"
    plt.savefig(path)
    plt.close()
    print(f"[OK] Saved {path}")


# ── Figure E: Carbon Savings Bar Chart (clean summary) ───────────────────────
def fig_carbon_savings_bar():
    """
    Horizontal diverging bar: carbon saved (positive) vs SLO violation cost
    (negative axis), for a clear editorial-style figure.
    """
    df = load_results()
    policies = df["Policy"].tolist()
    carbon_red = df["Carbon Reduction"].tolist()
    slo_viol   = df["SLO Violation Rate (%)"].tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6),
                                    gridspec_kw={"wspace": 0.05})

    colors = [POLICY_COLORS.get(p, "#888") for p in policies]
    y = np.arange(len(policies))

    # Left: Carbon reduction
    bars1 = ax1.barh(y, carbon_red, color=colors, edgecolor="white",
                     linewidth=0.8, height=0.65, alpha=0.9)
    ax1.set_yticks(y)
    ax1.set_yticklabels(policies, fontsize=10)
    ax1.set_xlabel("Carbon Reduction vs. Baseline (%)", fontsize=11)
    ax1.set_xlim(0, 105)
    ax1.set_title("Carbon Reduction", fontsize=12, fontweight="bold")
    ax1.xaxis.grid(True, linestyle="--", alpha=0.4)
    for bar, val in zip(bars1, carbon_red):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}%", va="center", fontsize=9.5, fontweight="bold",
                 color=POLICY_COLORS.get(policies[list(carbon_red).index(val)], "#333"))

    # Right: SLO violations
    bars2 = ax2.barh(y, slo_viol, color=colors, edgecolor="white",
                     linewidth=0.8, height=0.65, alpha=0.9)
    ax2.set_yticks(y)
    ax2.set_yticklabels([], fontsize=10)
    ax2.set_xlabel("SLO Violation Rate (%)", fontsize=11)
    ax2.set_xlim(0, 85)
    ax2.set_title("SLO Violations", fontsize=12, fontweight="bold")
    ax2.axvline(5, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
    ax2.text(5.5, len(policies)-0.6, "← 5% threshold", fontsize=8.5, color="red")
    ax2.xaxis.grid(True, linestyle="--", alpha=0.4)
    for bar, val in zip(bars2, slo_viol):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}%", va="center", fontsize=9.5, fontweight="bold",
                 color="#444")

    fig.suptitle("Carbon Reduction vs. SLO Violations Across Scheduling Policies",
                 fontsize=13, fontweight="bold", y=1.01)

    plt.tight_layout()
    path = OUT / "carbon_savings_bar.png"
    plt.savefig(path)
    plt.close()
    print(f"[OK] Saved {path}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n[*] Generating premium research figures...\n")
    fig_carbon_heatmap()
    fig_radar_chart()
    fig_latency_cdf()
    fig_bubble_tradeoff()
    fig_carbon_savings_bar()
    print(f"\n[OK] All premium figures saved to {OUT.resolve()}\n")
