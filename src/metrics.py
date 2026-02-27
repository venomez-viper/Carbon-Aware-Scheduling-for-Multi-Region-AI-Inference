import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CarbonMetrics")

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

REGION_COLORS = {
    'US-East':   '#E15759',
    'US-West':   '#59A14F',
    'EU-West':   '#4E79A7',
    'EU-North':  '#76B7B2',
    'Singapore': '#F28E2B',
}

# FIX 1: use \u03b1 (single backslash = actual α character) so keys match CSV
POLICY_COLORS = {
    'Latency-First':        '#E15759',
    'Carbon-First':         '#59A14F',
    'Hybrid (\u03b1=0.2)': '#9467bd',
    'Hybrid (\u03b1=0.3)': '#8c564b',
    'Hybrid (\u03b1=0.5)': '#4E79A7',
    'Hybrid (\u03b1=0.7)': '#F28E2B',
    'Constrained Hybrid':   '#17becf',
}


# ─── Figure 1: Regional Carbon Intensity & Latency ───────────────────────────
def plot_regional_carbon_latency(output_dir: Path) -> None:
    logger.info("Generating Figure 1: Regional carbon & latency comparison...")
    from config import LATENCY_MATRIX, BASE_CARBON_INTENSITY, REGIONS
    regions = REGIONS
    carbon = [BASE_CARBON_INTENSITY[r] for r in regions]
    latency_from_us_east = [int(LATENCY_MATRIX.loc['US-East', r]) for r in regions]

    x = np.arange(len(regions))
    width = 0.55
    fig, ax1 = plt.subplots(figsize=(10, 6))
    bars = ax1.bar(x, carbon, width,
                   color=[REGION_COLORS[r] for r in regions],
                   edgecolor='black', linewidth=0.8, alpha=0.85, zorder=3)
    ax1.set_xlabel('Cloud Region', fontsize=12, labelpad=8)
    # FIX 2: use plain text gCO2eq to avoid subscript font warning on Windows
    ax1.set_ylabel('Avg Carbon Intensity (gCO2eq/kWh)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(regions, rotation=15, ha='right')
    ax1.set_ylim(0, 430)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)

    for bar, val in zip(bars, carbon):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                 f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2 = ax1.twinx()
    ax2.plot(x, latency_from_us_east, color='#2c2c2c', marker='D',
             linewidth=2, markersize=8, zorder=4, label='Latency from US-East')
    ax2.set_ylabel('Network Latency from US-East (ms)', fontsize=12)
    ax2.set_ylim(0, 300)
    for xi, lat in zip(x, latency_from_us_east):
        ax2.annotate(f'{lat} ms', (xi, lat), textcoords='offset points',
                     xytext=(0, 10), ha='center', fontsize=9, color='#2c2c2c')

    plt.title('Figure 1: Carbon Intensity and Network Latency by Cloud Region',
              fontsize=13, pad=14, fontweight='bold')
    bar_patches = [mpatches.Patch(color=REGION_COLORS[r], label=r) for r in regions]
    lat_line = plt.Line2D([0], [0], color='#2c2c2c', marker='D', linewidth=2, label='Latency (ms)')
    ax1.legend(handles=bar_patches + [lat_line], loc='upper left',
               fontsize=9, framealpha=0.85, ncol=2)

    out_path = output_dir / 'regional_carbon_latency.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"Saved Figure 1 to {out_path}")


# ─── Figure 2: Trade-off Pareto Curve ────────────────────────────────────────
def plot_tradeoff_curve(input_csv: Path, output_dir: Path) -> None:
    logger.info(f"Generating Figure 2 (Pareto tradeoff) from {input_csv}...")
    df = pd.read_csv(input_csv, encoding='utf-8')

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axhspan(0, 5, alpha=0.08, color='green', zorder=0)
    ax.text(1, 2.5, '<-- Viable zone (SLO violations < 5%)',
            fontsize=9, color='green', alpha=0.7, style='italic')

    for _, row in df.iterrows():
        policy = row['Policy']
        color = POLICY_COLORS.get(policy, '#888888')
        ax.scatter(row['Carbon Reduction'], row['SLO Violation Rate (%)'],
                   s=160 if 'Constrained' in policy else 120,
                   c=color, edgecolors='black', linewidths=0.8, zorder=5,
                   marker='D' if 'Constrained' in policy else 'o')
        ax.annotate(policy, (row['Carbon Reduction'], row['SLO Violation Rate (%)']),
                    textcoords='offset points', xytext=(6, 4),
                    fontsize=8.5, fontweight='semibold', color='#222222')

    ax.set_xlabel('Carbon Reduction vs. Latency-First Baseline (%)', fontsize=12)
    ax.set_ylabel('SLO Violation Rate (%)', fontsize=12)
    ax.set_title('Figure 2: Carbon Reduction vs. SLO Violations - Policy Trade-off Curve',
                 fontsize=12, fontweight='bold', pad=14)
    legend_handles = [mpatches.Patch(color=POLICY_COLORS.get(p, '#888'), label=p)
                      for p in df['Policy']]
    ax.legend(handles=legend_handles, fontsize=8.5, loc='upper left',
              framealpha=0.85, title='Scheduling Policy')

    out_path = output_dir / 'tradeoff_curve.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"Saved Figure 2 to {out_path}")


# ─── Figure 3: Request Routing Distribution ──────────────────────────────────
def plot_routing_distribution(input_csv: Path, output_dir: Path) -> None:
    logger.info(f"Generating Figure 3: Routing distribution from {input_csv}...")
    df = pd.read_csv(input_csv, encoding='utf-8')
    regions = ['US-East', 'US-West', 'EU-West', 'EU-North', 'Singapore']
    region_cols = [c for c in df.columns if c in regions]

    if not region_cols:
        logger.warning("No region distribution columns found. Skipping routing distribution plot.")
        return

    row_totals = df[region_cols].sum(axis=1)
    df_pct = df[region_cols].div(row_totals, axis=0) * 100

    # FIX 3: use actual α character to match CSV policy names
    target_policies = ['Latency-First', 'Carbon-First', 'Hybrid (\u03b1=0.5)', 'Constrained Hybrid']
    mask = df['Policy'].isin(target_policies)
    df_sel = df_pct[mask].copy()
    labels = df.loc[mask, 'Policy'].values

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    bottoms = np.zeros(len(labels))

    for region in regions:
        if region in df_sel.columns:
            vals = df_sel[region].values
            ax.bar(x, vals, bottom=bottoms, color=REGION_COLORS[region],
                   edgecolor='white', linewidth=0.6, label=region, alpha=0.9)
            for xi, v, b in zip(x, vals, bottoms):
                if v > 8:
                    ax.text(xi, b + v / 2, f'{v:.0f}%',
                            ha='center', va='center', fontsize=8.5,
                            color='white', fontweight='bold')
            bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha='right', fontsize=10)
    ax.set_ylabel('Percentage of Requests Routed (%)', fontsize=12)
    ax.set_ylim(0, 110)
    ax.set_title('Figure 3: Request Routing Distribution Across Regions by Policy',
                 fontsize=12, fontweight='bold', pad=14)
    ax.legend(title='Region', loc='upper right', bbox_to_anchor=(1.18, 1),
              fontsize=9, framealpha=0.85)

    out_path = output_dir / 'routing_distribution.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"Saved Figure 3 to {out_path}")


# ─── Figure 4: Per-Workload SLO Violations ───────────────────────────────────
def plot_workload_slo_violations(input_csv: Path, output_dir: Path) -> None:
    logger.info(f"Generating Figure 4: Workload SLO violation chart from {input_csv}...")
    df = pd.read_csv(input_csv, encoding='utf-8')

    fig, ax = plt.subplots(figsize=(12, 6))
    # FIX 4: pass POLICY_COLORS dict (not a list) — seaborn >= 0.12 requires dict with hue
    sns.barplot(
        data=df,
        x='Workload',
        y='SLO_Violation_Rate_%',
        hue='Policy',
        palette=POLICY_COLORS,
        edgecolor='black',
        linewidth=0.6,
        alpha=0.9,
        ax=ax
    )
    ax.set_title('Figure 4: SLO Violation Rate by AI Inference Workload and Policy',
                 fontsize=12, fontweight='bold', pad=14)
    ax.set_xlabel('AI Workload Model', fontsize=12)
    ax.set_ylabel('SLO Violation Rate (%)', fontsize=12)
    ax.tick_params(axis='x', rotation=10)
    ax.legend(title='Routing Policy', loc='upper left',
              bbox_to_anchor=(1, 1), fontsize=8.5)

    out_path = output_dir / 'workload_slo_violations.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"Saved Figure 4 to {out_path}")


# ─── Figure 5: Carbon Traces ──────────────────────────────────────────────────
def plot_carbon_traces(input_csv: Path, output_dir: Path) -> None:
    logger.info(f"Generating Figure 5: Carbon traces from {input_csv}...")
    df = pd.read_csv(input_csv, index_col='hour', encoding='utf-8')

    fig, ax = plt.subplots(figsize=(12, 5))
    for region in df.columns:
        color = REGION_COLORS.get(region, None)
        ax.plot(df.index, df[region], label=region, linewidth=2, alpha=0.9, color=color)

    ax.set_title('Figure 5: Hourly Carbon Intensity by Cloud Region (7-Day Simulation)',
                 fontsize=12, fontweight='bold', pad=14)
    ax.set_xlabel('Hour of Simulation', fontsize=12)
    # FIX 2: plain text gCO2eq to avoid subscript font warning on Windows
    ax.set_ylabel('Carbon Intensity (gCO2eq/kWh)', fontsize=12)
    ax.legend(title='Region', loc='upper right', fontsize=9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)

    out_path = output_dir / 'carbon_traces.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"Saved Figure 5 to {out_path}")
    
# ─── Prior Work Comparison Table (PNG) ────────────────────────────────────────
def render_prior_work_comparison(results_csv: Path, output_dir: Path) -> None:
    """
    Renders a styled comparison table: This Work vs CASPER, CASA, Microsoft, Google.
    Recommended by out_pdf2.txt §4.6.
    """
    logger.info("Generating prior work comparison table...")

    df_sim = pd.read_csv(results_csv)
    # Best hybrid: α=0.7 has the best SLO + carbon balance
    hybrid_07 = df_sim[df_sim['Policy'] == 'Hybrid (α=0.7)'].iloc[0]
    constrained = df_sim[df_sim['Policy'] == 'Constrained Hybrid'].iloc[0]

    our_finding = (
        f"Hybrid α=0.7: {hybrid_07['Carbon Reduction']:.0f}% carbon "
        f"reduction, {hybrid_07['SLO Violation Rate (%)']:.1f}% SLO violations\n"
        f"Constrained Hybrid: {constrained['Carbon Reduction']:.0f}% reduction, "
        f"{constrained['SLO Violation Rate (%)']:.1f}% SLO violations"
    )

    data = {
        'System': ['CASPER (2023)', 'CASA (2024)', 'Microsoft (2023)',
                'Google CICS (2021)', 'This Work (2026)'],
        'Workload': ['Web services\n(Wikimedia)', 'Serverless\nFaaS',
                    'General\ncompute', 'Datacenter\nworkloads',
                    'AI Inference\n(BERT, ResNet)'],
        'Regions': ['6 AWS\nregions', 'Multi-region\nFaaS', 'Azure\nglobal',
                    'Google\nfleet', '5 regions\n(US/EU/Asia)'],
        'Key Finding': [
            'Up to 70% carbon reduction\n(400–500 ms latency relaxation)',
            '2.6× carbon reduction,\n1.4× SLO improvement',
            'Location-shift: 75% SCI reduction\nvs time-shift: 15%',
            'Carbon-aware temporal shifting\nof deferrable workloads',
            our_finding
        ],
        'Alignment': [
            'Similar trade-off shape;\nour α=0.2 achieves ~87% reduction',
            'Different workload model;\nours targets live inference',
            'Confirms spatial > temporal;\nour spatial-only approach',
            'Our work: non-deferrable\ninference, spatial routing',
            '— (This Work)'
        ],
    }
    table_df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(18, 5))
    ax.axis('off')

    col_widths = [0.12, 0.12, 0.10, 0.36, 0.30]
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc='left',
        loc='center',
        colWidths=col_widths
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 3.2)

    # Style header
    for j in range(len(table_df.columns)):
        cell = table[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')

    # Highlight "This Work" row
    for j in range(len(table_df.columns)):
        cell = table[len(table_df), j]
        cell.set_facecolor('#d4efdf')

    # Alternate row shading
    for i in range(1, len(table_df)):
        shade = '#f8f9fa' if i % 2 == 0 else '#ffffff'
        for j in range(len(table_df.columns)):
            if i < len(table_df):
                table[i, j].set_facecolor(shade)

    ax.set_title('Comparison with Related Carbon-Aware Scheduling Systems',
                fontsize=13, fontweight='bold', pad=16, x=0.5, y=1.02)

    out_path = output_dir / 'prior_work_comparison.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close()
    logger.info(f"Saved prior work comparison table to {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / 'outputs'

    trace_path    = output_dir / 'data'   / 'carbon_intensity_traces.csv'
    results_path  = output_dir / 'tables' / 'simulation_results.csv'
    workload_path = output_dir / 'tables' / 'per_workload_results.csv'
    graphs_dir    = output_dir / 'graphs'

    # Figure 1: Regional comparison (static data from config)
    plot_regional_carbon_latency(graphs_dir)

    if results_path.exists():
        # Figure 2: Enhanced Pareto tradeoff curve
        plot_tradeoff_curve(results_path, graphs_dir)
        # Figure 3: Routing distribution
        plot_routing_distribution(results_path, graphs_dir)
        # Prior Work Comparison Table
        render_prior_work_comparison(results_path, graphs_dir)
    else:
        logger.warning(f"Results file {results_path} not found. Run simulation first.")

    if workload_path.exists():
        # Figure 4: Per-workload SLO violations
        plot_workload_slo_violations(workload_path, graphs_dir)
    else:
        logger.warning(f"Workload file {workload_path} not found. Run simulation first.")

    if trace_path.exists():
        # Figure 5: Carbon intensity traces
        plot_carbon_traces(trace_path, graphs_dir)
    else:
        logger.warning(f"Trace file {trace_path} not found. Run simulation first.")


if __name__ == '__main__':
    main()



