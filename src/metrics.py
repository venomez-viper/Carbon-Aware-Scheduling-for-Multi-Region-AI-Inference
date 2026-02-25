import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CarbonMetrics")

# Apply consistent styling
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def plot_tradeoff_curve(input_csv: Path, output_dir: Path) -> None:
    """
    Plots the trade-off curve between carbon reduction and SLO violations.
    """
    logger.info(f"Generating tradeoff curve from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    plt.figure(figsize=(8, 6))
    
    # Use seaborn scatterplot for better aesthetics
    ax = sns.scatterplot(
        data=df, 
        x='Carbon_Reduction_%', 
        y='SLO_Violation_Rate_%', 
        hue='Policy', 
        s=150, 
        palette='Set1',
        edgecolor='black',
        alpha=0.8
    )
    
    # Annotate points directly on the plot
    for i in range(df.shape[0]):
        plt.text(
            x=df['Carbon_Reduction_%'][i] + 0.5, 
            y=df['SLO_Violation_Rate_%'][i] + 0.5, 
            s=df['Policy'][i], 
            horizontalalignment='left', 
            size='small', 
            color='black', 
            weight='semibold'
        )
                 
    plt.title('Trade-off: Carbon Reduction vs. SLO Violations', fontsize=14, pad=15)
    plt.xlabel('Carbon Reduction vs Baseline (%)', fontsize=12)
    plt.ylabel('SLO Violation Rate (%)', fontsize=12)
    
    out_path = output_dir / 'tradeoff_curve.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    logger.info(f"Saved tradeoff curve to {out_path}")
    
def plot_carbon_traces(input_csv: Path, output_dir: Path) -> None:
    """
    Plots the hourly carbon intensity traces for each cloud region.
    """
    logger.info(f"Generating carbon traces plot from {input_csv}...")
    df = pd.read_csv(input_csv, index_col='hour')
    
    plt.figure(figsize=(10, 5))
    
    # Plot each region
    for region in df.columns:
        plt.plot(df.index, df[region], label=region, linewidth=2, alpha=0.9)
        
    plt.title('Hourly Carbon Intensity by Region', fontsize=14, pad=15)
    plt.xlabel('Hour of Simulation', fontsize=12)
    plt.ylabel('Carbon Intensity (gCO$_2$/kWh)', fontsize=12)
    plt.legend(title="Region", loc='upper left', bbox_to_anchor=(1, 1))
    
    out_path = output_dir / 'carbon_traces.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    logger.info(f"Saved carbon traces to {out_path}")

def plot_workload_slo_violations(input_csv: Path, output_dir: Path) -> None:
    """
    Plots a grouped bar chart of SLO violation rates per workload across different policies.
    This provides deep insight into how specific policies affect different AI models.
    """
    logger.info(f"Generating workload SLO violation chart from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    plt.figure(figsize=(10, 6))
    
    # Create grouped bar chart
    sns.barplot(
        data=df,
        x='Workload',
        y='SLO_Violation_Rate_%',
        hue='Policy',
        palette='Set2',
        edgecolor='black',
        alpha=0.9
    )
    
    plt.title('SLO Violation Rate by AI Inference Workload', fontsize=14, pad=15)
    plt.xlabel('AI Workload Model', fontsize=12)
    plt.ylabel('SLO Violation Rate (%)', fontsize=12)
    plt.legend(title="Routing Policy", loc='upper left', bbox_to_anchor=(1, 1))
    
    out_path = output_dir / 'workload_slo_violations.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    logger.info(f"Saved workload SLO violation chart to {out_path}")

def main():
    base_dir = Path('../')
    data_dir = base_dir / 'data'
    output_dir = base_dir / 'outputs'
    
    trace_path = data_dir / 'carbon' / 'carbon_intensity_traces.csv'
    results_path = output_dir / 'tables' / 'simulation_results.csv'
    workload_path = output_dir / 'tables' / 'workload_results.csv'
    graphs_dir = output_dir / 'graphs'
    
    if results_path.exists():
        plot_tradeoff_curve(results_path, graphs_dir)
    else:
        logger.warning(f"Results file {results_path} not found. Run simulation first.")
        
    if workload_path.exists():
        plot_workload_slo_violations(workload_path, graphs_dir)
    else:
        logger.warning(f"Workload summary file {workload_path} not found. Run simulation first.")
        
    if trace_path.exists():
        plot_carbon_traces(trace_path, graphs_dir)
    else:
        logger.warning(f"Trace file {trace_path} not found. Run simulation first.")

if __name__ == '__main__':
    main()
