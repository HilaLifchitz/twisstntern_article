import os
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import contextlib
import io
from matplotlib.colors import SymLogNorm, Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Import your analysis functions
from dataCompare.metrics import compute_all_metrics
from dataCompare.residual_analysis import perform_enhanced_grid_analysis
from twisstntern.utils import return_triangle_coord

# --- Suppress all output during metric and triangle calculations ---
class suppress_output:
    def __enter__(self):
        self._stdout = contextlib.redirect_stdout(io.StringIO())
        self._stderr = contextlib.redirect_stderr(io.StringIO())
        self._stdout.__enter__()
        self._stderr.__enter__()
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stdout.__exit__(exc_type, exc_val, exc_tb)
        self._stderr.__exit__(exc_type, exc_val, exc_tb)


def extract_x_value(filename, pattern):
    # pattern: 'ne' or 'm0.'
    if pattern == 'ne':
        match = re.search(r'ne(\d+\.\d+)', filename)
        if match:
            return float(match.group(1))
    elif pattern == 'm0.':
        match = re.search(r'_m0\.(\d+)', filename)
        if match:
            return float('0.' + match.group(1))
    return None


def main():
    parser = argparse.ArgumentParser(description="Compare TRUTH file to pattern-matched CSVs and plot metrics.")
    parser.add_argument('subdir', type=str, help='Subdirectory containing CSV files')
    parser.add_argument('--pattern', type=str, required=True, help="Pattern for files (e.g., 'ne' or 'm0.')")
    parser.add_argument('--truth', type=str, required=True, help='Filename of the TRUTH CSV')
    parser.add_argument('--colormap', type=str, default='magma', help='Colormap for L2 ternary plots')
    parser.add_argument('--alpha', type=float, default=0.1, help='Grid granularity (default: 0.1)')
    parser.add_argument('--output', type=str, default='compare_results', help='Output directory for plots')
    args = parser.parse_args()

    sns.set_context('notebook')
    sns.set_style('whitegrid')

    subdir = args.subdir
    pattern = args.pattern
    truth_file = args.truth
    cmap_name = args.colormap
    alpha = args.alpha
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # --- Load TRUTH data ---
    truth_path = os.path.join(subdir, truth_file)
    with suppress_output():
        from dataCompare.metrics import load_data
        data1 = load_data(truth_path)

    # --- Find all pattern-matched files (excluding TRUTH) ---
    files = [f for f in os.listdir(subdir) if f.endswith('.csv') and f != truth_file and pattern in f]
    files.sort()

    # --- Initialize results storage ---
    results = []
    ternary_l2_arrays = []
    ternary_results_dfs = []
    x_values = []

    for f in files:
        x_val = extract_x_value(f, pattern)
        if x_val is None:
            continue
        file_path = os.path.join(subdir, f)
        with suppress_output():
            data2 = load_data(file_path)
            metrics = compute_all_metrics(data1, data2, alpha=alpha)
            results_df, _ = perform_enhanced_grid_analysis(data1, data2, alpha)
        results.append({
            'x_value': x_val,
            'filename': f,
            'L_2': metrics['L2_distance'],
            'chi2': metrics['chi2_statistic'],
            'p_value': metrics['p_value'],
            'wasserstein': metrics['wasserstein_euclidean'],
            'wasserstein_kl': metrics['wasserstein_kl'],
        })
        l2_per_triangle = np.sqrt(results_df['residual_squared'].values)
        ternary_l2_arrays.append(l2_per_triangle)
        ternary_results_dfs.append(results_df)
        x_values.append(x_val)

    # --- Compute robust colorbar limits (1st/99th percentiles) ---
    all_l2 = np.concatenate(ternary_l2_arrays)
    vmin = np.nanpercentile(all_l2, 1)
    vmax = np.nanpercentile(all_l2, 99)
    linthresh = max(1e-5, (vmax-vmin)/100)
    norm = SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    # --- Convert results to DataFrame and sort by x_value ---
    results_df = pd.DataFrame(results).sort_values('x_value')
    x_values_sorted = results_df['x_value'].values

    # --- Plot Wasserstein & Wasserstein-KL ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1, ax2 = axes
    fig.suptitle(f'Comparison Metrics vs {pattern} (TRUTH vs pattern-matched files)', fontsize=16, fontweight='bold')

    # --- Plot 1: Wasserstein distances ---
    ax1.plot(results_df['x_value'], results_df['wasserstein'], 'o-', label='Wasserstein', color='#3B82F6', linewidth=2)
    ax1.plot(results_df['x_value'], results_df['wasserstein_kl'], 'o-', label='Wasserstein-KL', color='#F59E42', linewidth=2)
    ax1.set_xlabel(f'{pattern} Value', fontsize=13)
    ax1.set_ylabel('Distance Value', fontsize=13)
    ax1.set_title('Wasserstein Distances', fontsize=14)
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: L2, Chi2, and p-value (secondary y-axis) ---
    ax2.plot(results_df['x_value'], results_df['L_2'], 's-', label='L²', color='#6366F1', linewidth=2)
    ax2.plot(results_df['x_value'], results_df['chi2'], 's-', label='chi Statistic', color='#F59E42', linewidth=2)
    ax2.set_xlabel(f'{pattern} Value', fontsize=13)
    ax2.set_ylabel('Value (linear scale)', fontsize=13)
    ax2.set_title(r'$L^2$, $\,\chi^2$ Statistic and p-value', fontsize=14)
    ax2.grid(True, alpha=0.3)

    ax2b = ax2.twinx()
    ax2b.plot(results_df['x_value'], results_df['p_value'], '^-k', label='p-value', linewidth=2)
    ax2b.set_yscale('log')
    ax2b.set_ylabel('p-value (log scale)', fontsize=13)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / 'metrics_vs_pattern.png', dpi=300)
    plt.close(fig)

    # --- Grid of ternary L2 plots ---
    import math
    n_plots = len(x_values)
    ncols = 3
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    fig.suptitle(f'L² Distance per Triangle (TRUTH vs pattern-matched files)', fontsize=18, fontweight='bold', y=0.92)

    for idx, (results_df_tri, x_val) in enumerate(zip(ternary_results_dfs, x_values)):
        row, col = divmod(idx, ncols)
        ax = axes[row, col] if nrows > 1 else axes[col]
        # --- Plot using the notebook's function, but override color normalization ---
        # plot_ternary_base_twiss(ax, alpha)  # You may need to import this if not in scope
        # For now, just plot triangles
        for i, row_df in results_df_tri.iterrows():
            has_data = (row_df['count_data'] > 0) or (row_df['count_model'] > 0)
            trianglex, triangley, _ = return_triangle_coord(
                row_df['T1_bounds'][0], row_df['T1_bounds'][1],
                row_df['T2_bounds'][0], row_df['T2_bounds'][1],
                row_df['T3_bounds'][0], row_df['T3_bounds'][1]
            )
            if not has_data:
                ax.fill(trianglex, triangley, color='white', edgecolor='grey', alpha=0.5)
            else:
                color = cmap(norm(np.sqrt(row_df['residual_squared'])))
                ax.fill(trianglex, triangley, color=color, edgecolor='none', alpha=0.9)
        ax.set_title(f"L² (Truth vs {pattern}={x_val:.3f})", fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for idx in range(n_plots, nrows*ncols):
        row, col = divmod(idx, ncols)
        fig.delaxes(axes[row, col] if nrows > 1 else axes[col])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('L² Distance (shared scale, SymLogNorm)', fontsize=13)

    plt.subplots_adjust(left=0.05, right=0.9, top=0.88, bottom=0.08, wspace=0.25, hspace=0.25)
    plt.savefig(output_dir / 'ternary_L2_grid.png', dpi=300)
    plt.close(fig)

    print(f"Analysis complete! Plots saved to {output_dir}")

if __name__ == '__main__':
    main() 