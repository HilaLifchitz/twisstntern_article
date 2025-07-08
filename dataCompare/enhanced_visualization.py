#!/usr/bin/env python
"""
Enhanced visualization functions with smart hatch handling and improved colorbar behavior.

This module provides enhanced versions of the main visualization functions with:
1. hatch parameter (boolean) to control empty triangle appearance
2. All colorbars starting from 1 instead of 0
3. Improved professional styling
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import pandas as pd
from pathlib import Path

# Import from twisstntern package
import twisstntern
from twisstntern.utils import return_triangle_coord, h
from twisstntern.visualization import get_professional_colormap


def draw_empty_triangle_enhanced(ax, trianglex, triangley, hatch=True):
    """
    Draw an empty triangle with optional hatching.
    
    Args:
        ax: Matplotlib axis
        trianglex, triangley: Triangle coordinates
        hatch (bool): If True, adds hatching pattern. If False, just white fill.
    """
    if hatch:
        # Current behavior: white with hatching
        empty_triangle = Polygon(
            list(zip(trianglex, triangley)),
            closed=True,
            facecolor='white',
            edgecolor='grey',
            hatch='///',
            linewidth=0.5
        )
    else:
        # New behavior: plain white
        empty_triangle = Polygon(
            list(zip(trianglex, triangley)),
            closed=True,
            facecolor='white',
            edgecolor='none',
            linewidth=0
        )
    ax.add_patch(empty_triangle)


def plot_ternary_heatmap_data_enhanced(data, granularity, file_name, hatch=True, grid_color="#3E3E3E"):
    """
    Enhanced version of plot_ternary_heatmap_data with hatch control.
    
    Args:
        data: DataFrame with T1, T2, T3 columns
        granularity: Grid granularity (float or string)
        file_name: Output filename prefix
        hatch (bool): If True, empty triangles are hatched. If False, plain white.
        grid_color: Color for grid lines
    
    Returns:
        matplotlib.figure.Figure object
    """
    if granularity == "superfine":
        alpha = 0.05
    elif granularity == "fine":
        alpha = 0.1
    elif granularity == "coarse":
        alpha = 0.25
    else:
        alpha = float(granularity)

    def create_triangular_grid(alpha):
        triangles = []
        steps = int(1 / alpha)
        for k in range(steps):
            a1 = round(k * alpha, 10)
            b1 = round((k + 1) * alpha, 10)
            T2_upper_limit = round(1 - k * alpha, 10)
            T2_steps = round(T2_upper_limit / alpha)
            a3_1 = round(1 - (k + 1) * alpha, 10)
            b3_1 = round(1 - k * alpha, 10)
            for T2_step in range(T2_steps):
                a2 = round(T2_step * alpha, 10)
                b2 = round((T2_step + 1) * alpha, 10)
                if a3_1 >= 0:
                    triangles.append({
                        'T1': (a1, b1),
                        'T2': (a2, b2),
                        'T3': (a3_1, b3_1)
                    })
                a3_2 = round(a3_1 - alpha, 10)
                b3_2 = round(b3_1 - alpha, 10)
                if a3_2 >= 0:
                    triangles.append({
                        'T1': (a1, b1),
                        'T2': (a2, b2),
                        'T3': (a3_2, b3_2)
                    })
                a3_1 = a3_2
                b3_1 = b3_2
        return triangles

    def n_twisstcompare(a1, b1, a2, b2, a3, b3, data):
        if a1 == 0:
            condition_a1 = a1 <= data.T1
        else:
            condition_a1 = a1 < data.T1
        if a2 == 0:
            condition_a2 = a2 <= data.T2
        else:
            condition_a2 = a2 < data.T2
        if a3 == 0:
            condition_a3 = a3 <= data.T3
        else:
            condition_a3 = a3 < data.T3
        n = len(data[(condition_a1 & (data.T1 <= b1)) &
                     (condition_a2 & (data.T2 <= b2)) &
                     (condition_a3 & (data.T3 <= b3))])
        return n

    triangles = create_triangular_grid(alpha)
    counts = []
    for triangle in triangles:
        count = n_twisstcompare(
            triangle['T1'][0], triangle['T1'][1],
            triangle['T2'][0], triangle['T2'][1],
            triangle['T3'][0], triangle['T3'][1],
            data
        )
        counts.append(count)
    counts = np.array(counts)
    values = counts
    
    # Always start from 1 instead of 0
    vmin = 1
    vmax = np.max(values) if np.any(values > 0) else 1

    # Use professional colormap
    cmap = get_professional_colormap(style="viridis", truncate=False)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes()
    triangle_x = [0, -0.5, 0.5, 0]
    triangle_y = [h, 0, 0, h]
    ax.plot(triangle_x, triangle_y, color="k", linewidth=1)

    # Plot filled triangles
    for idx, triangle in enumerate(triangles):
        (a1, b1), (a2, b2), (a3, b3) = triangle['T1'], triangle['T2'], triangle['T3']
        trianglex, triangley, _ = return_triangle_coord(a1, b1, a2, b2, a3, b3)
        
        if values[idx] == 0:
            # Use enhanced empty triangle function with hatch control
            draw_empty_triangle_enhanced(ax, trianglex, triangley, hatch=hatch)
        else:
            # Use colormap for non-zero values
            color = cmap(norm(values[idx]))
            ax.fill(trianglex, triangley, color=color, edgecolor='none')

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Colorbar (starts from 1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = inset_axes(ax, width="3%", height="80%", loc='center right',
                     bbox_to_anchor=(0.05, 0, 1, 1), bbox_transform=ax.transAxes, borderpad=1)
    cbar = plt.colorbar(sm, cax=cax)
    cbar.ax.set_title('Count', fontsize=10, pad=6)
    cbar.set_ticks([vmin, vmax])
    cbar.ax.set_yticklabels([f"{vmin}", f"{vmax}"])

    hatch_suffix = "_hatched" if hatch else "_plain"
    title = f"{file_name}_enhanced_heatmap{hatch_suffix}.png"
    fig.savefig(title, dpi=300, bbox_inches="tight")
    abs_path = os.path.abspath(title)
    print(f"Enhanced heatmap saved as: {abs_path}")
    return fig


def plot_heatmap_data_enhanced(ax, results_df, alpha, data_type='count_data', title='Data', vmax=None, vmin=None, override_cmap=None, hatch_data=True, hatch_residuals=True, hatch_L2=True):
    """
    Enhanced version of plot_heatmap_data with separate hatch control for different plot types.
    
    Args:
        ax: Matplotlib axis
        results_df: Results dataframe  
        alpha: Grid granularity
        data_type: Column to plot
        title: Plot title
        vmax, vmin: Colorbar limits
        override_cmap: Custom colormap
        hatch_data (bool): If True, empty triangles in data/model plots are hatched
        hatch_residuals (bool): If True, empty triangles in residual plots are hatched
        hatch_L2 (bool): If True, empty triangles in L2 plots are hatched
    """
    plot_ternary_base_enhanced(ax, alpha)
    
    # Determine which hatch setting to use based on plot type
    if data_type == 'count_residual' or 'residual' in data_type:
        current_hatch = hatch_residuals
        cmap = sns.color_palette("RdBu_r", as_cmap=True) if override_cmap is None else override_cmap
        vmax_val = results_df[data_type].abs().max()
        vmin_val = -vmax_val
        norm = Normalize(vmin=vmin_val, vmax=vmax_val)
        values = results_df[data_type]
        empty_logic = lambda row: row['count_data'] == 0 and row['count_model'] == 0
    elif data_type in ['count_data', 'count_model']:
        current_hatch = hatch_data
        cmap = plt.get_cmap("viridis") if override_cmap is None else override_cmap
        vmax_val = vmax if vmax is not None else results_df[data_type].max()
        # Start from 1 instead of 0 for count data
        vmin_val = vmin if vmin is not None else 1
        norm = Normalize(vmin=vmin_val, vmax=vmax_val)
        values = results_df[data_type]
        empty_logic = lambda row: row[data_type] == 0
    elif data_type == 'l2':
        current_hatch = hatch_L2
        cmap = override_cmap if override_cmap is not None else plt.get_cmap("magma")
        l2_per_triangle = np.sqrt(results_df['residual_squared'].values)
        vmax_val = vmax if vmax is not None else np.max(l2_per_triangle)
        # Start from 1 instead of 0 for L2 distances  
        vmin_val = vmin if vmin is not None else 1
        norm = Normalize(vmin=vmin_val, vmax=vmax_val)
        values = l2_per_triangle
        empty_logic = lambda row: (row['count_data'] == 0 and row['count_model'] == 0)
    else:
        current_hatch = hatch_data  # Default to data hatch setting
        cmap = plt.get_cmap("viridis") if override_cmap is None else override_cmap
        vmax_val = vmax if vmax is not None else results_df[data_type].max()
        # Start from 1 instead of 0
        vmin_val = vmin if vmin is not None else 1
        norm = Normalize(vmin=vmin_val, vmax=vmax_val)
        values = results_df[data_type]
        empty_logic = lambda row: row['count_data'] == 0

    for idx, row in results_df.iterrows():
        value = values[idx]
        trianglex, triangley, direction = twisstntern.utils.return_triangle_coord(
            row['T1_bounds'][0], row['T1_bounds'][1],
            row['T2_bounds'][0], row['T2_bounds'][1],
            row['T3_bounds'][0], row['T3_bounds'][1]
        )
        if empty_logic(row):
            draw_empty_triangle_enhanced(ax, trianglex, triangley, hatch=current_hatch)
        else:
            color = cmap(norm(value))
            ax.fill(trianglex, triangley, color=color, edgecolor='none', alpha=0.8)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    sns.despine(ax=ax, top=True, right=True)


def plot_ternary_base_enhanced(ax, alpha):
    """
    Enhanced version of ternary base plot.
    """
    # Draw main triangle edges
    x_side_T2 = np.linspace(0, 0.5, 100)
    x_side_T3 = np.linspace(-0.5, 0, 100)
    
    ax.plot(x_side_T2, twisstntern.utils.T2(0, x_side_T2), "k", linewidth=1)
    ax.plot(x_side_T3, twisstntern.utils.T3(0, x_side_T3), "k", linewidth=1)
    ax.hlines(y=0, xmin=-0.5, xmax=0.5, color="k", linewidth=1)
    
    # Draw grid lines
    grid_color = sns.color_palette("muted")[7]
    for i in range(1, int(1 / alpha)):
        y = i * alpha
        ax.hlines(y=y * h, xmin=twisstntern.utils.T1_lim(y)[0], xmax=twisstntern.utils.T1_lim(y)[1], 
                 color=grid_color, linewidth=0.8, alpha=0.6)
        
        x2 = np.linspace(twisstntern.utils.T2_lim(y)[0], twisstntern.utils.T2_lim(y)[1], 100)
        ax.plot(x2, twisstntern.utils.T2(y, x2), color=grid_color, linewidth=0.8, alpha=0.6)
        
        x3 = np.linspace(twisstntern.utils.T3_lim(y)[0], twisstntern.utils.T3_lim(y)[1], 100)
        ax.plot(x3, twisstntern.utils.T3(y, x3), color=grid_color, linewidth=0.8, alpha=0.6)
    
    # Vertical line through center
    ax.vlines(x=0, ymin=0, ymax=h, colors=grid_color, ls=':', linewidth=1.0, alpha=0.7)
    
    # Labels
    label_colors = sns.color_palette("bright", 3)
    ax.text(-0.02, 0.88, 'T1', size=13, fontweight='bold', color=label_colors[0])
    ax.text(0.54, -0.02, 'T3', size=13, fontweight='bold', color=label_colors[1]) 
    ax.text(-0.58, -0.02, 'T2', size=13, fontweight='bold', color=label_colors[2])
    
    # Remove ticks and spines
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_ternary_with_colorbar_enhanced(ax, results_df, alpha, data_type, title, vmin, vmax, cmap, cbar_label, hatch_data=True, hatch_residuals=True, hatch_L2=True):
    """
    Enhanced version with separate hatch control for different plot types and colorbar starting from 1.
    """
    plot_heatmap_data_enhanced(ax, results_df, alpha, data_type, title, vmax, vmin, cmap, 
                              hatch_data=hatch_data, hatch_residuals=hatch_residuals, hatch_L2=hatch_L2)
    
    # Ensure colorbar starts from 1 for count data
    if 'count' in data_type and vmin == 0:
        vmin = 1
        
    norm = Normalize(vmin=vmin, vmax=vmax)
    cax = inset_axes(ax, width="3%", height="60%", loc='right', borderpad=2)
    cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.ax.set_title(cbar_label, fontsize=12, fontweight='normal', pad=10)
    return cb


def create_enhanced_comparison_plot_with_hatch_control(data1, data2, results_df, statistics, alpha, 
                                                      hatch_data=True, hatch_residuals=True, hatch_L2=True, 
                                                      output_path=None):
    """
    Create enhanced comparison plot with separate hatch controls for each plot type.
    
    Args:
        data1, data2: Input datasets
        results_df: Results dataframe
        statistics: Statistics dictionary  
        alpha: Grid granularity
        hatch_data (bool): Hatch control for data/model plots
        hatch_residuals (bool): Hatch control for residual plots
        hatch_L2 (bool): Hatch control for L2 distance plots
        output_path: Output file path
    
    Returns:
        matplotlib.figure.Figure object
    """
    import matplotlib.gridspec as gridspec
    
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1.1], hspace=0.25, wspace=0.25)
    
    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])  # Data plot
    ax2 = fig.add_subplot(gs[0, 1])  # Model plot  
    ax3 = fig.add_subplot(gs[0, 2])  # Residuals plot
    ax4 = fig.add_subplot(gs[1, 0])  # L2 plot
    ax5 = fig.add_subplot(gs[1, 1:]) # Statistics plot
    
    # Get data ranges for consistent colorbars
    min_count = min(results_df['count_data'].min(), results_df['count_model'].min())
    max_count = max(results_df['count_data'].max(), results_df['count_model'].max())
    
    # Ensure count colorbar starts from 1
    if min_count == 0:
        min_count = 1
    
    cmap_seq = plt.get_cmap("viridis")
    
    # Data plot (uses hatch_data)
    plot_ternary_with_colorbar_enhanced(ax1, results_df, alpha, 'count_data', 'Data', 
                                       min_count, max_count, cmap_seq, "Count",
                                       hatch_data=hatch_data, hatch_residuals=hatch_residuals, hatch_L2=hatch_L2)
    
    # Model plot (uses hatch_data)  
    plot_ternary_with_colorbar_enhanced(ax2, results_df, alpha, 'count_model', 'Model',
                                       min_count, max_count, cmap_seq, "Count", 
                                       hatch_data=hatch_data, hatch_residuals=hatch_residuals, hatch_L2=hatch_L2)
    
    # Residuals plot (uses hatch_residuals)
    plot_heatmap_data_enhanced(ax3, results_df, alpha, 'count_residual', 'Residuals',
                              hatch_data=hatch_data, hatch_residuals=hatch_residuals, hatch_L2=hatch_L2)
    vmax_resid = results_df['count_residual'].abs().max()
    norm_resid = Normalize(vmin=-vmax_resid, vmax=vmax_resid)
    cmap_resid = sns.color_palette("RdBu_r", as_cmap=True)
    cax3 = inset_axes(ax3, width="3%", height="60%", loc='right', borderpad=2)
    cb3 = plt.colorbar(mpl.cm.ScalarMappable(norm=norm_resid, cmap=cmap_resid), cax=cax3)
    cb3.ax.set_title("Count", fontsize=12, fontweight='normal', pad=10)
    
    # L2 plot (uses hatch_L2)
    plot_heatmap_data_enhanced(ax4, results_df, alpha, 'l2', 'L² Distance per Triangle',
                              hatch_data=hatch_data, hatch_residuals=hatch_residuals, hatch_L2=hatch_L2)
    l2_per_triangle = np.sqrt(results_df['residual_squared'].values)
    vmax_l2 = np.max(l2_per_triangle) if len(l2_per_triangle) > 0 else 1
    norm_l2 = Normalize(vmin=1, vmax=vmax_l2)  # Start L2 from 1 too
    cmap_l2 = plt.get_cmap("magma")
    cax4 = inset_axes(ax4, width="3%", height="60%", loc='right', borderpad=2)
    cb4 = plt.colorbar(mpl.cm.ScalarMappable(norm=norm_l2, cmap=cmap_l2), cax=cax4)
    cb4.ax.set_title("L² Distance", fontsize=12, fontweight='normal', pad=10)
    
    # Statistics plot
    plot_enhanced_statistics(ax5, statistics, hatch_data, hatch_residuals, hatch_L2)
    
    # Add overall title with hatch info
    hatch_status = f"Data: {'✓' if hatch_data else '✗'}, Residuals: {'✓' if hatch_residuals else '✗'}, L2: {'✓' if hatch_L2 else '✗'}"
    fig.suptitle(f'Enhanced Topology Weight Analysis (Hatch - {hatch_status})', fontsize=16, y=0.95)
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Enhanced comparison plot saved as: {os.path.abspath(output_path)}")
    
    return fig


def plot_enhanced_statistics(ax, statistics, hatch_data, hatch_residuals, hatch_L2):
    """
    Plot enhanced statistics with hatch configuration info.
    """
    # Clear the axis
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Create statistics text
    stats_text = f"""
Enhanced Analysis Statistics:

Statistical Measures:
• Sinkhorn Distance: {statistics.get('sinkhorn_distance', 'N/A'):.6f}
• Mean Absolute Error: {statistics.get('mae', 'N/A'):.4f}  
• Root Mean Square Error: {statistics.get('rmse', 'N/A'):.4f}
• Total Data Points: {statistics.get('total_data_points', 'N/A'):,}
• Total Model Points: {statistics.get('total_model_points', 'N/A'):,}

Triangle Coverage:
• Non-empty Triangles: {statistics.get('num_non_empty_triangles', 'N/A')}
• Empty Triangles: {statistics.get('num_empty_triangles', 'N/A')}

Visualization Settings:
• Data/Model Empty Triangles: {'Hatched' if hatch_data else 'Plain White'}
• Residual Empty Triangles: {'Hatched' if hatch_residuals else 'Plain White'}  
• L² Empty Triangles: {'Hatched' if hatch_L2 else 'Plain White'}
• Colorbar Range: Starts from 1 (not 0)
"""
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))


# Example usage function
def demo_enhanced_visualization():
    """
    Demonstrate the enhanced visualization with separate hatch controls.
    """
    print("🎨 Enhanced Visualization Demo with Separate Hatch Controls")
    print("=" * 60)
    print("Key features:")
    print("✅ hatch_data=True      -> Data/Model empty triangles with hatching")
    print("✅ hatch_data=False     -> Data/Model empty triangles plain white") 
    print("✅ hatch_residuals=True -> Residual empty triangles with hatching")
    print("✅ hatch_residuals=False-> Residual empty triangles plain white")
    print("✅ hatch_L2=True        -> L² empty triangles with hatching")
    print("✅ hatch_L2=False       -> L² empty triangles plain white")
    print("✅ All colorbars start from 1 instead of 0")
    print("✅ Professional styling and colors")
    print()
    print("Example usage:")
    print("# All hatched (current behavior)")
    print("plot_ternary_heatmap_data_enhanced(data, 0.1, 'output', hatch=True)")
    print()
    print("# Mixed hatching - data hatched, residuals plain, L2 hatched")
    print("create_enhanced_comparison_plot_with_hatch_control(")
    print("    data1, data2, results_df, stats, alpha=0.1,")
    print("    hatch_data=True, hatch_residuals=False, hatch_L2=True)")
    print() 
    print("# All plain white empty triangles")
    print("create_enhanced_comparison_plot_with_hatch_control(")
    print("    data1, data2, results_df, stats, alpha=0.1,")
    print("    hatch_data=False, hatch_residuals=False, hatch_L2=False)")


if __name__ == "__main__":
    demo_enhanced_visualization() 