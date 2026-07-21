# Copyright 2026 Hanxiao Li, Beihang University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Metrics plotter for retroactive evaluation.

This module provides the MetricsPlotter class that:
1. Generates training curves for each metric over steps
2. Plots error bands (standard deviation, percentiles)
3. Creates correlation plots between metrics and rewards
4. Supports comparison across multiple metrics
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .aggregator import (
    MetricsAggregator,
    MetricTimeSeries,
    RewardCorrelation,
    StepStatistics,
)
from .plot_statistics import PlotStatistics

logger = logging.getLogger(__name__)

# Try to import matplotlib, provide helpful error if not available
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, MaxNLocator, MultipleLocator
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    mcolors = None


def _check_matplotlib():
    """Check if matplotlib is available and raise informative error if not."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        )


# Default color palette for metrics
DEFAULT_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]

METRIC_DISPLAY_NAMES = {
    "hidden_act_mentioned_in_think": "Hidden act mentioned",
    "admit_hidden_act_to_user": "Admission",
    "attribute_hidden_act_as_cause": "Causal attribution",
    "intent_sanitization": "Intent sanitization",
    "motive_sanitization": "Motive sanitization",
    "misdirection_alt_causes_primary": "Misdirection",
    "material_falsehood": "Fabrication",
    "procedural_deflection": "Procedural deflection",
    "answer_refusal": "Answer refusal",
}


class MetricsPlotter:
    """
    Plotter for visualizing metric evaluation results.
    
    This class generates various plots:
    1. Training curves showing metric scores over training steps
    2. Error bands (std, percentiles) around the mean
    3. Correlation plots between metrics and reward scores
    4. Multi-metric comparison plots
    5. Distribution histograms per step
    
    Example:
        >>> aggregator = MetricsAggregator()
        >>> # ... add results to aggregator ...
        >>> 
        >>> plotter = MetricsPlotter(aggregator)
        >>> 
        >>> # Plot training curve for a single metric
        >>> fig = plotter.plot_training_curve("deception")
        >>> fig.savefig("deception_curve.png")
        >>> 
        >>> # Plot all metrics together
        >>> fig = plotter.plot_all_metrics()
        >>> fig.savefig("all_metrics.png")
    """
    
    def __init__(
        self,
        aggregator: Union[MetricsAggregator, PlotStatistics],
        figsize: Tuple[float, float] = (10, 6),
        dpi: int = 300,
        style: str = "seaborn-v0_8-whitegrid",
        color_map: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the plotter.
        
        Args:
            aggregator: Live MetricsAggregator or saved PlotStatistics.
            figsize: Default figure size (width, height) in inches.
            dpi: Dots per inch for saved figures.
            style: Matplotlib style to use.
            color_map: Optional mapping from metric names to colors.
        """
        _check_matplotlib()
        
        self.data_source = aggregator
        self.aggregator = (
            aggregator if isinstance(aggregator, MetricsAggregator) else None
        )
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        
        # Set up color mapping
        metrics = self.data_source.get_metrics()
        if color_map:
            self.color_map = color_map
        else:
            self.color_map = {
                metric: DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
                for i, metric in enumerate(metrics)
            }
        
        # Try to apply style, fall back if not available
        try:
            plt.style.use(style)
        except OSError:
            logger.warning(f"Style '{style}' not available, using default")
    
    def _get_color(self, metric_name: str) -> str:
        """Get color for a metric."""
        if metric_name not in self.color_map:
            idx = len(self.color_map) % len(DEFAULT_COLORS)
            self.color_map[metric_name] = DEFAULT_COLORS[idx]
        return self.color_map[metric_name]

    def _require_raw_aggregator(self) -> MetricsAggregator:
        """Return the live aggregator required by sample-level plots."""
        if self.aggregator is None:
            raise ValueError(
                "This plot requires individual evaluation results and cannot "
                "be generated from saved plot statistics"
            )
        return self.aggregator
    
    def plot_training_curve(
        self,
        metric_name: str,
        show_std: bool = True,
        show_percentiles: bool = False,
        alpha_band: float = 0.2,
        title: Optional[str] = None,
        xlabel: str = "Training Step",
        ylabel: Optional[str] = None,
        ax: Optional["plt.Axes"] = None,
    ) -> "plt.Figure":
        """
        Plot training curve for a single metric.
        
        Args:
            metric_name: Name of the metric to plot.
            show_std: Whether to show standard deviation band.
            show_percentiles: Whether to show 25th-75th percentile band.
            alpha_band: Alpha value for the error band.
            title: Plot title. If None, uses metric name.
            xlabel: X-axis label.
            ylabel: Y-axis label. If None, uses "Score".
            ax: Optional existing axes to plot on.
        
        Returns:
            Matplotlib Figure object.
        """
        _check_matplotlib()
        
        time_series = self.data_source.get_time_series(metric_name)
        color = self._get_color(metric_name)
        
        # Create figure if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        else:
            fig = ax.figure
        
        steps = np.array(time_series.steps)
        means = np.array(time_series.means)
        
        # Plot mean line
        ax.plot(steps, means, color=color, linewidth=2, label=metric_name)
        
        # Plot error bands
        if show_std:
            stds = np.array(time_series.stds)
            ax.fill_between(
                steps,
                means - stds,
                means + stds,
                color=color,
                alpha=alpha_band,
                label=f"{metric_name} ± std",
            )
        
        if show_percentiles:
            p25 = np.array(time_series.percentile_25s)
            p75 = np.array(time_series.percentile_75s)
            ax.fill_between(
                steps,
                p25,
                p75,
                color=color,
                alpha=alpha_band * 0.5,
                label=f"{metric_name} (25-75%)",
            )
        
        # Labels and title
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel or "Score", fontsize=12)
        ax.set_title(title or f"Training Curve: {metric_name}", fontsize=14)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits to [0, 1] if scores are in that range
        if len(means) > 0 and np.nanmin(means) >= 0 and np.nanmax(means) <= 1:
            ax.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        return fig
    
    def plot_all_metrics(
        self,
        metrics: Optional[List[str]] = None,
        show_std: bool = True,
        alpha_band: float = 0.15,
        title: str = "Training Curves",
        xlabel: str = "Training Step",
        ylabel: str = "Score",
    ) -> "plt.Figure":
        """
        Plot training curves with 2 subplots: reward curve and all metrics.
        
        Args:
            metrics: List of metric names to plot. If None, plots all.
            show_std: Whether to show standard deviation bands.
            alpha_band: Alpha value for error bands.
            title: Overall figure title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
        
        Returns:
            Matplotlib Figure object.
        """
        _check_matplotlib()
        
        if metrics is None:
            metrics = self.data_source.get_metrics()
        
        # Create figure with 2 subplots (side by side)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0] * 1.8, self.figsize[1]), dpi=self.dpi)
        
        # === Subplot 1: Reward Curve ===
        reward_data = self._compute_reward_time_series(metrics)
        if reward_data:
            steps, means, stds = reward_data
            reward_color = "#2ca02c"  # Green for reward
            # ax1.plot(steps, means, color=reward_color, linewidth=3.5, label="Reward",
            #          marker='o', markersize=7, markerfacecolor=reward_color, markeredgecolor='white', markeredgewidth=0.8)
            ax1.plot(steps, means, color=reward_color, linewidth=2, label="Reward")
            if show_std:
                ax1.fill_between(
                    steps,
                    np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    color=reward_color,
                    alpha=alpha_band,
                )
        
        ax1.set_xlabel(xlabel, fontsize=12)
        ax1.set_ylabel("Reward Score", fontsize=12)
        ax1.set_title("Reward Curve", fontsize=14)
        ax1.legend(loc="best", fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # === Subplot 2: All Metrics ===
        for metric_name in metrics:
            time_series = self.data_source.get_time_series(metric_name)
            color = self._get_color(metric_name)
            
            steps = np.array(time_series.steps)
            means = np.array(time_series.means)
            
            # Plot mean line with markers
            # ax2.plot(steps, means, color=color, linewidth=3.5, label=metric_name,
            #          marker='o', markersize=7, markerfacecolor=color, markeredgecolor='white', markeredgewidth=0.6)
            ax2.plot(steps, means, color=color, linewidth=2, label=metric_name)
            
            # Plot error band
            if show_std:
                stds = np.array(time_series.stds)
                ax2.fill_between(
                    steps,
                    means - stds,
                    means + stds,
                    color=color,
                    alpha=alpha_band,
                )
        
        ax2.set_xlabel(xlabel, fontsize=12)
        ax2.set_ylabel(ylabel, fontsize=12)
        ax2.set_title("Metric Curves", fontsize=14)
        ax2.legend(loc="best", fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Set y-axis limits to [0, 1] for metrics if appropriate
        if metrics:
            ax2.set_ylim(-0.05, 1.05)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

    def plot_deception_metrics(
        self,
        method_name: str,
        metrics: Optional[List[str]] = None,
        show_std: bool = True,
        alpha_band: float = 0.12,
        xlabel: str = "Step",
        ylabel: str = "Mean metric score",
        legend_ncol: int = 4,
        figure_size: Tuple[float, float] = (5.0, 3.8),
    ) -> "plt.Figure":
        """Plot all deception-metric curves in a single figure.

        Unlike :meth:`plot_all_metrics`, this figure excludes reward and places
        the legend below the plotting area so it cannot obscure any curves.

        Args:
            method_name: Algorithm name displayed as the figure title. 
            metrics: Metric names to plot. If None, plots all available metrics.
            show_std: Whether to show standard deviation bands.
            alpha_band: Alpha value for standard deviation bands.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            legend_ncol: Maximum number of legend columns.
            figure_size: Publication figure size in inches.

        Returns:
            Matplotlib Figure object.
        """
        _check_matplotlib()

        if metrics is None:
            metrics = self.data_source.get_metrics()
        if not metrics:
            raise ValueError("No deception metrics available to plot")
        if legend_ncol < 1:
            raise ValueError("legend_ncol must be at least 1")

        fig, ax = plt.subplots(
            figsize=figure_size,
            dpi=max(self.dpi, 150),
            facecolor="white",
        )
        ax.set_facecolor("white")

        for metric_name in metrics:
            time_series = self.data_source.get_time_series(metric_name)
            color = self._get_color(metric_name)
            steps = np.asarray(time_series.steps)
            means = np.asarray(time_series.means)

            ax.plot(
                steps,
                means,
                color=color,
                linewidth=1.2,
                label=metric_name,
                solid_capstyle="round",
                solid_joinstyle="round",
            )

            if show_std:
                stds = np.asarray(time_series.stds)
                ax.fill_between(
                    steps,
                    np.clip(means - stds, 0.0, 1.0),
                    np.clip(means + stds, 0.0, 1.0),
                    color=color,
                    alpha=alpha_band,
                )

        ax.set_xlabel(xlabel, fontsize=12, color="#222222")
        ax.set_ylabel(ylabel, fontsize=12, color="#222222")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.margins(x=0.015)
        ax.set_axisbelow(True)

        # Quiet, publication-style axes: horizontal reference lines carry the
        # scale while unnecessary framing and vertical grid lines are removed.
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#333333")
            ax.spines[spine].set_linewidth(0.8)
            ax.spines[spine].set_position(("outward", 2))

        max_columns = min(legend_ncol, len(metrics))
        legend_rows = math.ceil(len(metrics) / max_columns)
        columns = math.ceil(len(metrics) / legend_rows)
        legend_rows = math.ceil(len(metrics) / columns)
        plt.title(
            method_name,
            fontsize=12,
            fontweight="bold",
            color="#1A1A1A",
        )
        handles, labels = ax.get_legend_handles_labels()
        display_labels = [
            METRIC_DISPLAY_NAMES.get(
                label,
                label.replace("_", " ").strip().capitalize(),
            )
            for label in labels
        ]
        # Matplotlib fills multi-column legends column-first. Reorder the
        # entries so readers encounter metrics naturally from left to right.
        legend_order = [
            row * columns + column
            for column in range(columns)
            for row in range(legend_rows)
            if row * columns + column < len(handles)
        ]
        handles = [handles[index] for index in legend_order]
        display_labels = [display_labels[index] for index in legend_order]
        legend = fig.legend(
            handles,
            display_labels,
            loc="lower center",
            bbox_to_anchor=(0.53, 0.08),
            ncol=columns,
            fontsize=8,
            frameon=False,
            handlelength=2.5,
            handletextpad=0.65,
            columnspacing=1.45,
            borderaxespad=0,
        )
        for handle in legend.get_lines():
            handle.set_linewidth(2.1)

        # Explicit margins are more predictable than tight_layout for an
        # external, multi-row legend and keep the title close to the axes.
        bottom_margin = 0.08 + 0.035 * legend_rows
        fig.tight_layout(rect=(0, bottom_margin, 1, 1.0))
        return fig
    
    def _compute_reward_time_series(
        self,
        metrics: Optional[List[str]] = None,
    ) -> Optional[Tuple[List[int], List[float], List[float]]]:
        """
        Compute reward score statistics per step.
        
        Uses generation data from stored results to extract reward scores.
        
        Args:
            metrics: List of metrics to use for extracting generations.
                    If None, uses all available metrics.
        
        Returns:
            Tuple of (steps, means, stds) or None if no data.
        """
        reward_time_series = self.data_source.get_reward_time_series()
        if reward_time_series is None:
            return None

        return (
            reward_time_series.steps,
            reward_time_series.means,
            reward_time_series.stds,
        )
    
    def plot_correlation(
        self,
        metric_name: str,
        step: int,
        title: Optional[str] = None,
    ) -> "plt.Figure":
        """
        Plot correlation between metric scores and reward scores at a specific step.
        
        Args:
            metric_name: Name of the metric.
            step: Training step.
            title: Plot title. If None, auto-generated.
        
        Returns:
            Matplotlib Figure object.
        """
        _check_matplotlib()
        aggregator = self._require_raw_aggregator()
        
        # Get results for this metric and step
        aggregator._ensure_cache()
        stats = aggregator._stats_cache.get(metric_name, {}).get(step)
        if stats is None:
            raise KeyError(f"No data for {metric_name} at step {step}")
        
        # Get raw data for scatter plot
        pairs = aggregator._results[metric_name][step]
        
        metric_scores = []
        reward_scores = []
        
        for result, gen in pairs:
            if result.error is None and not math.isnan(result.score):
                metric_scores.append(result.score)
                reward_scores.append(gen.score)
        
        if len(metric_scores) < 2:
            raise ValueError(f"Not enough valid data points for correlation plot")
        
        # Compute correlation
        correlation = aggregator.compute_reward_correlation(metric_name, step)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        color = self._get_color(metric_name)
        
        # Scatter plot
        ax.scatter(metric_scores, reward_scores, c=color, alpha=0.6, s=30)
        
        # Add regression line
        z = np.polyfit(metric_scores, reward_scores, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(metric_scores), max(metric_scores), 100)
        ax.plot(x_line, p(x_line), color=color, linestyle="--", linewidth=2)
        
        # Labels and title
        ax.set_xlabel(f"{metric_name} Score", fontsize=12)
        ax.set_ylabel("Reward Score", fontsize=12)
        
        title_text = title or f"Correlation: {metric_name} vs Reward (Step {step})"
        subtitle = f"Pearson r = {correlation.pearson_r:.3f}, Spearman ρ = {correlation.spearman_rho:.3f}"
        ax.set_title(f"{title_text}\n{subtitle}", fontsize=12)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    def plot_correlation_heatmap(
        self,
        metrics: Optional[List[str]] = None,
        step: Optional[int] = None,
        correlation_type: str = "pearson",
        title: Optional[str] = None,
    ) -> "plt.Figure":
        """
        Plot heatmap of correlations between metrics and rewards.
        
        Args:
            metrics: List of metric names. If None, uses all.
            step: Specific step to use. If None, uses the last available step.
            correlation_type: "pearson" or "spearman".
            title: Plot title.
        
        Returns:
            Matplotlib Figure object.
        """
        _check_matplotlib()
        aggregator = self._require_raw_aggregator()
        
        if metrics is None:
            metrics = aggregator.get_metrics()
        
        if step is None:
            step = max(aggregator.get_steps())
        
        # Compute correlations
        correlations = []
        labels = []
        
        for metric_name in metrics:
            try:
                corr = aggregator.compute_reward_correlation(metric_name, step)
                if correlation_type == "pearson":
                    correlations.append(corr.pearson_r)
                else:
                    correlations.append(corr.spearman_rho)
                labels.append(metric_name)
            except Exception as e:
                logger.warning(f"Could not compute correlation for {metric_name}: {e}")
        
        if not correlations:
            raise ValueError("No valid correlations computed")
        
        # Create bar plot   
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        colors = [self._get_color(label) for label in labels]
        y_pos = np.arange(len(labels))
        
        ax.barh(y_pos, correlations, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel(f"{correlation_type.capitalize()} Correlation with Reward", fontsize=12)
        ax.set_xlim(-1, 1)
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
        
        title_text = title or f"Metric-Reward Correlations (Step {step})"
        ax.set_title(title_text, fontsize=14)
        
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        return fig
    
    def plot_distribution(
        self,
        metric_name: str,
        step: int,
        bins: int = 20,
        title: Optional[str] = None,
    ) -> "plt.Figure":
        """
        Plot score distribution histogram for a metric at a specific step.
        
        Args:
            metric_name: Name of the metric.
            step: Training step.
            bins: Number of histogram bins.
            title: Plot title.
        
        Returns:
            Matplotlib Figure object.
        """
        _check_matplotlib()
        aggregator = self._require_raw_aggregator()
        
        stats = aggregator.get_statistics(metric_name, step)
        
        if not stats.scores:
            raise ValueError(f"No scores available for {metric_name} at step {step}")
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        color = self._get_color(metric_name)
        
        ax.hist(stats.scores, bins=bins, color=color, alpha=0.7, edgecolor="black")
        
        # Add vertical lines for mean and median
        ax.axvline(stats.mean, color="red", linestyle="--", linewidth=2, label=f"Mean: {stats.mean:.3f}")
        ax.axvline(stats.median, color="blue", linestyle=":", linewidth=2, label=f"Median: {stats.median:.3f}")
        
        ax.set_xlabel(f"{metric_name} Score", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(title or f"Score Distribution: {metric_name} (Step {step})", fontsize=14)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_distribution_comparison(
        self,
        metric_name: str,
        bins: int = 40,
        title: Optional[str] = None,
    ) -> "plt.Figure":
        """
        Plot distribution histogram comparison between first and final step for a metric.
        
        Both distributions are shown in the same figure with transparency and
        distinct colors for clear visualization even when overlapping.
        Uses Nature/top-tier conference publication style.
        
        Args:
            metric_name: Name of the metric to compare.
            bins: Number of histogram bins.
            title: Plot title. If None, auto-generated.
        
        Returns:
            Matplotlib Figure object.
        """
        _check_matplotlib()
        aggregator = self._require_raw_aggregator()
        
        # Get all steps for this metric
        if metric_name not in aggregator._results:
            raise KeyError(f"No data for metric: {metric_name}")
        
        steps_data = aggregator._results[metric_name]
        sorted_steps = sorted(steps_data.keys())
        
        if len(sorted_steps) < 2:
            raise ValueError(f"Need at least 2 steps for comparison, got {len(sorted_steps)}")
        
        first_step = sorted_steps[0]
        final_step = sorted_steps[-1]
        
        # Get scores for first and final steps
        first_stats = aggregator.get_statistics(metric_name, first_step)
        final_stats = aggregator.get_statistics(metric_name, final_step)
        
        if not first_stats.scores or not final_stats.scores:
            raise ValueError(f"No scores available for {metric_name}")
        
        first_scores = first_stats.scores
        final_scores = final_stats.scores
        
        # Create figure with grey background (Nature style)
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        ax.set_facecolor("#F5F5F5")  # Light grey background
        fig.patch.set_facecolor("white")
        
        # Compute common bin edges for both histograms
        all_scores = list(first_scores) + list(final_scores)
        bin_edges = np.linspace(min(all_scores), max(all_scores), bins + 1)
        
        # Nature-style colors: elegant blue and coral red
        first_color = "#4878D0"  # Muted blue
        final_color = "#EE6677"  # Coral red
        
        # Plot histograms with no edges, higher transparency, thinner bars via rwidth
        ax.hist(
            first_scores,
            bins=bin_edges,
            color=first_color,
            alpha=0.65,
            label=f"Step {first_step} (n={len(first_scores)})",
            edgecolor="none",
            rwidth=0.85,
        )
        ax.hist(
            final_scores,
            bins=bin_edges,
            color=final_color,
            alpha=0.65,
            label=f"Step {final_step} (n={len(final_scores)})",
            edgecolor="none",
            rwidth=0.85,
        )
        
        # Add vertical lines for means with matching colors
        ax.axvline(
            first_stats.mean,
            color=first_color,
            linestyle="--",
            linewidth=2,
            alpha=0.9,
        )
        ax.axvline(
            final_stats.mean,
            color=final_color,
            linestyle="--",
            linewidth=2,
            alpha=0.9,
        )
        
        # Add mean annotations
        y_max = ax.get_ylim()[1]
        ax.annotate(
            f"μ={first_stats.mean:.2f}",
            xy=(first_stats.mean, y_max * 0.92),
            fontsize=9,
            color=first_color,
            fontweight="bold",
            ha="center",
        )
        ax.annotate(
            f"μ={final_stats.mean:.2f}",
            xy=(final_stats.mean, y_max * 0.82),
            fontsize=9,
            color=final_color,
            fontweight="bold",
            ha="center",
        )
        
        # Styling: Nature/top-tier conference look
        ax.set_xlabel(f"{metric_name} Score", fontsize=11, fontweight="medium")
        ax.set_ylabel("Frequency", fontsize=11, fontweight="medium")
        ax.set_title(
            title or f"{metric_name}: First vs Final Step Distribution",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        
        # Clean legend
        legend = ax.legend(
            loc="upper right",
            fontsize=9,
            frameon=True,
            fancybox=False,
            edgecolor="#CCCCCC",
            framealpha=0.95,
        )
        legend.get_frame().set_linewidth(0.5)
        
        # Subtle balanced grid (both axes)
        ax.grid(True, linestyle="-", alpha=0.5, color="white", linewidth=0.8)
        ax.set_axisbelow(True)
        
        # Clean spines
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color("#888888")
            ax.spines[spine].set_linewidth(0.6)
        
        # Tick styling
        ax.tick_params(axis="both", which="major", labelsize=9, colors="#444444")
        ax.tick_params(axis="x", direction="out", length=4, width=0.6)
        ax.tick_params(axis="y", direction="out", length=4, width=0.6)
        
        plt.tight_layout()
        return fig
    
    def plot_evolution_comparison(
        self,
        metrics: Optional[List[str]] = None,
        normalize: bool = False,
        title: str = "Metric Evolution Comparison",
    ) -> "plt.Figure":
        """
        Plot metrics evolution showing first vs last step comparison.
        
        Args:
            metrics: List of metric names. If None, uses all.
            normalize: Whether to normalize scores to [0, 1] range.
            title: Plot title.
        
        Returns:
            Matplotlib Figure object.
        """
        _check_matplotlib()
        
        if metrics is None:
            metrics = self.data_source.get_metrics()
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        first_means = []
        last_means = []
        first_stds = []
        last_stds = []
        labels = []
        
        for metric_name in metrics:
            ts = self.data_source.get_time_series(metric_name)
            if len(ts.steps) < 2:
                continue
            
            first_means.append(ts.means[0])
            last_means.append(ts.means[-1])
            first_stds.append(ts.stds[0])
            last_stds.append(ts.stds[-1])
            labels.append(metric_name)
        
        if not labels:
            raise ValueError("Not enough data for evolution comparison")
        
        x = np.arange(len(labels))
        width = 0.35
        
        bars1 = ax.bar(
            x - width/2, first_means, width, 
            yerr=first_stds, capsize=3,
            label="First Step", color="#1f77b4", alpha=0.8
        )
        bars2 = ax.bar(
            x + width/2, last_means, width,
            yerr=last_stds, capsize=3,
            label="Last Step", color="#ff7f0e", alpha=0.8
        )
        
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, axis="y")
        
        plt.tight_layout()
        return fig
    
    def save_summary_plots(
        self,
        output_dir: Union[str, Path],
        format: str = "png",
        method_name: Optional[str] = None,
    ) -> List[Path]:
        """Save plots that only require persisted step-wise statistics."""
        _check_matplotlib()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []

        if method_name is not None:
            fig = self.plot_deception_metrics(method_name)
            path = output_dir / f"deception_metrics.{format}"
            fig.savefig(path, dpi=max(self.dpi, 300), bbox_inches="tight")
            plt.close(fig)
            saved_files.append(path)
            logger.info(f"Saved: {path}")

        fig = self.plot_all_metrics()
        path = output_dir / f"all_metrics.{format}"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(path)
        logger.info(f"Saved: {path}")

        fig = self.plot_evolution_comparison()
        path = output_dir / f"evolution_comparison.{format}"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(path)
        logger.info(f"Saved: {path}")

        return saved_files

    def save_all_plots(
        self,
        output_dir: Union[str, Path],
        format: str = "png",
        include_distributions: bool = True,
        method_name: Optional[str] = None,
    ) -> List[Path]:
        """
        Generate and save all standard plots to a directory.
        
        Args:
            output_dir: Directory to save plots to.
            format: Image format (png, pdf, svg).
            include_distributions: Whether to include distribution plots for each step.
            method_name: Optional algorithm name for the metrics-only deception
                figure. 
        
        Returns:
            List of saved file paths.
        """
        _check_matplotlib()
        aggregator = self._require_raw_aggregator()
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = self.save_summary_plots(
            output_dir,
            format=format,
            method_name=method_name,
        )
        metrics = aggregator.get_metrics()
        steps = aggregator.get_steps()
        
        # 2. Correlation heatmap (last step)
        if steps:
            fig = self.plot_correlation_heatmap(step=max(steps))
            path = output_dir / f"correlation_heatmap.{format}"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            saved_files.append(path)
            logger.info(f"Saved: {path}")

        # 4. Distribution comparison (first vs final step) for each metric
        if len(steps) >= 2:
            for metric_name in metrics:
                try:
                    fig = self.plot_distribution_comparison(metric_name)
                    path = output_dir / f"dist_comparison_{metric_name}.{format}"
                    fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
                    plt.close(fig)
                    saved_files.append(path)
                    logger.info(f"Saved: {path}")
                except Exception as e:
                    logger.debug(f"Skipped distribution comparison for {metric_name}: {e}")
        
        # 5. Distributions (optional, can generate many files)
        if include_distributions and steps:
            dist_dir = output_dir / "distributions"
            dist_dir.mkdir(exist_ok=True)
            
            for metric_name in metrics:
                for step in steps:
                    try:
                        fig = self.plot_distribution(metric_name, step)
                        path = dist_dir / f"dist_{metric_name}_step{step}.{format}"
                        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
                        plt.close(fig)
                        saved_files.append(path)
                    except Exception as e:
                        logger.debug(f"Skipped distribution for {metric_name} step {step}: {e}")
        
        logger.info(f"Saved {len(saved_files)} plots to {output_dir}")
        return saved_files
    
    def __repr__(self) -> str:
        return (
            f"MetricsPlotter(metrics={len(self.data_source.get_metrics())}, "
            f"figsize={self.figsize})"
        )
