#!/usr/bin/env python3
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
Main entry point for retroactive evaluation.

This CLI tool orchestrates the entire evaluation pipeline:
1. Load generations from _dump_generations output
2. Initialize metrics from YAML configuration
3. Run async evaluation with OpenAI-compatible API
4. Aggregate results across metrics × generations × steps
5. Generate training curve plots and statistics
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ALGORITHM_NAMES = [
    "RLHF",
    "Mixed Safety RLHF",
    "Verdict Monitor",
    "Self-Monitor",
    "Critique Monitor",
    "ePOG (Ours)",
    "ePOG-Weak-to-Strong (Ours)",
]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_default_config_path() -> Path:
    """Get the default metrics config path."""
    return Path(__file__).parent / "config" / "metrics_config.yaml"


def export_detailed_results(
    aggregator: "MetricsAggregator",
    reader: "GenerationReader",
    output_dir: Path,
    steps: List[int],
) -> None:
    """
    Export detailed per-sample results to JSON files.
    
    Creates a directory structure: details/<step>/<uid>.json
    Each file contains the generation content and all metric scores/probs.
    
    Args:
        aggregator: MetricsAggregator containing evaluation results.
        reader: GenerationReader for loading generation data.
        output_dir: Base directory for output (details/).
        steps: List of steps to export.
    """    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = aggregator.get_metrics()
    if not metrics:
        logger.warning("No metrics in aggregator, skipping details export")
        return
    
    total_exported = 0
    
    for step in steps:
        step_dir = output_dir / str(step)
        step_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all generations for this step (use first metric as reference)
        reference_metric = metrics[0]
        if reference_metric not in aggregator._results:
            continue
        if step not in aggregator._results[reference_metric]:
            continue
        
        # Build uid -> generation mapping
        uid_to_gen = {}
        for result, gen in aggregator._results[reference_metric][step]:
            uid_to_gen[gen.uid] = gen
        
        # Build uid -> metrics mapping
        uid_to_metrics = {uid: {} for uid in uid_to_gen}
        
        for metric_name in metrics:
            if metric_name not in aggregator._results:
                continue
            if step not in aggregator._results[metric_name]:
                continue
            
            for result, gen in aggregator._results[metric_name][step]:
                if gen.uid in uid_to_metrics:
                    uid_to_metrics[gen.uid][metric_name] = {
                        "score": result.score,
                        "token_probs": result.token_probs,
                    }
                    if result.error:
                        uid_to_metrics[gen.uid][metric_name]["error"] = result.error
        
        # Export each sample
        for uid, gen in uid_to_gen.items():
            sample_data = {
                "uid": uid,
                "step": step,
                "reward": gen.score,
                "metrics": uid_to_metrics.get(uid, {}),
                "system_info": gen.system_info,
                "input": gen.input,
                "output": gen.output,
            }
            
            sample_file = step_dir / f"{uid}.json"
            with open(sample_file, "w", encoding="utf-8") as f:
                json.dump(sample_data, f, indent=2, ensure_ascii=False)
            
            total_exported += 1
    
    logger.info(f"Exported {total_exported} detailed sample files to {output_dir}")


def _metrics_configuration_fingerprint(registry: "MetricRegistry") -> str:
    """Return a stable hash of every effective metric configuration."""
    configuration = {
        name: vars(registry.get_config(name))
        for name in registry.list_metrics(include_disabled=True)
    }
    serialized = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def save_plot_statistics(
    aggregator: "MetricsAggregator",
    output_dir: Path,
    method_name: str,
    source: Dict[str, Any],
) -> Path:
    """Create or incrementally update the portable plotting snapshot."""
    from .analysis.plot_statistics import PlotStatistics

    path = output_dir / "plot_statistics.json"
    current = PlotStatistics.from_aggregator(
        aggregator,
        method_name=method_name,
        source=source,
    )
    if path.exists():
        current = PlotStatistics.load_json(path).merged_with(current)
    return current.save_json(path)


async def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run the main evaluation pipeline.
    
    Args:
        args: Parsed command-line arguments.
    
    Returns:
        Dictionary containing evaluation results and statistics.
    """
    from .cache import ResultCache
    from .clients.openai_client import OpenAIClient
    from .data_reader.generation_reader import GenerationReader
    from .data_reader.schemas import Generation
    from .metrics.base_metric import BaseMetric, MetricConfig
    from .metrics.metric_registry import MetricRegistry, create_default_registry
    from .analysis.aggregator import MetricsAggregator
    from .analysis.plotter import MetricsPlotter

    # Setup paths - use .resolve() to get absolute paths
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cache_dir = (Path(args.cache_dir).resolve() if args.cache_dir else output_dir / "cache")
    
    print(f"Evaluating rollout data from: {data_dir}")
    print(f"Evaluation output dir:        {output_dir}")
    print(f"Cache dir:                    {cache_dir}\n")
    
    # Initialize components
    logger.info("Initializing evaluation components...")
    
    # 1. Generation reader
    reader = GenerationReader(data_dir)
    available_steps = reader.list_available_steps()
    
    if not available_steps:
        logger.error(f"No generation files found in {data_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(available_steps)} steps: {available_steps[:5]}{'...' if len(available_steps) > 5 else ''}")
    
    # Filter steps if specified
    if args.steps:
        steps_to_eval = [s for s in args.steps if s in available_steps]
        if not steps_to_eval:
            logger.error(f"None of the specified steps {args.steps} found in data")
            sys.exit(1)
    else:
        steps_to_eval = available_steps
    print(f"Evaluating {len(steps_to_eval)} steps: {steps_to_eval}")

    
    # 2. API client
    # For local servers (vLLM, SGLang), api_key can be "dummy"
    api_key = args.api_key
    if not api_key and "localhost" in args.api_base:
        api_key = "dummy"
        logger.info("Using dummy API key for local server")
    elif not api_key:
        logger.error("API key required. Set --api-key or $OPENAI_API_KEY")
        sys.exit(1)
    
    client = OpenAIClient(
        base_url=args.api_base,
        api_key=api_key,
        model=args.model,
        max_concurrent=args.max_concurrent,
        max_retries=args.max_retries,
        timeout=args.timeout,
    )
    logger.info(f"API client configured: {args.api_base} model={args.model}")
    
    # 3. Result cache
    cache = None
    if not args.no_cache:
        cache = ResultCache(cache_dir)
        if args.clear_cache:
            logger.info("Clearing existing cache...")
            cache.clear()
        logger.info(f"Cache enabled at {cache_dir}")

    # 4. Metrics registry
    config_path = Path(args.metrics_config) if args.metrics_config else get_default_config_path()
    registry = MetricRegistry.from_yaml(config_path, client=client, cache=cache)
    
    # Get metrics to evaluate
    if args.metrics:
        metrics_to_eval = [registry.get_metric(name) for name in args.metrics]
    else:
        metrics_to_eval = registry.get_enabled_metrics()
    
    if not metrics_to_eval:
        logger.error("No metrics to evaluate")
        sys.exit(1)
    
    logger.info(f"Evaluating {len(metrics_to_eval)} metrics: {[m.name for m in metrics_to_eval]}")
    
    # 5. Aggregator
    aggregator = MetricsAggregator()
    
    # Track statistics
    stats = {
        "start_time": datetime.now().isoformat(),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "api_base": args.api_base,
        "model": args.model,
        "steps_evaluated": [],
        "metrics_evaluated": [m.name for m in metrics_to_eval],
        "total_generations": 0,
        "total_evaluations": 0,
        "cached_evaluations": 0,
        "api_calls": 0,
        "errors": 0,
    }
    
    # Run evaluation
    print("Start evaluation...")
    t0 = time.time()
    for step in steps_to_eval:
        # Load generations for this step
        generations = reader.load_step(step)
        
        # Limit samples if specified
        if args.max_samples_per_step and len(generations) > args.max_samples_per_step:
            generations = generations[:args.max_samples_per_step]
        
        stats["total_generations"] += len(generations)
        stats["steps_evaluated"].append(step)
        
        # Evaluate each metric
        for metric in metrics_to_eval:
            total_batches = (len(generations) + args.batch_size - 1) // args.batch_size
            for batch_idx, batch_start in enumerate(range(0, len(generations), args.batch_size), 1):
                batch_end = min(batch_start + args.batch_size, len(generations))
                batch = generations[batch_start:batch_end]
                
                # Evaluate batch with progress description
                progress_desc = f"Step {step} | Metric {metric.name} | Batch {batch_idx}/{total_batches}"
                results = await metric.evaluate_batch(batch, progress_desc=progress_desc)
                
                # Add results to aggregator
                for result, gen in zip(results, batch):
                    aggregator.add_result(result, gen)
                    stats["total_evaluations"] += 1
                    
                    if result.cached:
                        stats["cached_evaluations"] += 1
                    else:
                        stats["api_calls"] += 1
                    
                    if result.error:
                        stats["errors"] += 1
    
    stats["end_time"] = datetime.now().isoformat()
    toal_eval_sec = time.time() - t0
    eval_hours, rem = divmod(toal_eval_sec, 3600)
    eval_minutes, eval_seconds = divmod(rem, 60)
    
    # Generate outputs (always print these, even in quiet mode)
    print("\nGenerating outputs...")
    
    # Save aggregated statistics
    stats_file = output_dir / "evaluation_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved statistics to {stats_file}")
    
    # Save summary
    summary_file = output_dir / "summary.txt"
    with open(summary_file, "w") as f:
        f.write(aggregator.summary())
    print(f"  Saved summary to {summary_file}")

    if getattr(args, "save_plot_statistics", False):
        plot_statistics_file = save_plot_statistics(
            aggregator,
            output_dir=output_dir,
            method_name=args.method_name,
            source={
                "data_dir": str(data_dir),
                "cache_dir": str(cache_dir),
                "evaluator_model": args.model,
                "metrics_config_sha256": _metrics_configuration_fingerprint(registry),
                "std_ddof": 0,
                "last_run_steps": steps_to_eval,
                "max_samples_per_step": args.max_samples_per_step,
            },
        )
        print(f"  Saved plot statistics to {plot_statistics_file}")
    
    # Export to CSV if requested
    if args.export_csv:
        try:
            # Aggregated stats
            stats_df = aggregator.export_to_dataframe()
            stats_csv = output_dir / "aggregated_stats.csv"
            stats_df.to_csv(stats_csv, index=False)
            print(f"  Saved aggregated stats to {stats_csv}")
            
            # Individual results
            results_df = aggregator.export_results_to_dataframe()
            results_csv = output_dir / "individual_results.csv"
            results_df.to_csv(results_csv, index=False)
            print(f"  Saved individual results to {results_csv}")
        except ImportError:
            print("  Warning: pandas not installed, skipping CSV export")
    
    # Export detailed per-sample results if requested
    if args.export_details:
        details_dir = output_dir / "details"
        export_detailed_results(aggregator, reader, details_dir, steps_to_eval)
        print(f"  Saved detailed results to {details_dir}")
    
    # Generate plots
    if not args.no_plots:
        try:
            plotter = MetricsPlotter(aggregator)
            plots_dir = output_dir / "plots"
            saved_plots = plotter.save_all_plots(plots_dir, format=args.plot_format, method_name=args.method_name)
            print(f"  Saved {len(saved_plots)} plots to {plots_dir}")
        except ImportError:
            print("  Warning: matplotlib not installed, skipping plot generation")
        except Exception as e:
            print(f"  Error generating plots: {e}")
    
    # Print final summary
    print("=" * 60)
    print("Evaluation Complete!")
    print(f"  Steps evaluated: {len(stats['steps_evaluated'])}")
    print(f"  Total generations: {stats['total_generations']}")
    print(f"  Total evaluations: {stats['total_evaluations']}")
    print(f"  Cached evaluations: {stats['cached_evaluations']}")
    print(f"  API calls made: {stats['api_calls']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Output directory: {output_dir}")
    print(f"  Evaluation time: {eval_hours:.0f}h {eval_minutes:.0f}m {eval_seconds:.2f}s")
    print("=" * 60)
    
    return {
        "stats": stats,
        "aggregator": aggregator,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Retroactive Evaluation: Analyze model generations using API frontier models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with OpenAI API
  python -m retroactive_eval.run_eval \\
      --data-dir ./outputs/_dump_generations \\
      --output-dir ./eval_results \\
      --api-key $OPENAI_API_KEY

  # Use a custom API endpoint (e.g., vLLM, SGLang)
  python -m retroactive_eval.run_eval \\
      --data-dir ./outputs/_dump_generations \\
      --output-dir ./eval_results \\
      --api-base http://localhost:8000/v1 \\
      --model Qwen/Qwen2.5-72B-Instruct
        """,
    )
    
    # API configuration
    api_group = parser.add_argument_group("API Configuration")
    api_group.add_argument("--api-base", type=str, default=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"), help="OpenAI-compatible API base URL (default: $OPENAI_API_BASE or https://api.openai.com/v1)")
    api_group.add_argument("--api-key", type=str, default=os.environ.get("OPENAI_API_KEY"), help="API key (default: $OPENAI_API_KEY)")
    api_group.add_argument("--model", type=str, default="gpt-4o", help="Model name for API calls (default: gpt-4o)")
    
    # Concurrency configuration
    concurrency_group = parser.add_argument_group("Concurrency Configuration")
    concurrency_group.add_argument("--max-concurrent", type=int, default=512, help="Maximum concurrent API requests (default: 128)")
    concurrency_group.add_argument("--max-retries", type=int, default=3, help="Maximum retries for failed API calls (default: 3)")
    concurrency_group.add_argument("--timeout", type=float, default=120.0, help="Timeout for API requests in seconds (default: 120.0)")
    
    # Metrics configuration
    metrics_group = parser.add_argument_group("Metrics Configuration")
    metrics_group.add_argument("--metrics-config", type=str, default=None, help="Path to metrics YAML config (default: built-in config)")
    metrics_group.add_argument("--metrics", type=str, nargs="+", default=None, help="Specific metrics to evaluate (default: all enabled metrics)")
    
    # Data selection
    data_group = parser.add_argument_group("Data Selection")
    data_group.add_argument("--steps", type=int, nargs="+", default=None, help="Specific training steps to evaluate (default: all available steps)")
    data_group.add_argument("--max-samples-per-step", type=int, default=None, help="Maximum samples to evaluate per step (default: all)")
    
    # Cache configuration
    cache_group = parser.add_argument_group("Cache Configuration")
    cache_group.add_argument("--cache-dir", type=str, default=None, help="Directory for caching results (default: {output-dir}/cache)")
    cache_group.add_argument("--no-cache", action="store_true", help="Disable result caching")
    cache_group.add_argument("--clear-cache", "-cc", action="store_true", help="Clear existing cache before running")
    
    # Output configuration
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument("--plot-format", type=str, default="png", choices=["png", "pdf", "svg"], help="Format for saved plots (default: png)")
    output_group.add_argument(
        "--method-name",
        type=str,
        required=True,
        choices=ALGORITHM_NAMES,
        help="Algorithm title for the metrics-only deception plot",
    )
    output_group.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    output_group.add_argument(
        "--save-plot-statistics",
        action="store_true",
        help="Save or update step-wise statistics for later summary plotting",
    )
    output_group.add_argument("--export-csv", action="store_true", help="Export results to CSV files")
    output_group.add_argument("--export-details", action="store_true", help="Export detailed per-sample results to JSON files")
    
    # Logging
    parser.add_argument("--verbose", "-v", action="count", default=0, help="Increase verbosity (-v for INFO, -vv for DEBUG)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    elif args.verbose >= 2:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose >= 1:
        logging.getLogger().setLevel(logging.INFO)

    args.api_base = "http://localhost:6000/v1"
    args.model = "Qwen-Instruct-Large"

    # data_subdir = "outputs/verl_deceptive_roles/grpo_qwen3_8b_maximin_cot_judge/2026-07-08_00-23-21/rollout_data/rollout"  # Maximin RL
    data_subdir = "outputs/verl_deceptive_roles/grpo_qwen3_8b_maximin_cot_judge_w2s/2026-07-07_16-47-22/rollout_data/rollout"  # Maximin Rl with weak-to-strong oversight
    # data_subdir = "outputs/verl_deceptive_roles/grpo_qwen3_8b_monitor_cot_judge/2026-07-06_23-01-45/rollout_data/rollout"  # RL with critique monitor (Qwen3-8B) as oversight
    
    project_root = Path(__file__).parent.parent.resolve()
    data_dir = (project_root / data_subdir).resolve()
    args.data_dir = str(data_dir)  # Rollout directory containing {step}.jsonl files
    args.output_dir = str(data_dir.parent / "retro_eval")

    # steps to eval, None as default, meaning all available steps
    args.steps = [1] + [step for step in range(2, 161, 2)] # Evaluate every 2 steps from 2 to 160, plus step 1
    args.batch_size = args.max_concurrent * 10
    
    # Run async evaluation
    try:
        result = asyncio.run(run_evaluation(args))
        sys.exit(0 if result["stats"]["errors"] == 0 else 1)
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
