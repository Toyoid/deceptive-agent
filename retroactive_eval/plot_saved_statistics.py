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
"""Generate summary figures without loading individual cached results."""

import argparse
from pathlib import Path
from typing import List, Optional


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    from .run_eval import ALGORITHM_NAMES

    parser = argparse.ArgumentParser(
        description="Plot retroactive-evaluation summary statistics",
    )
    parser.add_argument(
        "--statistics",
        type=Path,
        required=True,
        help="Path to plot_statistics.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Plot output directory (default: <statistics-dir>/plots)",
    )
    parser.add_argument(
        "--plot-format",
        choices=["png", "pdf", "svg"],
        default="png",
    )
    parser.add_argument(
        "--method-name",
        choices=ALGORITHM_NAMES,
        default=None,
        help="Optional title override; defaults to the saved method name",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> List[Path]:
    import matplotlib

    # This command only writes image files and must work on headless machines.
    matplotlib.use("Agg", force=True)

    from .analysis.plot_statistics import PlotStatistics
    from .analysis.plotter import MetricsPlotter

    args = parse_args(argv)
    statistics_path = args.statistics.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else statistics_path.parent / "plots"
    )

    statistics = PlotStatistics.load_json(statistics_path)
    method_name = args.method_name or statistics.method_name
    saved_files = MetricsPlotter(statistics).save_summary_plots(
        output_dir,
        format=args.plot_format,
        method_name=method_name,
    )
    print(f"Saved {len(saved_files)} summary plots to {output_dir}")
    return saved_files


if __name__ == "__main__":
    main()
