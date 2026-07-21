"""Tests for portable, step-wise plotting statistics."""

import json

import numpy as np
import pytest

pytest.importorskip("matplotlib")
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from retroactive_eval.analysis.aggregator import MetricsAggregator
from retroactive_eval.analysis.plot_statistics import PlotStatistics
from retroactive_eval.analysis.plotter import MetricsPlotter
from retroactive_eval.data_reader.schemas import Generation
from retroactive_eval.metrics.base_metric import MetricResult
from retroactive_eval.plot_saved_statistics import main as plot_saved_main


SOURCE = {
    "evaluator_model": "test-model",
    "metrics_config_sha256": "config-hash",
    "std_ddof": 0,
}


def _build_aggregator(steps=(1, 2), score_offset=0.0):
    aggregator = MetricsAggregator()
    metrics = ("omission", "misdirection")
    for step in steps:
        generations = [
            Generation(
                input=f"input-{step}-{index}",
                output=f"output-{step}-{index}",
                score=0.2 * index + 0.01 * step,
                step=step,
            )
            for index in range(3)
        ]
        for metric_index, metric_name in enumerate(metrics):
            for sample_index, generation in enumerate(generations):
                aggregator.add_result(
                    MetricResult(
                        generation_uid=generation.uid,
                        metric_name=metric_name,
                        score=(
                            score_offset
                            + 0.1 * step
                            + 0.05 * metric_index
                            + 0.01 * sample_index
                        ),
                        token_probs={},
                        step=step,
                        cached=sample_index < 2,
                    ),
                    generation,
                )
    return aggregator


def _line_data(figure):
    return [
        [
            (line.get_label(), line.get_xdata().tolist(), line.get_ydata().tolist())
            for line in axis.lines
        ]
        for axis in figure.axes
    ]


def test_plot_statistics_round_trip_is_complete_and_standard_json(tmp_path):
    aggregator = _build_aggregator()
    statistics = PlotStatistics.from_aggregator(
        aggregator,
        method_name="Critique Monitor",
        source=SOURCE,
    )

    path = statistics.save_json(tmp_path / "plot_statistics.json")
    loaded = PlotStatistics.load_json(path)

    assert loaded.to_dict() == statistics.to_dict()
    assert loaded.get_metrics() == aggregator.get_metrics()
    assert len([record for record in loaded.records if record.series_type == "reward"]) == 2
    metric_record = next(
        record
        for record in loaded.records
        if record.series_type == "metric"
        and record.name == "omission"
        and record.step == 1
    )
    assert metric_record.cached_count == 2
    assert metric_record.api_count == 1
    assert metric_record.count == 3
    assert "NaN" not in path.read_text(encoding="utf-8")


def test_empty_scores_are_serialized_as_json_null(tmp_path):
    aggregator = MetricsAggregator()
    generation = Generation(input="i", output="o", score=0.5, step=1)
    aggregator.add_result(
        MetricResult.error_result(
            generation_uid=generation.uid,
            metric_name="omission",
            step=1,
            error="failed",
        ),
        generation,
    )
    statistics = PlotStatistics.from_aggregator(
        aggregator, method_name="RLHF", source=SOURCE
    )

    path = statistics.save_json(tmp_path / "plot_statistics.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    metric_record = next(
        record for record in data["records"] if record["series_type"] == "metric"
    )

    assert metric_record["mean"] is None
    assert np.isnan(PlotStatistics.load_json(path).get_time_series("omission").means[0])


def test_incremental_merge_adds_and_replaces_steps():
    first = PlotStatistics.from_aggregator(
        _build_aggregator(steps=(1,)),
        method_name="RLHF",
        source=SOURCE,
    )
    second = PlotStatistics.from_aggregator(
        _build_aggregator(steps=(2,)),
        method_name="RLHF",
        source=SOURCE,
    )
    merged = first.merged_with(second)

    assert merged.get_steps() == [1, 2]

    replacement = PlotStatistics.from_aggregator(
        _build_aggregator(steps=(1,), score_offset=0.5),
        method_name="RLHF",
        source=SOURCE,
    )
    replaced = merged.merged_with(replacement)

    assert replaced.get_steps() == [1, 2]
    assert replaced.get_time_series("omission").means[0] == pytest.approx(0.61)


def test_merge_rejects_incompatible_provenance():
    first = PlotStatistics.from_aggregator(
        _build_aggregator(steps=(1,)),
        method_name="RLHF",
        source=SOURCE,
    )
    incompatible_source = dict(SOURCE, metrics_config_sha256="different")
    second = PlotStatistics.from_aggregator(
        _build_aggregator(steps=(2,)),
        method_name="RLHF",
        source=incompatible_source,
    )

    with pytest.raises(ValueError, match="different metrics_config_sha256"):
        first.merged_with(second)


def test_merge_rejects_different_generations_at_same_step():
    first = PlotStatistics.from_aggregator(
        _build_aggregator(steps=(1,)),
        method_name="RLHF",
        source=SOURCE,
    )
    different = _build_aggregator(steps=(1,))
    for pairs in different._results.values():
        for result, generation in pairs[1]:
            generation.uid = f"different-{result.generation_uid}"
    second = PlotStatistics.from_aggregator(
        different,
        method_name="RLHF",
        source=SOURCE,
    )

    with pytest.raises(ValueError, match="generation fingerprints differ"):
        first.merged_with(second)


def test_live_and_loaded_statistics_produce_equivalent_summary_plots(tmp_path):
    aggregator = _build_aggregator()
    statistics = PlotStatistics.from_aggregator(
        aggregator,
        method_name="Critique Monitor",
        source=SOURCE,
    )
    loaded = PlotStatistics.load_json(
        statistics.save_json(tmp_path / "plot_statistics.json")
    )
    live_plotter = MetricsPlotter(aggregator)
    saved_plotter = MetricsPlotter(loaded)

    live_all = live_plotter.plot_all_metrics()
    saved_all = saved_plotter.plot_all_metrics()
    live_deception = live_plotter.plot_deception_metrics("Critique Monitor")
    saved_deception = saved_plotter.plot_deception_metrics("Critique Monitor")
    live_evolution = live_plotter.plot_evolution_comparison()
    saved_evolution = saved_plotter.plot_evolution_comparison()

    assert _line_data(saved_all) == _line_data(live_all)
    assert _line_data(saved_deception) == _line_data(live_deception)
    assert [patch.get_height() for patch in saved_evolution.axes[0].patches] == [
        patch.get_height() for patch in live_evolution.axes[0].patches
    ]

    for figure in (
        live_all,
        saved_all,
        live_deception,
        saved_deception,
        live_evolution,
        saved_evolution,
    ):
        plt.close(figure)


def test_saved_statistics_cli_writes_only_three_summary_plots(tmp_path):
    statistics = PlotStatistics.from_aggregator(
        _build_aggregator(),
        method_name="Critique Monitor",
        source=SOURCE,
    )
    statistics_path = statistics.save_json(tmp_path / "plot_statistics.json")
    output_dir = tmp_path / "plots"

    saved_files = plot_saved_main(
        [
            "--statistics",
            str(statistics_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert {path.name for path in saved_files} == {
        "all_metrics.png",
        "deception_metrics.png",
        "evolution_comparison.png",
    }


def test_run_eval_accepts_save_plot_statistics_flag(monkeypatch):
    from retroactive_eval.run_eval import parse_args

    monkeypatch.setattr(
        "sys.argv",
        ["run_eval.py", "--method-name", "RLHF", "--save-plot-statistics"],
    )

    assert parse_args().save_plot_statistics is True
