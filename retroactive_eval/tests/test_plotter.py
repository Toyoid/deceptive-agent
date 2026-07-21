"""Tests for retroactive-evaluation plots."""

import pytest

pytest.importorskip("matplotlib")

from retroactive_eval.analysis.aggregator import MetricsAggregator
from retroactive_eval.analysis.plotter import ALGORITHM_NAMES, MetricsPlotter
from retroactive_eval.data_reader.schemas import Generation
from retroactive_eval.metrics.base_metric import MetricResult


def _aggregator_with_metrics() -> MetricsAggregator:
    aggregator = MetricsAggregator()
    for metric_index, metric_name in enumerate(("omission", "misdirection")):
        for step in (1, 2, 3):
            generation = Generation(
                input="test",
                output="test",
                score=10.0 + step,
                step=step,
                uid=f"{metric_name}-{step}",
            )
            result = MetricResult(
                generation_uid=generation.uid,
                metric_name=metric_name,
                score=0.1 * step + 0.05 * metric_index,
                token_probs={},
                step=step,
            )
            aggregator.add_result(result, generation)
    return aggregator


@pytest.mark.parametrize("method_name", ALGORITHM_NAMES)
def test_deception_metrics_uses_method_title_and_bottom_legend(method_name):
    plotter = MetricsPlotter(_aggregator_with_metrics())

    figure = plotter.plot_deception_metrics(method_name, show_std=False)

    assert figure._suptitle.get_text() == method_name
    assert len(figure.axes) == 1
    assert [line.get_label() for line in figure.axes[0].lines] == [
        "omission",
        "misdirection",
    ]
    assert [text.get_text() for text in figure.legends[0].get_texts()] == [
        "Omission",
        "Misdirection",
    ]
    assert figure.axes[0].get_legend() is None
    assert figure.get_size_inches().tolist() == pytest.approx([7.2, 4.6])
    assert figure.axes[0].get_ylim() == pytest.approx((0.0, 1.0))
    assert not figure.axes[0].spines["top"].get_visible()
    assert not figure.axes[0].spines["right"].get_visible()


def test_deception_metrics_rejects_unknown_method_name():
    plotter = MetricsPlotter(_aggregator_with_metrics())

    with pytest.raises(ValueError, match="Unknown method name"):
        plotter.plot_deception_metrics("Unknown Method")


def test_method_name_cli_argument_is_required(monkeypatch):
    from retroactive_eval.run_eval import parse_args

    monkeypatch.setattr("sys.argv", ["run_eval.py"])

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 2
