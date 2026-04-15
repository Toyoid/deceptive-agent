import numpy as np

from agent_system.self_monitor import compute_self_monitor_metrics, compute_self_monitor_metrics_by_source


def test_compute_self_monitor_metrics_reports_step_level_ratios():
    metrics = compute_self_monitor_metrics(
        is_valid=np.array([True, True, True, False]),
        is_unsafe=np.array([False, True, False, False]),
        traj_uids=np.array(["traj-a", "traj-a", "traj-b", "traj-b"], dtype=object),
    )

    assert metrics["self_monitor/valid_step_ratio"] == 0.75
    assert metrics["self_monitor/unsafe_step_ratio"] == 0.25
    assert "self_monitor/unsafe_traj_ratio" not in metrics


def test_compute_self_monitor_metrics_by_source_uses_trust_penalties():
    metrics = compute_self_monitor_metrics_by_source(
        data_sources=np.array(["ds1", "ds1", "ds2", "ds2"], dtype=object),
        is_valid=np.array([True, True, False, True]),
        is_unsafe=np.array([False, False, False, True]),
        traj_uids=np.array(["traj-a", "traj-a", "traj-b", "traj-b"], dtype=object),
        trust_penalties=np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
        prefix="val",
    )

    assert metrics["val/ds1/self_monitor/valid_step_ratio"] == 1.0
    assert metrics["val/ds1/self_monitor/unsafe_step_ratio"] == 0.0

    assert metrics["val/ds2/self_monitor/valid_step_ratio"] == 0.5
    assert metrics["val/ds2/self_monitor/unsafe_step_ratio"] == 0.5
    assert "val/ds1/self_monitor/unsafe_traj_ratio" not in metrics
    assert "val/ds2/self_monitor/unsafe_traj_ratio" not in metrics
