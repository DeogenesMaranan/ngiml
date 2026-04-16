from src.training_loop_helpers import _resolve_overlap_source, _resolve_validation_source


def test_resolve_validation_source_best_uses_lower_loss():
    raw = {"loss": 1.20, "f1": 0.40, "iou": 0.30}
    ema = {"loss": 1.35, "f1": 0.45, "iou": 0.33}
    source, metrics = _resolve_validation_source(
        raw_metrics=raw,
        ema_metrics=ema,
        policy="best",
        metric_key="loss",
    )
    assert source == "raw"
    assert metrics is raw


def test_resolve_validation_source_best_uses_higher_metric_for_non_loss():
    raw = {"loss": 1.20, "f1": 0.40, "iou": 0.30}
    ema = {"loss": 1.35, "f1": 0.45, "iou": 0.33}
    source, metrics = _resolve_validation_source(
        raw_metrics=raw,
        ema_metrics=ema,
        policy="best",
        metric_key="f1",
    )
    assert source == "ema"
    assert metrics is ema


def test_resolve_overlap_source_best_uses_higher_overlap_score():
    raw = {"f1": 0.56, "iou": 0.40}
    ema = {"f1": 0.51, "iou": 0.35}
    source, metrics = _resolve_overlap_source(
        raw_metrics=raw,
        ema_metrics=ema,
        policy="best",
    )
    assert source == "raw"
    assert metrics is raw
