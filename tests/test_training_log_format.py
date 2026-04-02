from src.training_loop_helpers import format_status_flags


def test_format_status_flags_compacts_and_skips_empty_values():
    formatted = format_status_flags(["best-iou -> best.pt", "", "checkpoint ckpt.pt", "   "])
    assert formatted == "best-iou -> best.pt | checkpoint ckpt.pt"


def test_format_status_flags_returns_none_when_empty():
    assert format_status_flags([]) == "none"
