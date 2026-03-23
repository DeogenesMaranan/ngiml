import torch

from src.model.losses import MultiStageLossConfig, MultiStageManipulationLoss


def test_default_stage_weights_are_increasing():
    loss = MultiStageManipulationLoss(MultiStageLossConfig(stage_weights=None))
    assert loss._stage_weights(4) == [0.25, 0.5, 0.75, 1.0]


def test_explicit_stage_weights_unchanged():
    loss = MultiStageManipulationLoss(MultiStageLossConfig(stage_weights=[0.1, 0.2, 0.3, 0.4]))
    assert loss._stage_weights(3) == [0.1, 0.2, 0.3]


def test_boundary_loss_uses_stage_zero_prediction(monkeypatch):
    loss = MultiStageManipulationLoss(
        MultiStageLossConfig(
            stage_weights=[1.0, 1.0],
            use_boundary_loss=True,
            boundary_weight=1.0,
        )
    )

    captured = {"shape": None}

    def _fake_boundary(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        captured["shape"] = tuple(pred.shape)
        return pred.new_tensor(0.0)

    assert loss.boundary_loss is not None
    monkeypatch.setattr(loss.boundary_loss, "forward", _fake_boundary)

    stage0 = torch.zeros((2, 1, 32, 32), dtype=torch.float32)
    stage1 = torch.zeros((2, 1, 16, 16), dtype=torch.float32)
    target = torch.zeros((2, 1, 32, 32), dtype=torch.float32)

    _ = loss([stage0, stage1], target)

    assert captured["shape"] == tuple(stage0.shape)
