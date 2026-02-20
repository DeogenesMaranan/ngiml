from src.model.losses import MultiStageLossConfig, MultiStageManipulationLoss


def test_default_stage_weights_are_increasing():
    loss = MultiStageManipulationLoss(MultiStageLossConfig(stage_weights=None))
    assert loss._stage_weights(4) == [0.25, 0.5, 0.75, 1.0]


def test_explicit_stage_weights_unchanged():
    loss = MultiStageManipulationLoss(MultiStageLossConfig(stage_weights=[0.1, 0.2, 0.3, 0.4]))
    assert loss._stage_weights(3) == [0.1, 0.2, 0.3]
