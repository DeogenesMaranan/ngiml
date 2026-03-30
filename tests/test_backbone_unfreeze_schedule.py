import torch.nn as nn

from tools.train_ngiml import _set_backbone_trainability_for_epoch


class _ToyEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.stem = nn.Conv2d(3, 4, kernel_size=1)
        self.backbone.blocks = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(4, 4, kernel_size=1)),
                nn.Sequential(nn.Conv2d(4, 4, kernel_size=1)),
                nn.Sequential(nn.Conv2d(4, 4, kernel_size=1)),
            ]
        )


class _ToySwin(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 4, kernel_size=1)
        self.stages = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(4, 4, kernel_size=1)),
                nn.Sequential(nn.Conv2d(4, 4, kernel_size=1)),
                nn.Sequential(nn.Conv2d(4, 4, kernel_size=1)),
            ]
        )


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.efficientnet = _ToyEfficientNet()
        self.swin = _ToySwin()


def _count_trainable_params(module: nn.Module) -> int:
    return sum(1 for p in module.parameters() if p.requires_grad)


def test_backbone_unfreeze_schedule_freezes_then_gradually_enables_groups():
    model = _ToyModel()

    phase = _set_backbone_trainability_for_epoch(
        model,
        epoch=0,
        freeze_backbone_epochs=3,
        progressive_unfreeze_epochs=3,
    )
    assert phase == "frozen"
    assert _count_trainable_params(model.efficientnet) == 0
    assert _count_trainable_params(model.swin) == 0

    phase = _set_backbone_trainability_for_epoch(
        model,
        epoch=3,
        freeze_backbone_epochs=3,
        progressive_unfreeze_epochs=3,
    )
    assert phase.startswith("progressive")
    eff_trainable_epoch3 = _count_trainable_params(model.efficientnet)
    swin_trainable_epoch3 = _count_trainable_params(model.swin)
    assert eff_trainable_epoch3 > 0
    assert swin_trainable_epoch3 > 0

    _set_backbone_trainability_for_epoch(
        model,
        epoch=4,
        freeze_backbone_epochs=3,
        progressive_unfreeze_epochs=3,
    )
    assert _count_trainable_params(model.efficientnet) > eff_trainable_epoch3
    assert _count_trainable_params(model.swin) > swin_trainable_epoch3

    phase = _set_backbone_trainability_for_epoch(
        model,
        epoch=5,
        freeze_backbone_epochs=3,
        progressive_unfreeze_epochs=3,
    )
    assert phase == "all-trainable"
    assert _count_trainable_params(model.efficientnet) == sum(
        1 for _ in model.efficientnet.parameters()
    )
    assert _count_trainable_params(model.swin) == sum(1 for _ in model.swin.parameters())
