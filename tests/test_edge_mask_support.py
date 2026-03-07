import torch

from src.data.config import Manifest, SampleRecord
from src.model.losses import MultiStageLossConfig, MultiStageManipulationLoss


def test_manifest_roundtrip_preserves_edge_mask_path():
    manifest = Manifest(
        samples=[
            SampleRecord(
                dataset="CASIA2",
                split="train",
                image_path="prepared/CASIA2/train/fake/sample_01.npz",
                mask_path=None,
                label=1,
                edge_mask_path="prepared/CASIA2/train/fake/sample_01_edge.png",
            )
        ],
        normalization_mode="imagenet",
    )

    recovered = Manifest.from_dataframe(manifest.to_dataframe())

    assert recovered.samples[0].edge_mask_path == "prepared/CASIA2/train/fake/sample_01_edge.png"


def test_boundary_loss_uses_explicit_edge_target_when_present():
    loss_fn = MultiStageManipulationLoss(
        MultiStageLossConfig(
            dice_weight=0.0,
            bce_weight=0.0,
            use_boundary_loss=True,
            boundary_weight=1.0,
        )
    )
    preds = [torch.zeros((2, 1, 8, 8), dtype=torch.float32)]
    target = torch.zeros((2, 1, 8, 8), dtype=torch.float32)
    edge_target = torch.zeros((2, 1, 8, 8), dtype=torch.float32)
    edge_target[0, :, 3:5, 3:5] = 1.0
    edge_present = torch.tensor([True, False])

    explicit_loss = loss_fn(preds, target, edge_target=edge_target, edge_target_present=edge_present)
    fallback_loss = loss_fn(preds, target)

    assert explicit_loss.item() > fallback_loss.item()