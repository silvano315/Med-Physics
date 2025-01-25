import torch
import torch.nn as nn
import torch.functional as F
from pathlib import Path
from transformers import SegformerForSemanticSegmentation
from transformers import TrainingArguments, Trainer, TrainerCallback
from src.src_ACDC_ds.evaluation.metrics_and_loss import CombinedLoss

class CardiacSegmenter(nn.Module):
    """Cardiac segmentation model based on SegFormer-B0."""

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        """
        Args:
            num_classes: Number of segmentation classes
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze the backbone during training
        """
        super().__init__()

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            num_channels=1
        )

        if freeze_backbone:
            for param in self.model.segformer.parameters():
              param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      """
      Forward pass

      Args:
          x: Input tensor of shape (B, 1, H, W)

      Returns:
          Logits of shape (B, num_classes, H, W)
      """

      outputs = self.model(x)
      logits = F.interpolate(
          outputs.logits,
          size=x.shape[-2:],
          mode="bilinear",
          align_corners=False
      )
      return logits