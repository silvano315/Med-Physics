import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class CombinedLoss(nn.Module):
  """Combined Dice and Cross Entropy loss for multi-class segmentation."""

  def __init__(self, num_classes: int = 4, dice_weight: float = 0.5):
      """
      Args:
          num_classes: Number of classes including background
          dice_weight: Weight for dice loss (1 - dice_weight for CE)
      """

      super().__init__()
      self.num_classes = num_classes
      self.dice_weight = dice_weight
      self.ce = nn.CrossEntropyLoss()

  def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Args:
        pred: (B, C, H, W) Predicted logits
        target: (B, H, W) Target labels

    Returns:
        Combined weighted loss
    """

    ce_loss = self.ce(pred, target)

    pred_softmax = F.softmax(pred, dim=1)
    dice_loss = 0

    for class_idx in range(self.num_classes):
      pred_class = pred_softmax[:, class_idx]
      target_class = (target == class_idx).float()

      intersection = (pred_class * target_class).sum()
      union = pred_class.sum() + target_class.sum()

      dice_score = (2.0 * intersection + 1e-5) / (union + 1e-5)
      dice_loss += (1 - dice_score)

    dice_loss = dice_loss / self.num_classes

    return self.dice_weight * ce_loss + (1 - self.dice_weight) * dice_loss
  
class SegmentationMetrics:
    """Evaluation metrics for segmentation."""

    @staticmethod
    def dice_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 4) -> Dict[str, float]:
        """
        Calculate per-class and mean Dice scores.

        Args:
            pred: (B, C, H, W) Predicted probabilities after softmax
            target: (B, H, W) Target labels
            num_classes: Number of classes including background

        Returns:
            Dictionary with per-class and mean Dice scores
        """

        dice_scores = {}

        for class_idx in range(num_classes):
          pred_class = (pred.argmax(dim=1) == class_idx).float()
          target_class = (target == class_idx).float()

          intersection = (pred_class * target_class).sum().item()
          union = pred_class.sum().item() + target_class.sum().item()

          if union > 0:
              dice = (2.0 * intersection) / union
          else:
              dice = 1.0 if intersection == 0 else 0.0

          dice_scores[f'dice_class_{class_idx}'] = dice

        dice_scores['dice_mean'] = sum(
            dice_scores[f'dice_class_{i}'] for i in range(num_classes)
        ) / num_classes

        return dice_scores
    
def compute_detailed_metrics(predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Compute detailed metrics for model evaluation.
    """
    metrics = {}

    # Per-class Dice Score
    dice_scores = SegmentationMetrics.dice_score(predictions, labels)
    metrics.update(dice_scores)

    # Mean IoU
    pred_masks = predictions.argmax(dim=1)
    for class_idx in range(4):
        pred_class = (pred_masks == class_idx)
        true_class = (labels == class_idx)

        intersection = (pred_class & true_class).sum().float()
        union = (pred_class | true_class).sum().float()

        iou = (intersection + 1e-6) / (union + 1e-6)
        metrics[f'iou_class_{class_idx}'] = iou.item()

    metrics['mean_iou'] = sum(metrics[f'iou_class_{i}'] for i in range(4)) / 4

    return metrics