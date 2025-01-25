import os
import torch
import urllib.request
import numpy as np
from segment_anything import sam_model_registry
from src_ACDC_ds.data.data_processing import prepare_image

def setup_sam():
    """Initialize and load SAM model."""

    sam_checkpoint = "sam_vit_b_01ec64.pth"
    checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    if not os.path.exists(sam_checkpoint):
        print("Downloading SAM checkpoint...")
        urllib.request.urlretrieve(checkpoint_url, sam_checkpoint)

    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=DEVICE)

    return sam


def segment_with_sam(image, points, point_labels, predictor):
    """
    Get segmentation for each class using SAM.

    Args:
        image: image to be segmented
        points: Array of points [N, 2]
        point_labels: Original label arrays (1,2,3)
        predictor: SAM predictor already initialized
    """

    image_rgb = prepare_image(image)
    predictor.set_image(image_rgb)

    masks = {}
    unique_classes = np.unique(point_labels)
    for class_id in unique_classes:
      class_points = points[point_labels == class_id]
      other_points = points[point_labels != class_id]

      input_points = np.vstack((class_points, other_points))
      input_labels = np.hstack((np.ones(len(class_points)), np.zeros(len(other_points))))

      masks_pred, scores, _ = predictor.predict(
          point_coords=input_points,
          point_labels=input_labels,
          multimask_output=True
      )

      best_mask = masks_pred[scores.argmax()]
      masks[class_id] = best_mask

    return masks