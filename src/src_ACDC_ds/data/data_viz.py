import numpy as np
from ipywidgets import interact
from matplotlib import pyplot as plt
from typing import Optional

def plot_slice_with_mask(image, label, scribble=None, title="Cardiac MRI Slice"):
    """
    Plot a cardiac MRI slice with optional overlay of the mask.

    Args:
        image: 2D numpy array of the image
        label: 2D numpy array of the label image
        scribble: Optional 2D numpy array of the scribble mask
        title: Plot title
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(image, cmap='gray')
    if scribble is not None:
        plt.imshow(scribble, alpha=0.3, cmap='jet')
        plt.title("With Scribble")
    else:
        plt.title("No Scribble")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(image, cmap='gray')
    plt.imshow(label, alpha=0.3, cmap='jet')
    plt.title("With Label")
    plt.axis('off')

    plt.suptitle(f"{title}", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_volume_slices(
    volume: np.ndarray,
    label: np.ndarray,
    scribble: Optional[np.ndarray] = None,
    start_slice: Optional[int] = None,
    num_slices: int = 3,
    step: Optional[int] = None,
    title: str = "Cardiac MRI Volumes"
) -> None:
    """
    Plot multiple slices from a 3D volume, optionally with overlayed labels and scribbles.

    Args:
      - volume (np.ndarray): The 3D volume to display, with shape (depth, height, width).
      - label (np.ndarray): The 3D label mask to overlay, with shape (depth, height, width).
      - scribble (Optional[np.ndarray]): An optional 3D scribble mask to overlay, with shape (depth, height, width).
      - start_slice (Optional[int]): The starting slice index. If None, the function centers the slices.
      - num_slices (int): The number of slices to display.
      - step (Optional[int]): Step size between slices. Default is 1.
      - title (str): The title for the entire figure.
    """
    if start_slice is None:
        middle = volume.shape[0] // 2
        start_slice = middle - (num_slices // 2)

    if step is None:
        step = 1

    rows = 3 if scribble is not None else 2

    fig, axes = plt.subplots(rows, num_slices, figsize=(5 * num_slices, 5 * rows))
    for i in range(num_slices):
        slice_idx = start_slice + (i * step)
        axes[0,i].imshow(volume[slice_idx], cmap='gray')
        axes[0,i].set_title(f'Slice {slice_idx}', fontsize=20)
        axes[0,i].axis('off')

        axes[1,i].imshow(volume[slice_idx], cmap='gray')
        axes[1,i].imshow(label[slice_idx], alpha = 0.3, cmap='jet')
        axes[1,i].set_title('With Label', fontsize=20)
        axes[1,i].axis('off')

        if scribble is not None:
          axes[2,i].imshow(volume[slice_idx], cmap='gray')
          axes[2,i].imshow(scribble[slice_idx], alpha = 0.3, cmap='jet')
          axes[2,i].set_title('With Scribble', fontsize=20)
          axes[2,i].axis('off')

    plt.suptitle(f'{title}', fontsize=28)
    plt.tight_layout()
    plt.show()


def volume_slider(volume: np.ndarray, label: np.ndarray):
    """
    Create an interactive slider to navigate the volume with the label side by side.

    Args:
      - volume (np.ndarray): The 3D volume to be displayed (depth, height, width).
      - label (np.ndarray): The 3D label mask corresponding to the volume.
    """
    def view_slice(slice_idx: int):
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(volume[slice_idx], cmap='gray')
        plt.title(f'Volume - Slice {slice_idx}')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(volume[slice_idx], cmap='gray')
        plt.imshow(label[slice_idx], alpha=0.3, cmap='jet')
        plt.title(f'Volume con Label - Slice {slice_idx}')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    interact(view_slice, slice_idx=(0, volume.shape[0] - 1))


def visualize_points(image, points, point_labels):
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')

    classes = np.unique(point_labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))

    for class_id, color in zip(classes, colors):
        mask = point_labels == class_id
        plt.plot(points[mask, 0], points[mask, 1], '.',
                color=color, label=f'Class {class_id}', markersize=10)

    plt.legend()
    plt.axis('off')
    plt.title('Points per Class')
    plt.show()


def visualize_results(image, label, masks):
    fig, ax = plt.subplots(2, len(masks) + 1, figsize=(5*(len(masks) + 1), 5))

    ax[0,0].imshow(image, cmap='gray')
    ax[0,0].set_title('Original')
    ax[0,0].axis('off')

    colors = plt.cm.rainbow(np.linspace(0, 1, len(masks)))
    for i, (class_id, mask) in enumerate(masks.items(), 1):
        ax[0,i].imshow(image, cmap='gray')
        ax[0,i].imshow(mask, alpha=0.4, cmap='jet')
        ax[0,i].set_title(f'Class {class_id}')
        ax[0,i].axis('off')

    ax[1,0].imshow(image, cmap='gray')
    ax[1,0].imshow(label, alpha=0.3, cmap='jet')
    ax[1,0].set_title('True Label')
    ax[1,0].axis('off')

    fig.delaxes(ax[1, 1])
    fig.delaxes(ax[1, 2])
    fig.delaxes(ax[1, 3])

    plt.tight_layout()
    plt.show()