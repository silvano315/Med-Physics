import pandas as pd
import re
import h5py
import numpy as np
from collections import defaultdict
from scipy.ndimage import binary_dilation
from pathlib import Path
from typing import List, Dict

def list_h5_files(data_dir: Path, subset: str = 'training'):
    """
    List all H5 files in the specified directory.

    Args:
        data_dir: Base directory containing the dataset
        subset: 'training' or 'testing'

    Returns:
        List of paths to H5 files
    """
    if 'training' in subset:
        pattern = f"**/*_{subset}/*.h5"
    else:
        pattern = f"**/*_{subset}_*/*.h5"

    return list(data_dir.glob(pattern))


def create_dataset_info(file_lists: Dict[str, List[Path]]) -> pd.DataFrame:
    """
    Create a DataFrame with information about the dataset files.

    Args:
        file_lists: Dictionary with keys 'training_volumes', 'training_slices', 'testing_volumes'
                   and corresponding lists of Path objects

    Returns:
        DataFrame with columns: patient_id, frame, slice (if applicable), type, path
    """
    all_data = []

    pattern = r'patient(\d+)_frame(\d+)(?:_slice_(\d+))?'

    for data_type, files in file_lists.items():
        for file_path in files:
            match = re.search(pattern, file_path.name)
            if match:
                patient_id = match.group(1)
                frame = match.group(2)
                slice_num = match.group(3)

                data_entry = {
                    'patient_id': int(patient_id),
                    'frame': int(frame),
                    'slice': int(slice_num) if slice_num else None,
                    'type': data_type,
                    'path': str(file_path)
                }
                all_data.append(data_entry)

    df = pd.DataFrame(all_data)
    df = df.sort_values(['patient_id', 'frame', 'slice'])

    return df


def load_h5_data(file_path: str):
    """
    Load data from H5 file.

    Args:
        file_path: Path to H5 file

    Returns:
        Dictionary containing image and mask data
    """
    with h5py.File(file_path, 'r') as f:
        print(f"Available keys in {Path(file_path).name}:", list(f.keys()))

        data = {}
        if 'image' in f:
            data['image'] = f['image'][:]
        if 'label' in f:
            data['label'] = f['label'][:]
        if 'scribble' in f:
            data['scribble'] = f['scribble'][:]

        return data
    
def prepare_image(image):
  """Prepare medical image for SAM."""
  # Normalize to [0, 255]
  image_norm = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

  # Convert to RGB by repeating the channel
  image_rgb = np.stack([image_norm] * 3, axis=-1)
  return image_rgb

def get_points_from_scribble_only(scribble, n_points_per_class=10, dilation_size=3):
    """
    It extracts training points using scribbles only.

    Args:
        scribble: numpy array with values 0-4 (0 and 4 for white background and outer boundary)
        n_points_per_class: number of points per class
        dilation_size: size of the dilation to find nearby points
    """

    points = []
    point_labels = []

    for class_id in range(1, 4):
        class_points = scribble == class_id
        if not np.any(class_points):
            continue

        dilated = binary_dilation(class_points, iterations=dilation_size)

        valid_points = dilated & (scribble == 4)
        y_coords, x_coords = np.where(valid_points)

        if len(x_coords) > 0:
            n_points = min(n_points_per_class, len(x_coords))
            idx = np.random.choice(len(x_coords), n_points, replace=False)
            points.append(np.column_stack([x_coords[idx], y_coords[idx]]))
            point_labels.extend([class_id] * n_points)

    if points:
        all_points = np.vstack(points)
        point_labels = np.array(point_labels)
    else:
        all_points = np.array([])
        point_labels = np.array([])

    return all_points, point_labels

def analyze_slice_statistics(data_dir: Path) -> None:
    """
    Analyze statistics of individual slice files.

    Args:
        data_dir: Directory containing the h5 slice files
    """
    image_shapes = set()
    label_distributions = defaultdict(int)
    total_slices = 0

    for h5_file in data_dir.glob("patient*_frame*_slice_*.h5"):
        try:
            with h5py.File(h5_file, 'r') as f:
                total_slices += 1

                img_shape = f['image'][()].shape
                image_shapes.add(img_shape)

                unique_labels = tuple(sorted(np.unique(f['label'][()])))
                label_distributions[unique_labels] += 1

                if total_slices % 100 == 0:
                    print(f"Processed {total_slices} slices...")

        except Exception as e:
            print(f"Error processing {h5_file.name}: {str(e)}")

    print("\nDataset Statistics:")
    print("-" * 50)
    print(f"Total number of slices analyzed: {total_slices}")

    print("\nUnique image shapes found:")
    for shape in sorted(image_shapes):
        print(f"- {shape}")

    print("\nLabel distributions found:")
    for labels, count in sorted(label_distributions.items()):
        print(f"- Classes {labels}: {count} slices ({count/total_slices*100:.2f}%)")