import SimpleITK as sitk
import numpy as np
import os
import matplotlib.pyplot as plt
import statistics
from tqdm import tqdm

def list_files_by_creation_time(directory):
    # 파일 리스트를 가져옵니다
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.nii') or f.endswith('.nii.gz')]

    # 파일을 생성 시간 순으로 정렬합니다
    files.sort(key=os.path.getctime)

    return files

def read_nifti_file(file_path):
    """Read a NIfTI file and handle potential RuntimeError."""
    try:
        image = sitk.ReadImage(file_path)
        return image
    except RuntimeError as e:
        print(f"Error reading file {file_path}: {e}")
        return None
import pydicom

def cos_similarity(v1, v2):
    """Calculate the cosine similarity between two vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def determine_plane(image_orientation, tol=0.01):
    """
    Determine the imaging plane from the Image Orientation (Patient) tag.
    
    Args:
    - image_orientation: list of float, the Image Orientation (Patient) tag values.
    - tol: float, tolerance level for cosine similarity comparison.
    
    Returns:
    - str, the determined imaging plane ("Axial", "Sagittal", "Coronal", or "Unknown").
    """
    x, y = np.array(image_orientation[:3], dtype=float), np.array(image_orientation[3:], dtype=float)
    
    axial_x = np.array([1, 0, 0], dtype=float)
    axial_y = np.array([0, 1, 0], dtype=float)
    
    sagittal_x = np.array([0, 1, 0], dtype=float)
    sagittal_y = np.array([0, 0, -1], dtype=float)
    
    coronal_x = np.array([1, 0, 0], dtype=float)
    coronal_y = np.array([0, 0, -1], dtype=float)
    
    if cos_similarity(x, axial_x) > 1 - tol and cos_similarity(y, axial_y) > 1 - tol:
        return "Axial"
    elif cos_similarity(x, sagittal_x) > 1 - tol and cos_similarity(y, sagittal_y) > 1 - tol:
        return "Sagittal"
    elif cos_similarity(x, coronal_x) > 1 - tol and cos_similarity(y, coronal_y) > 1 - tol:
        return "Coronal"
    else:
        return "Unknown"

def calculate_intensity_in_direction(mask_array, img_array, direction):
    """Calculate the intensity distribution in the specified direction."""
    sequence_intensities = []
    if direction == 'axial':
        for slice_index in range(mask_array.shape[0]):
            label_mask = mask_array[slice_index] == 4
            if np.any(label_mask):
                labeled_intensities = img_array[slice_index][label_mask]
                average_intensity = np.mean(labeled_intensities)
                sequence_intensities.append(average_intensity)
    elif direction == 'sagittal':
        for slice_index in range(mask_array.shape[2]):
            label_mask = mask_array[:, :, slice_index] == 4
            if np.any(label_mask):
                labeled_intensities = img_array[:, :, slice_index][label_mask]
                average_intensity = np.mean(labeled_intensities)
                sequence_intensities.append(average_intensity)
    elif direction == 'coronal':
        for slice_index in range(mask_array.shape[1]):
            label_mask = mask_array[:, slice_index, :] == 4
            if np.any(label_mask):
                labeled_intensities = img_array[:, slice_index, :][label_mask]
                average_intensity = np.mean(labeled_intensities)
                sequence_intensities.append(average_intensity)
    return sequence_intensities
def load_nii_with_simpleitk(file_path):
    """Load a NIfTI file using SimpleITK."""
    return sitk.ReadImage(file_path)

def display_slice_with_mask(img, mask, slice_index, label=4):
    """Display a slice of the image with the mask overlay for a specific label."""
    img_array = sitk.GetArrayFromImage(img)
    mask_array = sitk.GetArrayFromImage(mask)

    plt.figure(figsize=(10, 5))

    # Display the image slice
    plt.subplot(1, 2, 1)
    plt.imshow(img_array[slice_index, :, :], cmap='gray')
    plt.title("Image Slice")

    # Display the image slice with the mask overlay
    plt.subplot(1, 2, 2)
    masked = np.ma.masked_where(mask_array[slice_index, :, :] != label, mask_array[slice_index, :, :])
    plt.imshow(img_array[slice_index, :, :], cmap='gray')
    plt.imshow(masked, cmap='autumn', alpha=0.5)  # Mask overlay in 'autumn' colormap
    plt.title("Image with Mask Overlay")

    plt.show()
def calculate_intensity_distribution(img, mask, label):
    """Calculate the intensity distribution for the specified label in the mask across all slices."""
    mask_array = sitk.GetArrayFromImage(mask)
    img_array = sitk.GetArrayFromImage(img)
    label_mask = mask_array == label
    return img_array[label_mask]    