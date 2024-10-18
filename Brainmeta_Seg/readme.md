
# Brain Metastasis Segmentation Project

## Project Description

This project focuses on the segmentation of brain metastasis using 3D CT scans. The model architecture is entirely based on `nnUNet v2`, specifically utilizing a 3D cascade UNet model.

The purpose of the project is to accurately segment brain metastases, assisting in clinical analysis and treatment planning.

## Model Architecture

The segmentation is performed using a 3D cascade UNet model built on `nnUNet v2`. The model has been trained and validated on a private dataset.

## Installation

The installation process has been designed to be automatic when executing the inference file. No manual setup is required; the necessary dependencies will be installed at runtime.

## Data

- **Training Data**: Due to privacy concerns, the dataset used for training is not publicly available.
- **Total Data**: 1138 3D CT images.
  - **Training Set**: 911 images.
  - **Validation Set**: 227 images.
  
  For the validation set, the model achieved a Dice score of:
  - **Mean Dice**: 0.7372
  - **Standard Deviation**: 0.2727
  
  _Note: The model was also evaluated on an external dataset, achieving a Dice score of 0.7587 (mean) and 0.2878 (standard deviation). This evaluation was based on a minimum brain metastasis volume of 100mm^3._

  External dataset information:
  
  **Paper**: Ocaña-Tienda, Beatriz, et al. "A comprehensive dataset of annotated brain metastasis MR images with clinical and radiomic data." Scientific data 10.1 (2023): 208.

  **Download Link**: [Brain Metastasis Dataset](https://molab.es/datasets-brain-metastasis-1/?type=metasrd)

## Model Weights

You can download the pre-trained model weights from the following link:  
[Download Model Weights](https://drive.google.com/file/d/1-pauKADV0gEfen2Jip9Jwjdttx6oDa_w/view?usp=drive_link)

Once downloaded, place the `nnUNet_results` directory at the same level as the `brainmeta_prediction.sh` file.

## Execution

To run the inference on new CT images, follow these steps:

1. Place the desired CT image files in the `_inputs` folder.
2. Ensure that the filenames for the images end with `_0000.nii.gz`. This is required for the inference process.
3. Run the following command to execute the segmentation:

   ```bash
   bash brainmeta_prediction.sh
   ```
