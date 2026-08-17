# Hybrid U-Net and Active Contour for Brain Tumor Segmentation

## Project Overview

This project focuses on brain tumor segmentation from MRI scans using a hybrid approach that combines U-Net segmentation with Active Contour refinement.

The project uses the BraTS 2020 dataset and processes FLAIR MRI images to identify and refine tumor regions.

## Duration

4 months

## Team Size

3 members

## Domain

Artificial Intelligence / Medical Image Processing

## Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- NiBabel
- scikit-image
- U-Net
- Active Contour

## Project Workflow

1. Load BraTS 2020 MRI data.
2. Extract FLAIR MRI slices.
3. Preprocess the MRI images using normalization, CLAHE and Gaussian smoothing.
4. Train a U-Net model for tumor segmentation.
5. Apply thresholding and morphological operations to clean the predicted mask.
6. Remove small false-positive regions using connected-component analysis.
7. Apply Active Contour to refine the tumor boundaries.
8. Visualize the final tumor segmentation over the original MRI image.

## My Contribution

I contributed to the MRI image preprocessing, U-Net-based segmentation pipeline, post-processing of the predicted tumor mask, and Active Contour-based boundary refinement.

## Dataset

BraTS 2020 dataset.

The dataset is not included in this repository.

## Project Outcome

The project produced a tumor segmentation pipeline that combines deep-learning-based segmentation with Active Contour refinement to obtain a more refined tumor boundary.

## Note

This project is an academic project for medical image segmentation research and is not intended for clinical diagnosis or medical decision-making.
