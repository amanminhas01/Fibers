# Diagnosis of Osteoporosis by Analysing Trabecular Microstructure of Hip Radiographs
This repository provides the implementation and supporting files for the proposed framework for osteoporosis screening from proximal femur/hip radiographs using automated trabecular structure segmentation, handcrafted trabecular microstructural feature extraction, and SVM-based classification.

The repository is intended to support reproducibility of the inference and external evaluation workflow described in the associated research work.

# 1. Overview

The proposed framework consists of the following major stages:

1. Input hip/proximal femur radiograph
2. Image preprocessing and resizing
3. Automated segmentation of trabecular structures
4. Extraction of trabecular microstructural features from the predicted segmentation
5. Feature scaling using the fitted training scaler
6. SVM-based osteoporosis classification
7. Prediction of normal or osteoporotic class

The classification stage uses features extracted from the predicted segmentation masks, rather than manually annotated reference masks. This reflects the intended automated end-to-end workflow.

# 2. Repository Structure
Fibers/ 
│ 
├── README.md 
├── requirements.txt 
├── config.py 
├── model.py 
├── inference_image.py 
├── final intigrated app.py 
│ 
├── outputsResnet34/ 
│ 
├── best_valid_epoch_dice.pth 
│ └── ... 
│ 
├── svm/ 
│ 
├── svm_trabecular_model.pkl 
│ 
├── scaler.pkl 
│ 
├── feature_ranking.csv 
│ 
├── confusion_matrix.png 
│ └── roc_curve.png 
│ 
├── class_0/ 
│└── ... 
│ └── class_1/ 
└── ...

# Main files
File	Description
model.py	Definition of the proposed ImprovedSegModel segmentation architecture
config.py	Model/class configuration
inference_image.py	Segmentation inference for individual images
final intigrated app.py	Integrated segmentation and osteoporosis classification application
requirements.txt	Python package dependencies and tested package versions
outputsResnet34/best_valid_epoch_dice.pth	Trained segmentation model checkpoint
svm/svm_trabecular_model.pkl	Trained SVM classification model
svm/scaler.pkl	Feature standardisation/scaling model
