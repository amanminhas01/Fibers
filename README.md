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
File	                               Description 
model.py      	           Definition of the proposed ImprovedSegModel segmentation architecture
config.py	                 Model/class configuration
inference_image.py	       Segmentation inference for individual images
final intigrated app.py	   Integrated segmentation and osteoporosis classification application
requirements.txt	         Python package dependencies and tested package versions
outputsResnet34/           Trained segmentation model checkpoint
best_valid_epoch_dice.pth	
svm/svm_trabecular         Trained SVM classification model
_model.pkl
svm/scaler.pkl	           Feature standardisation/scaling model


# 3. Environment
he implementation was developed and tested using Python 3.11.

The required Python packages and their versions are provided in:

requirements.txt

It is recommended to create a clean virtual environment before installing the dependencies.

Create a virtual environment
python3.11 -m venv venv

Activate the environment on Linux:

source venv/bin/activate
Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. GPU Support
The segmentation model can use a CUDA-enabled GPU when available.

The implementation automatically selects the available device according to the code configuration.

To verify PyTorch CUDA availability:

import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

If CUDA is unavailable, the model can be executed using the CPU, although inference may be slower.

# 5. Segmentation Model
The segmentation network is implemented in:

model.py

The proposed ImprovedSegModel uses a ResNet-34-based encoder together with the additional architectural components implemented in the repository, including CoordConv, SoftPool, ASPP and decoder modules.

The trained segmentation checkpoint is provided at:

outputsResnet34/best_valid_epoch_dice.pth

The checkpoint is loaded during inference and used to generate the trabecular segmentation mask.

# 6. Segmentation Classes
The segmentation configuration is defined in config.py.

The model uses the following classes:

background
fiber1
fiber2
fiber3
fiber4

The predicted segmentation is obtained from the model output using the class with the highest predicted score at each pixel.

# 7. Image Preprocessing

For inference, input images are resized to:

512 × 512 pixels

The image is converted to the required input representation and normalised using the parameters implemented in the inference/application code.

The normalisation values used by the current implementation are:

Mean = [0.457, 0.433, 0.400]
Std  = [0.239, 0.235, 0.239]

The same preprocessing should be used when reproducing the reported inference results.

# 8. Running Segmentation Inference
The repository provides:

inference_image.py

for segmentation inference.

From the repository root, run:

python inference_image.py

The script loads the trained segmentation checkpoint and processes the input according to the preprocessing procedure implemented in the script.

Predicted segmentation results are saved according to the output configuration in the script.
# 9. Integrated Osteoporosis Classification
The complete workflow is implemented in:

final intigrated app.py

The integrated application performs:

Input radiograph
       ↓
Image preprocessing
       ↓
Trabecular segmentation
       ↓
Feature extraction
       ↓
Feature scaling
       ↓
SVM classification
       ↓
Normal / Osteoporotic prediction

The application uses the following trained files:

outputsResnet34/best_valid_epoch_dice.pth
svm/svm_trabecular_model.pkl
svm/scaler.pkl

The SVM model was trained using trabecular features extracted from the automated segmentation output.

# 10. Trabecular Feature Extraction
The integrated implementation extracts multiple handcrafted descriptors from the predicted trabecular regions.

The implemented feature extraction includes descriptors based on:

Intensity statistics
Intensity histograms
Gray-Level Co-occurrence Matrix (GLCM)
Local Binary Pattern (LBP)
Histogram of Oriented Gradients (HOG)
Canny edge information
Segmentation-mask characteristics
Fractal-dimension-related features
Trabecular graph/network features

The extracted feature vector is subsequently transformed using the provided scaler:

svm/scaler.pkl

and passed to:

svm/svm_trabecular_model.pkl

# 11. SVM Classification
The trained SVM classifier is provided as:

svm/svm_trabecular_model.pkl

The fitted feature scaler is provided as:

svm/scaler.pkl

Both files should be used together because the classifier expects the feature representation to be transformed using the same scaling procedure used during training.

The integrated application obtains the SVM probability for the osteoporotic class and applies the classification threshold implemented in the application.

The current implementation uses:

Probability > 0.30 → Osteoporotic
Probability ≤ 0.30 → Normal

The probability value displayed by the application represents the SVM-estimated probability for the osteoporotic class.

# 12. External Evaluation Dataset

The repository includes the external evaluation images used for the reported external validation experiment.

The external evaluation set contains:

Total images: 105

Normal:       57
Osteoporotic: 48

These images were collected approximately 15–16 months after the training dataset and were not used during model optimisation.

The class-wise images are organised in the repository under the corresponding class directories.

# 13. Reproducing the External Evaluation

To reproduce the external evaluation:

Clone the repository.
Create the Python environment.
Install the packages listed in requirements.txt.

Ensure that the trained segmentation checkpoint is available in:

outputsResnet34/best_valid_epoch_dice.pth

Ensure that the trained SVM and scaler are available in:

svm/svm_trabecular_model.pkl
svm/scaler.pkl
Run the integrated inference/application workflow.
Process the 105 external evaluation images.
Compare the predicted labels with the corresponding ground-truth class labels.

The reported external evaluation contains 57 normal and 48 osteoporotic cases.

# 14. Model Files

The repository provides the trained models required for inference:

Segmentation
outputsResnet34/best_valid_epoch_dice.pth
SVM classifier
svm/svm_trabecular_model.pkl
Feature scaler
svm/scaler.pkl

These files allow the trained pipeline to be used without retraining the models.

# 15. Reproducibility Notes

For reproducibility, the following should remain unchanged when reproducing the reported results:

Image preprocessing
Image resizing
Input normalisation
Segmentation checkpoint
Feature extraction procedure
Feature ordering
Feature scaling
SVM classifier
Classification threshold
External evaluation images
Ground-truth class assignments

The repository includes the trained segmentation model, SVM classifier, scaler, inference implementation and dependency specification required for the inference workflow.

# 16. Training and Data Availability

The repository is primarily provided to facilitate reproduction of the trained inference and evaluation workflow.

Patient-level clinical information and the original clinical dataset are subject to the applicable ethical and data-governance restrictions and are therefore not provided as part of this repository.

The publicly provided external evaluation images are included to facilitate independent assessment of the released inference pipeline.

# 17. Citation

If you use this implementation or the associated methodology in your research, please cite the corresponding publication:



The complete bibliographic information should be updated here once the associated article is formally published.

# 18. Disclaimer

This repository is provided for research and reproducibility purposes. The implementation is not intended to replace clinical diagnosis or professional medical assessment.

The predictions generated by the software should not be used as a standalone clinical decision-making tool.

# 19. Contact

For questions regarding the implementation or reproducibility of the research, please use the issue tracker associated with this repository.
