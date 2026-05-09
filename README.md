# Binary-Change-Detection_EO-SAR-Image-Pairs


# Project README

## Project Title & Description
**Binary Change Detection on EO/SAR Imagery**
This project implements a deep learning pipeline to detect changes between satellite imagery pairs. We used a Siamese-style CNN architecture to classify image pairs as either 'No-Change' or 'Change'.

## Requirements
- Python 3.10+
- torch==2.1.0
- torchvision==0.16.0
- datasets==2.15.0
- scikit-learn==1.2.2
- matplotlib
- tqdm
- numpy

## Environment Setup
```bash
# Install python dependencies
pip install torch torchvision datasets scikit-learn matplotlib tqdm
```

## Dataset Structure
The dataset is loaded directly from Hugging Face: `doron333/change-detection-dataset`. It consists of TIFF images and labels. Labels 0, 1 are treated as 'No-Change' (0), and 2, 3 as 'Change' (1).

## Training & Evaluation
The model is trained using a weighted CrossEntropyLoss to handle class imbalance and an Adam optimizer.

## Results
| Split | IoU | Precision | Recall | F1-Score |
|-------|-----|-----------|--------|----------|
| Validation | 1.00 | 1.00 | 1.00 | 1.00 |
| Test | 1.00 | 1.00 | 1.00 | 1.00 |

*Note: The perfect scores suggest a potential data leak in the dataset splits or an extremely high correlation between features and labels in this specific dataset sample.*
