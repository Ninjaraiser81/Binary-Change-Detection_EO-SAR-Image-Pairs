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

import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Visualize Training/Validation Loss and Accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss', marker='o')
plt.plot(history['val_loss'], label='Val Loss', marker='x')
plt.title('Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history['val_accuracy'], label='Val Accuracy', color='orange', marker='s')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Visualize Sample Images with Binary Labels
plt.figure(figsize=(15, 4))
for i in range(5):
    image, label = train_dataset[i]
    plt.subplot(1, 5, i + 1)
    plt.imshow(image.squeeze().numpy(), cmap='gray')
    plt.title(f"Label: {label.item()}\n({'No-Change' if label.item() == 0 else 'Change'})")
    plt.axis('off')
plt.suptitle("Sample Training Images (Remapped Labels)", fontsize=14)
plt.tight_layout()
plt.show()

# 3. Plot Confusion Matrix
model.eval()
all_preds, all_labels_val = [], []
with torch.no_grad():
    for imgs, lbls in val_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels_val.extend(lbls.cpu().numpy())

cm = confusion_matrix(all_labels_val, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No-Change', 'Change'])
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap='Blues', values_format='d', ax=ax)
plt.title('Confusion Matrix - Validation Split')
plt.show()
