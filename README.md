# IIITH Campus Image Prediction

This repository contains a collection of models designed to localize images taken at various locations across the IIIT-Hyderabad campus. The pipeline includes region classification, viewing angle prediction, and geographic coordinate estimation.

## 📌 Overview

The system comprises three major components:

1. **Region Classification**  
   Classifies an image into one of 15 predefined campus regions using a fine-tuned CNN.

2. **Viewing Angle Prediction**  
   Predicts the direction the camera was facing (angle from North) using region-specific regressors.

3. **Coordinate Prediction**  
   Estimates the (latitude, longitude) location of the image by classifying it to a representative anchor within the predicted region.

---

## 📂 Notebooks

### 1. Region Classification
**Notebook:** `region_classification.ipynb`  
Trains a **ConvNeXtV2 Tiny** model (`convnextv2_tiny.fcmae_ft_in1k`) for 15-way classification of images into campus regions.  
- **Training Strategy:** Fine-tuned on 6000+ campus images using `CrossEntropyLoss`.
- **Preprocessing:** Resize to 224×224, normalize with ImageNet stats, and apply augmentations (flips, crops).
- **Output:** Region ID (1–15)

### 2. Viewing Angle Prediction
**Notebook:** `angle_prediction.ipynb`  
Implements a **two-stage ensemble** for predicting the viewing direction in degrees:
- **Stage 1:** A frozen ConvNeXtV2 model classifies the image region.
- **Stage 2:** A region-specific ConvNeXtV2 model regresses the angle in `[cos, sin]` form using `MSELoss`.
- **Postprocessing:** Angle recovered using `atan2(sin, cos)`.

### 3. Coordinate Prediction (Anchor Classification)
**Notebook:** `coordinate_prediction.ipynb`  
Uses **fused features** and **anchor classification** for coordinate prediction:
- **Feature Extraction:** Extracts and fuses features from frozen ConvNeXtV2 (CNN) and ViT (Transformer).
- **Anchor Setup:** Applies K-Means clustering to region-wise coordinates to define `N` anchor points per region.
- **Classification:** Trains a region-specific MLP using `CrossEntropyLoss` to classify the fused feature into one of the region's anchor indices.

---

## 🧠 Key Ideas

- Trained **region-specific expert models** for angle and coordinate prediction, guided by a shared region classifier.
- Converted coordinate regression to a **classification task** via anchor clustering.
- Combined features from **CNNs and Transformers** to improve semantic understanding and localization accuracy.
