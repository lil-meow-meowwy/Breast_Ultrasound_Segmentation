# Multiclass Segmentation of Breast Ultrasound Images Using Deep Learning

## Overview
This project aims to develop a deep learning solution for automating the segmentation of breast ultrasound images, focusing on identifying three key tissue types: normal, benign, and malignant. Using the BUSI dataset, a modified U-Net architecture was employed to achieve high segmentation accuracy and support early diagnosis of breast cancer.

---

## Project Highlights
- **Objective:** Automate the segmentation of breast ultrasound images to enhance diagnostic precision and efficiency.
- **Dataset:** BUSI (Breast Ultrasound Images Dataset), comprising 780 labeled images across three classes: normal, benign, and malignant.
- **Model Architecture:** Customized U-Net with residual blocks for improved feature extraction and segmentation.
- **Key Results:** Achieved a validation accuracy of **94.76%** and an Intersection over Union (IoU) of **90.97%**.

---

## Skills Demonstrated
- **Deep Learning:** Model design and training for image segmentation.
- **Data Preprocessing:** Image resizing, normalization, and mask encoding.
- **Evaluation Metrics:** Proficient use of metrics like accuracy, F1-score, recall, precision, and IoU.
- **Python Libraries:** PyTorch, TensorFlow, NumPy, Matplotlib, and Seaborn.

---

## Methodology
1. **Data Preprocessing:**
   - Images resized to 256x256 pixels.
   - Normalized pixel values to [0, 1] range.
   - Encoded masks for multiclass segmentation.
2. **Model Development:**
   - Adapted U-Net architecture with residual blocks.
   - Used softmax activation for multiclass predictions.
   - Optimized with categorical cross-entropy loss.
3. **Training and Evaluation:**
   - Dataset split: 80% training, 15% validation, 5% testing.
   - Metrics evaluated per class and averaged across epochs.
4. **Visualization:**
   - Plotted training/validation loss and accuracy over epochs.
   - Compared ground truth masks with model predictions.

---

## Results
### Metrics
| Metric       | Validation Value |
|--------------|------------------|
| Accuracy     | 94.76%           |
| F1-Score     | 94.17%           |
| Precision    | 94.69%           |
| Recall       | 94.76%           |
| IoU          | 90.97%           |

### Sample Visualizations
| Original Image | Ground Truth Mask | Model Prediction |
|----------------|-------------------|------------------|

| ![image](https://github.com/user-attachments/assets/e54b0228-919c-4a14-95f6-0f7692485d4d)
   | ![image](https://github.com/user-attachments/assets/b7f506fe-03b2-4364-8e22-65af15f732cc)
      | ![image](https://github.com/user-attachments/assets/08e8cb7a-049d-4790-8e5d-6966a27a106d)
     |
     
| ![image](https://github.com/user-attachments/assets/e9c6664d-22a9-4271-9b82-c701ddc9eb27)
   | ![image](https://github.com/user-attachments/assets/ba400e91-d8bd-47e5-b6bd-ebf205147f36)
      | ![image](https://github.com/user-attachments/assets/44222c85-51aa-46bb-a258-e74782a8e68a)
     |

---

## Challenges and Solutions
- **Low Image Contrast:** Addressed using image normalization techniques.
- **Class Imbalance:** Mitigated with data augmentation and weighted loss functions.
- **High Noise Levels:** Leveraged residual blocks to capture fine-grained features.

---

## Deliverables
- Fully trained U-Net model with optimized weights.
- Python scripts for preprocessing, training, and inference.
- Visualizations of segmentation results and performance metrics.
- Detailed documentation explaining methodology and results.

---

## Repository Links
- **[Dataset BUSI](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)**.

---

## Future Work
- Expand dataset to improve generalizability.
- Explore advanced architectures like DeepLab or Mask R-CNN.
- Integrate additional preprocessing techniques for noise reduction.

---


For questions or collaborations, feel free to reach out!
