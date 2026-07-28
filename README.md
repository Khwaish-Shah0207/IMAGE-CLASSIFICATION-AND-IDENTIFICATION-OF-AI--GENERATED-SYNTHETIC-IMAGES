# AI-Generated Image Detection using Convolutional Neural Networks (CNN)

## 📌 Overview

This project focuses on detecting whether an image is **REAL** or **AI-generated (FAKE)** using a Convolutional Neural Network (CNN). The model is trained on a curated subset of the **CIFAKE dataset**, which contains authentic images from CIFAR-10 and synthetic images generated using Stable Diffusion.

The project implements an end-to-end machine learning pipeline, including dataset preparation, image preprocessing, CNN model development, training, evaluation, and single-image prediction.

---

## 🎯 Objectives

- Build a CNN model to classify images as **REAL** or **FAKE**.
- Prepare a reproducible subset of the CIFAKE dataset.
- Preprocess images for efficient model training.
- Train and evaluate the model using classification metrics.
- Predict whether a new image is AI-generated with a confidence score.

---

## 📂 Dataset

**Dataset:** CIFAKE – Real and AI-Generated Synthetic Images

The original dataset contains **120,000 images**.

For this project, a reproducible subset was created:

| Split | REAL | FAKE |
|-------|------|------|
| Train | 400 | 400 |
| Test | 100 | 100 |

Images are resized to **32 × 32 pixels** before training.

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- pathlib
- shutil
- Google Colab / VS Code

---

## 🧠 CNN Architecture

The model consists of:

- Conv2D (32 filters)
- MaxPooling2D
- Conv2D (64 filters)
- MaxPooling2D
- Conv2D (128 filters)
- Flatten Layer
- Dense (128 units)
- Dropout (0.5)
- Dense (1 unit, Sigmoid)

---

## 🚀 Project Workflow

```
Raw CIFAKE Dataset
        │
        ▼
Dataset Preparation
        │
        ▼
Image Preprocessing
        │
        ▼
CNN Model
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Single Image Prediction
```

---

## 📁 Project Structure

```
AI-Generated-Image-Detection/
│
├── dataset/
│   ├── train/
│   └── test/
│
├── models/
│   └── cnn_model.keras
│
├── outputs/
│   ├── accuracy.png
│   ├── loss.png
│   └── confusion_matrix.png
│
├── prepare_dataset.py
├── preprocess.py
├── model.py
├── train.py
├── predict.py
│
├── requirements.txt
└── README.md
```

---

## 📊 Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Binary Cross-Entropy Loss

Training and validation performance is visualized using:

- Accuracy Curve
- Loss Curve
- Confusion Matrix

---

## 🔍 Features

- Reproducible dataset sampling
- Image normalization and preprocessing
- Custom CNN architecture
- Model training with TensorFlow/Keras
- Accuracy and loss visualization
- Single-image prediction with confidence score

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Generated-Image-Detection.git
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

```bash
python prepare_dataset.py
```

### 4. Train the model

```bash
python train.py
```

### 5. Predict a new image

```bash
python predict.py
```

Enter the image path when prompted.

---

## 📷 Sample Outputs

- Training Accuracy Curve
- Training Loss Curve
- Confusion Matrix
- REAL / FAKE Prediction with Confidence Score

---

## 🔮 Future Enhancements

- Train on the complete CIFAKE dataset
- Apply data augmentation techniques
- Use transfer learning (EfficientNet, ResNet)
- Integrate Grad-CAM for explainable AI
- Build a Streamlit or Flask web application
- Deploy the model using Docker and REST APIs

---

## 📚 Reference

Bird, J. J., & Lotfi, A. (2024). *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images*. IEEE Access.

---

## 👩‍💻 Author

**Khwaish Shah**
AI & Machine Learning Enthusiast
