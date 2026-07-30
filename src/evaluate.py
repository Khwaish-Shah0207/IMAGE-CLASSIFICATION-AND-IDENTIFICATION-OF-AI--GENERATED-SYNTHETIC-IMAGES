from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
from preprocess import load_datasets

# Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "cnn_model.keras"
OUTPUTS_DIR = BASE_DIR / "outputs"
# Load Dataset
train_dataset, test_dataset, class_names = load_datasets()
# Load Model
model = tf.keras.models.load_model(MODEL_PATH)
print("=" * 50)
print("MODEL LOADED SUCCESSFULLY")
print("=" * 50)
# Evaluate
loss, accuracy = model.evaluate(test_dataset, verbose=1)
print(f"\nTest Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy*100:.2f}%")
# Predictions
y_true = []
y_pred = []
for images, labels in test_dataset:
    predictions = model.predict(images, verbose=0)
    predictions = (predictions > 0.5).astype(int)
    y_true.extend(labels.numpy().flatten())
    y_pred.extend(predictions.flatten())
y_true = np.array(y_true, dtype=int)
y_pred = np.array(y_pred, dtype=int)

# Classification Report
print("\nClassification Report\n")
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names
)
print(report)
# Save report
with open(OUTPUTS_DIR / "classification_report.txt", "w") as file:
    file.write(report)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)
plt.figure(figsize=(6,6))
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig(OUTPUTS_DIR / "confusion_matrix.png")
plt.close()
print("\nConfusion Matrix saved!")
print("\nEvaluation completed successfully!")
