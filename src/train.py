from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from preprocess import load_datasets
from model import build_model
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
train_dataset, test_dataset, class_names = load_datasets()
model = build_model()
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=10
)

model.save(MODELS_DIR / "cnn_model.keras")
print("\nModel saved successfully!")
#Accuracy
plt.figure(figsize=(8,5))
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(OUTPUTS_DIR / "accuracy.png")
plt.close()
# Loss
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(OUTPUTS_DIR / "loss.png")
plt.close()
print("Graphs saved successfully!")
