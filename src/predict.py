from pathlib import Path
import tensorflow as tf
import numpy as np

# Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "cnn_model.keras"
# Load Model
model = tf.keras.models.load_model(MODEL_PATH)
print("Model Loaded Successfully!\n")
# Image Path
image_path = input("Enter image path: ").strip()

# Load Image

img = tf.keras.utils.load_img(
    image_path,
    target_size=(32, 32)
)

img_array = tf.keras.utils.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)
# Prediction
prediction = model.predict(img_array, verbose=0)[0][0]

if prediction >= 0.5:
    label = "FAKE"
    confidence = prediction

else:
    label = "REAL"
    confidence = 1 - prediction

print("=" * 40)
print(f"Prediction : {label}")
print(f"Confidence : {confidence*100:.2f}%")
print("=" * 40)