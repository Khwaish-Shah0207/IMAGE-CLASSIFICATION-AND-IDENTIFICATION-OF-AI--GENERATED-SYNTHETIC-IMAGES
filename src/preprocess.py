from pathlib import Path
import tensorflow as tf
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"
TEST_DIR = BASE_DIR / "dataset" / "test"
IMAGE_SIZE = (32, 32)
BATCH_SIZE = 32
SEED = 42
def load_datasets():
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="binary",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )
    class_names = train_dataset.class_names
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="binary",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    train_dataset = train_dataset.map(
        lambda x, y: (normalization_layer(x), y)
    )
    test_dataset = test_dataset.map(
        lambda x, y: (normalization_layer(x), y)
    )
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(AUTOTUNE)
    test_dataset = test_dataset.prefetch(AUTOTUNE)
    return train_dataset, test_dataset, class_names

if __name__ == "__main__":
    train_dataset, test_dataset, class_names = load_datasets()
    print("Training batches:", len(train_dataset))
    print("Testing batches:", len(test_dataset))
    print("Classes:", class_names)

