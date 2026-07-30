from pathlib import Path
import random
import shutil
random.seed(42)
BASE_DIR = Path(__file__).resolve().parent.parent
# Original downloaded dataset
SOURCE_DIR = BASE_DIR / "cifake_full"
# New smaller dataset
DEST_DIR = BASE_DIR / "dataset"
TRAIN_IMAGES_PER_CLASS = 400
TEST_IMAGES_PER_CLASS = 100
def copy_images(source_folder, destination_folder, num_images):
    destination_folder.mkdir(parents=True, exist_ok=True)
    image_files = [f for f in source_folder.iterdir() if f.is_file()]
    print(f"\nChecking folder:")
    print(source_folder)
    print(f"Found {len(image_files)} images.")
    if len(image_files) < num_images:
        raise ValueError(
            f"\nNot enough images!\n"
            f"Found: {len(image_files)}\n"
            f"Required: {num_images}"
        )
    selected_images = random.sample(image_files, num_images)
    for image in selected_images:
        shutil.copy2(image, destination_folder / image.name)
    print(f"Copied {num_images} images to:")
    print(destination_folder)
def main():
    print("=" * 60)
    print("        CIFAKE DATASET PREPARATION")
    print("=" * 60)
    # Training
    copy_images(
        SOURCE_DIR / "train" / "REAL",
        DEST_DIR / "train" / "REAL",
        TRAIN_IMAGES_PER_CLASS,
    )
    copy_images(
        SOURCE_DIR / "train" / "FAKE",
        DEST_DIR / "train" / "FAKE",
        TRAIN_IMAGES_PER_CLASS,
    )
    # Testing
    copy_images(
        SOURCE_DIR / "test" / "REAL",
        DEST_DIR / "test" / "REAL",
        TEST_IMAGES_PER_CLASS,
    )
    copy_images(
        SOURCE_DIR / "test" / "FAKE",
        DEST_DIR / "test" / "FAKE",
        TEST_IMAGES_PER_CLASS,
    )
    print("\n" + "=" * 60)
    print("Dataset preparation completed successfully!")
    print("=" * 60)
if __name__ == "__main__":
    main()
