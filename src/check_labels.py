from preprocess import load_datasets

train_dataset, test_dataset, class_names = load_datasets()

print("Class names:", class_names)

for images, labels in train_dataset.take(1):
    print("First 10 labels:")
    print(labels[:10].numpy().flatten())