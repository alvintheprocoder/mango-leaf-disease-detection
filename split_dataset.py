import os
import shutil
import random

SOURCE_DIR = "dataset/train"
TARGET_DIR = "dataset/test"
SPLIT_RATIO = 0.2   # 20% test data

random.seed(42)

for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    test_count = int(len(images) * SPLIT_RATIO)
    test_images = images[:test_count]

    target_class_path = os.path.join(TARGET_DIR, class_name)
    os.makedirs(target_class_path, exist_ok=True)

    for img in test_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(target_class_path, img)
        shutil.move(src, dst)

    print(f"{class_name}: moved {test_count} images to test set")

print("✅ Dataset split completed.")
