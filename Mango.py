import numpy as np
import matplotlib.pyplot as plt
import random
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Configuration
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20
DATA_DIR = 'dataset/train'

# Get class names
class_names = sorted(os.listdir(DATA_DIR))
num_classes = len(class_names)
print(f"Found {num_classes} classes: {class_names}")

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 80% train, 20% validation
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode='nearest'
)

# Training data generator
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Validation data generator
validation_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")

# Display sample images
plt.figure(figsize=(12, 8))
sample_batch = next(train_generator)
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(sample_batch[0][i])
    label_idx = np.argmax(sample_batch[1][i])
    plt.title(class_names[label_idx])
    plt.axis('off')
plt.tight_layout()
plt.savefig('sample_images.png')
plt.show()

# Build improved CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Test prediction on random validation image
val_batch = next(validation_generator)
idx = random.randint(0, len(val_batch[0])-1)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(val_batch[0][idx])
plt.title("Input Image")
plt.axis('off')

# Make prediction
img_array = np.expand_dims(val_batch[0][idx], axis=0)
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
confidence = predictions[0][predicted_class] * 100

plt.subplot(1, 2, 2)
plt.barh(class_names, predictions[0])
plt.xlabel('Probability')
plt.title(f'Predicted: {class_names[predicted_class]}\nConfidence: {confidence:.2f}%')
plt.tight_layout()
plt.savefig('prediction_result.png')
plt.show()

print(f"\nPrediction: {class_names[predicted_class]}")
print(f"Confidence: {confidence:.2f}%")
print(f"\nActual label: {class_names[np.argmax(val_batch[1][idx])]}")

# ==========================================================
# TEST SET EVALUATION (UNSEEN DATA)
# ==========================================================

import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, confusion_matrix

TEST_DIR = "dataset/test"

# Test data generator (NO augmentation)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# Evaluate model on test set
test_loss, test_acc = model.evaluate(test_generator, verbose=1)

# Error rate
test_error_rate = 1.0 - test_acc

# Predictions
y_probs = model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_probs, axis=1)
y_true = test_generator.classes

# Precision & Recall
precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Display results
print("\n" + "="*60)
print("TEST SET EVALUATION RESULTS")
print("="*60)
print(f"Test Accuracy Rate : {test_acc:.4f}")
print(f"Test Error Rate    : {test_error_rate:.4f}")
print(f"Mean Precision     : {precision:.4f}")
print(f"Mean Recall        : {recall:.4f}")
print("="*60)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix (Test Set)")
plt.colorbar()
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(range(num_classes), class_names, rotation=45, ha="right")
plt.yticks(range(num_classes), class_names)

for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()
