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