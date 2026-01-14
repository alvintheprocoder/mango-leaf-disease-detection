import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import random

# Load model
model = load_model('best_model.keras')
class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 
               'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']

# Test 3 random images
data_dir = 'dataset/train'
results = []

print("\nTesting 3 Random Images:\n" + "="*60)

for i in range(3):
    # Pick random image
    random_class = random.choice(class_names)
    class_path = os.path.join(data_dir, random_class)
    images = os.listdir(class_path)
    random_image = random.choice(images)
    img_path = os.path.join(class_path, random_image)
    
    # Load and predict
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class] * 100
    
    # Store results
    results.append({
        'img': img,
        'actual': random_class,
        'predicted': class_names[predicted_class],
        'confidence': confidence,
        'probs': predictions[0]
    })
    
    # Print results
    status = "✓ CORRECT" if random_class == class_names[predicted_class] else "✗ WRONG"
    print(f"\nTest {i+1}:")
    print(f"  Actual: {random_class}")
    print(f"  Predicted: {class_names[predicted_class]} ({confidence:.1f}%)")
    print(f"  {status}")

# Display all results
fig, axes = plt.subplots(3, 2, figsize=(12, 12))

for i, res in enumerate(results):
    # Show image
    axes[i, 0].imshow(res['img'])
    axes[i, 0].set_title(f"Actual: {res['actual']}", fontsize=10)
    axes[i, 0].axis('off')
    
    # Show predictions
    colors = ['green' if res['predicted'] == res['actual'] else 'red' if j == np.argmax(res['probs']) else 'lightblue' 
              for j in range(len(class_names))]
    axes[i, 1].barh(class_names, res['probs'], color=colors)
    axes[i, 1].set_xlabel('Confidence')
    axes[i, 1].set_title(f"Predicted: {res['predicted']} ({res['confidence']:.1f}%)", fontsize=10)
    axes[i, 1].set_xlim([0, 1])

plt.tight_layout()
plt.savefig('test_results.png', dpi=150, bbox_inches='tight')
print(f"\n{'='*60}")
print("Results saved to: test_results.png")
plt.show()
