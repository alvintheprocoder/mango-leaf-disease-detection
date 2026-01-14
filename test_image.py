import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load the trained model
model = load_model('best_model.keras')

# Disease classes
class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 
               'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']

def predict_disease(img_path):
    """Predict disease from an image file"""
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class] * 100
    
    # Display results
    plt.figure(figsize=(12, 5))
    
    # Show image
    plt.subplot(1, 2, 1)
    plt.imshow(image.load_img(img_path))
    plt.title(f"Input Image\n{os.path.basename(img_path)}")
    plt.axis('off')
    
    # Show predictions
    plt.subplot(1, 2, 2)
    colors = ['red' if i == predicted_class else 'lightblue' for i in range(len(class_names))]
    plt.barh(class_names, predictions[0], color=colors)
    plt.xlabel('Confidence')
    plt.title(f'Prediction: {class_names[predicted_class]}\nConfidence: {confidence:.2f}%')
    plt.xlim([0, 1])
    
    plt.tight_layout()
    plt.savefig('test_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return class_names[predicted_class], confidence

def test_random_images(num_images=5):
    """Test on random images from the dataset"""
    data_dir = 'dataset/train'
    
    for i in range(num_images):
        # Pick random class and image
        random_class = np.random.choice(class_names)
        class_path = os.path.join(data_dir, random_class)
        images = os.listdir(class_path)
        random_image = np.random.choice(images)
        img_path = os.path.join(class_path, random_image)
        
        print(f"\n{'='*50}")
        print(f"Test {i+1}/{num_images}")
        print(f"Actual: {random_class}")
        print(f"Image: {random_image}")
        print(f"{'='*50}")
        
        predicted, confidence = predict_disease(img_path)
        
        print(f"Predicted: {predicted}")
        print(f"Confidence: {confidence:.2f}%")
        
        if predicted == random_class:
            print("✓ CORRECT!")
        else:
            print("✗ INCORRECT")
        
        input("\nPress Enter to test next image...")

if __name__ == "__main__":
    print("Mango Leaf Disease Detection - Testing")
    print("="*50)
    print("\nOptions:")
    print("1. Test a specific image file")
    print("2. Test random images from dataset")
    
    choice = input("\nEnter choice (1 or 2): ")
    
    if choice == "1":
        img_path = input("\nEnter image path: ")
        if os.path.exists(img_path):
            disease, conf = predict_disease(img_path)
            print(f"\nPrediction: {disease}")
            print(f"Confidence: {conf:.2f}%")
        else:
            print("Image file not found!")
    
    elif choice == "2":
        num = input("\nHow many images to test? (default 5): ")
        num = int(num) if num else 5
        test_random_images(num)
    
    else:
        print("Invalid choice!")
