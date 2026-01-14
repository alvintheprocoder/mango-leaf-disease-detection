# Mango Leaf Disease Detection

Deep Learning project for classifying mango leaf diseases using Convolutional Neural Networks (CNN).

## 🍃 About
This project uses CNN to classify mango leaves into 8 disease categories:
- Anthracnose
- Bacterial Canker
- Cutting Weevil
- Die Back
- Gall Midge
- Healthy
- Powdery Mildew
- Sooty Mould

## 📊 Results
- **Training Accuracy: 90.73%**
- **Validation Accuracy: 90.25%**

## 🚀 Usage

### Training the Model
```bash
python Mango.py
```

### Testing
```bash
# Quick test on 3 random images
python quick_test.py

# Interactive testing
python test_image.py
```

## 📦 Dataset
Download from: [Mango Leaf Disease Dataset](https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset)

Extract to: `dataset/train/` folder

## 🛠️ Requirements
- Python 3.x
- TensorFlow/Keras
- NumPy
- Matplotlib

Install dependencies:
```bash
pip install tensorflow numpy matplotlib
```

## 📁 Project Structure
```
CSC566 Mini Project/
├── Mango.py              # Main training script
├── test_image.py         # Interactive testing
├── quick_test.py         # Quick testing
├── best_model.keras      # Trained model (not included in repo)
├── dataset/              # Dataset folder (not included in repo)
└── *.png                 # Generated visualizations
```

## 👥 Team Members
- [Add your names here]

## 📄 Course
CSC566: Image Processing - Mini Project
