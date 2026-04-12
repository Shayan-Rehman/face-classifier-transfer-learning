# Real-Time Face Classifier using Transfer Learning

## Overview
This project uses transfer learning with a pretrained ResNet18 model from PyTorch Torchvision to classify whether an uploaded facial image belongs to a target person or not.

The project includes:
- a Jupyter notebook for training and evaluation
- a Streamlit app for real-time image classification
- a saved PyTorch model file

## Files
- `CNN_Transfer_Learning.ipynb` - training notebook
- `app.py` - Streamlit web app
- `face_classifier.pth` - trained model weights

## Tools Used
- Python
- PyTorch
- Torchvision
- Streamlit
- Scikit-learn

## How to Run

Install dependencies:
```bash
pip install torch torchvision streamlit pillow
streamlit run app.py
