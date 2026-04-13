# Real-Time Face Classifier using Transfer Learning

## Overview
This project uses transfer learning with a pretrained ResNet18 model from PyTorch Torchvision to classify whether an uploaded facial image belongs to a target person or not.

The project includes:
- a Jupyter notebook for training and evaluation
- a Streamlit app for real-time image classification
- a trained PyTorch model file (generated after training)

## Files
- `CNN_Transfer_Learning.ipynb` - training notebook
- `app.py` - Streamlit web app
-  face_classifier.pth - generated after training (not included due to size)

## Model File

The trained model file (`face_classifier.pth`) is not included in this repository due to GitHub file size limits.

To generate the model file:
1. Run the Jupyter notebook (`CNN_Transfer_Learning.ipynb`)
2. The model will be saved automatically at the end of training

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
