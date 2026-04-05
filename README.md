# Neural Network Image Classifier

A convolutional neural network (CNN) that classifies online fashion retail products into sub-categories using image data and real-time data augmentation.

## Overview

This project applies deep learning to classify product images from a large fashion dataset. A CNN is trained with aggressive data augmentation to improve generalisation across a multi-class classification problem covering various clothing and accessory sub-categories.

## Features

- CNN with Conv2D, MaxPooling, Dropout, and Dense layers
- Real-time data augmentation: rotation (±30°), translation (±20%), and scaling (80–120%) via OpenCV
- Adam optimiser with configurable learning rate
- Multi-class softmax output
- Training and validation accuracy/loss plots

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| TensorFlow / Keras | Neural network construction and training |
| OpenCV (cv2) | Image loading and augmentation |
| NumPy / Pandas | Data handling |
| Matplotlib | Performance visualisation |
| scikit-learn | Train/test splitting |

## Model Architecture

```
Input (128×128 RGB)
  → Conv2D → MaxPooling2D
  → Conv2D → MaxPooling2D
  → Flatten
  → Dense → Dropout
  → Dense (Softmax)
```

## Setup

```bash
pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn
```

Update the dataset paths in `main.py`:

```python
dataset = 'path/to/images'
labels_file = 'path/to/styles.csv'
```

Then run:

```bash
python main.py
```
