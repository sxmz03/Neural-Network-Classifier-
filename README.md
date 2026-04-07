# Fashion Product Classifier

CNN trained to classify online fashion retail products into sub-categories from image data. Uses aggressive augmentation — rotation ±30°, translation ±20%, scaling 80–120% via OpenCV — to improve generalisation across the multi-class problem.

Architecture: Conv2D → MaxPooling → Dropout → Dense → softmax. Adam optimiser.

