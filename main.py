import cv2
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.optimizers import adam_v2
import matplotlib.pyplot as plt
import pandas as pd

#help with distributed dataset for tensorflow
def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)
data_adapter._is_distributed_dataset = _is_distributed_dataset


######################################################
#data reprocessing
dataset = r'C:/Users/charl/OneDrive/Documents/Uni Work/Artificial Intelligence/Dataset/images'
labels_file = r'C:/Users/charl/OneDrive/Documents/Uni Work/Artificial Intelligence/Dataset/styles.csv'


labels_df = pd.read_csv(labels_file)
labels_dict = dict(zip(labels_df["id"], labels_df["subCategory"]))

#get class names
classes = labels_df["subCategory"].unique().tolist()

#random transformations
def random_transform(image):
    #rotation
    angle = random.uniform(-30, 30)  # Rotate between -30 and 30 degrees
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    image = cv2.warpAffine(image, M, (w, h))

    #translation
    tx = random.uniform(-0.2 * w, 0.2 * w)  # Translate horizontally by up to 20%
    ty = random.uniform(-0.2 * h, 0.2 * h)  # Translate vertically by up to 20%
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    image = cv2.warpAffine(image, M, (w, h))

    #scaling
    scale = random.uniform(0.8, 1.2)  # Scale between 80% and 120%
    image = cv2.resize(image, None, fx=scale, fy=scale)

    #resize image
    image = cv2.resize(image, (128, 128))
    return image

data = []
labels = []

#makes dataset smaller
count = 0

for image_file in os.listdir(dataset):
    if image_file.endswith(".jpg") or image_file.endswith(".png"):
        image_path = os.path.join(dataset, image_file)

        #get and preprocess the image
        img = cv2.imread(image_path)
        img = cv2.resize(img, (128, 128))

        #calls random transformations
        img = random_transform(img)

        img = img / 255.0  # Normalize pixel values
        data.append(img)

        #get label for the image
        label = labels_dict.get(image_file)
        if label is None:
            print(f"Label missing for {image_file}, skipping.")
            continue
        labels.append(classes.index(label))
        print(count)
        if count == 4999:
            break
        count += 1

#convert data and labels to arrays
data = np.array(data)
labels = np.array(labels)

print(f"Dataset size: {len(data)} images")

#split training and test data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

######################################################
#CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),  # New convolutional layer
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),  # Additional convolutional layer
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),  # Increased size of dense layer
    Dropout(0.3),
    Dense(128, activation='relu'),  # Additional dense layer
    Dropout(0.1),
    Dense(len(classes), activation='softmax')  # Output layer
])

#adjust learning rate
optimizer = adam_v2.Adam(learning_rate=0.001)

#compile model
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



#####################################
#train model
history = model.fit(X_train, y_train, epochs=9, validation_split=0.2, batch_size=32)

#evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")#


#############################################
#get accuracy graph
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.show()