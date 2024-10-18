# skin_lesion_classifier
This project aims to create a system for detecting and classifying skin lesions using a deep learning model. The goal is to load an image of a mole or skin lesion and obtain information about its type, along with a detailed description. 
# Documentation for Skin Lesion Classification Code

## Introduction
This project aims to create a system for detecting and classifying skin lesions using a deep learning model. The goal is to load an image of a mole or skin lesion and obtain information about its type, along with a detailed description. The model was trained using a dataset of dermatoscopic lesion images, classified into different categories, including benign, malignant lesions, and other skin conditions.

## Code Structure
The code is divided into two main parts:
1. **Model Training**: Includes dataset loading, data preparation, training the NASNetMobile model, and fine-tuning.
2. **Prediction and Description**: Contains a function to load an image and predict the type of lesion, including a detailed description for each type.

### Part 1: Model Training

#### Importing Libraries
```python
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
```
These libraries are essential for data handling, deep learning, and model training. `TensorFlow` is used to create and train the deep learning model.

#### Loading Metadata and Images
```python
train_metadata_path = 'derm12345_metadata_train.csv'
test_metadata_path = 'derm12345_metadata_test.csv'

train_metadata_df = pd.read_csv(train_metadata_path)
test_metadata_df = pd.read_csv(test_metadata_path)
```
The metadata contains information about the images, such as the image ID and lesion label. These data are loaded from CSV files.

#### Function to Load Images
```python
def load_images(metadata_df, base_path):
    images = []
    labels = []
    for _, row in metadata_df.iterrows():
        image_path = os.path.join(base_path, row['image_id'] + '.jpg')
        if os.path.exists(image_path):
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img_array)
            labels.append(row['label'])
    return np.array(images), np.array(labels)
```
This function loads the images from the directories specified in the metadata. The images are resized to 224x224 to be compatible with the model.

#### Preparing the Dataset
```python
X_train_1, y_train_1 = load_images(train_metadata_df, 'derm12345_train_part_1')
X_train_2, y_train_2 = load_images(train_metadata_df, 'derm12345_train_part_2')

X_train = np.concatenate((X_train_1, X_train_2), axis=0)
y_train = np.concatenate((y_train_1, y_train_2), axis=0)

X_test, y_test = load_images(test_metadata_df, 'derm12345_test')
```
The training and test images are loaded and combined to form the complete training and test datasets.

#### Encoding Labels
```python
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
```
The labels are encoded into integer values using `LabelEncoder` to facilitate the model training process.

#### Splitting into Training and Validation Sets
```python
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
```
The training dataset is split into a training set and a validation set to evaluate model performance during training.

#### Data Augmentation
```python
data_gen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
```
Data augmentation is applied to increase the variability of the training dataset and improve the model's generalization capability.

#### Creating and Training the Model
```python
base_model = NASNetMobile(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(len(label_encoder.classes_), activation='softmax'))
```
The NASNetMobile base model is used as a feature extractor, followed by additional layers to adapt to the classification of skin lesions.

#### Model Optimization
```python
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
```
The model is compiled with the Adam optimizer, and dynamic learning rate reduction is applied to improve training efficiency.

### Part 2: Prediction and Description of Lesion

#### Prediction Function
```python
def predict_image_with_description(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    class_label = class_labels[class_idx]
```
This function loads an image and uses the trained model to predict the class of the lesion.

#### Class Descriptions
```python
descriptions = {
    'melanocytic_benign': "This is a benign melanocytic nevus, which is generally harmless.",
    'melanocytic_malignant': "This is a malignant melanoma. It is important to consult a dermatologist immediately.",
    'bdb': "Banal Dermal Nevus: A benign type of mole that is typically located in the dermis layer of the skin. It is usually harmless but should be monitored for changes in shape or color.",
    'sk': "Seborrheic Keratosis: A common noncancerous skin growth that often appears as a brown, black, or pale growth on the face, chest, shoulders, or back. It is generally harmless and does not require treatment unless it becomes irritated."
    # ... Other descriptions ...
}
```
Each class is accompanied by a detailed description, which helps provide meaningful information to the user regarding the type of lesion.

#### Example Usage
```python
image_to_predict = "foto67.jpg"
print(predict_image_with_description(image_to_predict))
```
This part of the code provides an example of how to use the function to predict the class of a lesion and get a description.

## Conclusion
This skin lesion classification system is designed to help in the diagnosis of skin lesions through the use of deep learning. The NASNetMobile model is used as a base to extract features from the images, while additional layers are added to adapt to the specificities of the skin lesion dataset. The system provides a detailed description for each type of lesion, making it a useful tool for users and healthcare professionals in evaluating skin lesions.

