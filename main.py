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

# Load metadata CSV files
train_metadata_path = 'derm12345_metadata_train.csv'
test_metadata_path = 'derm12345_metadata_test.csv'

train_metadata_df = pd.read_csv(train_metadata_path)
test_metadata_df = pd.read_csv(test_metadata_path)

# Combine train parts
train_part_1_path = 'derm12345_train_part_1'
train_part_2_path = 'derm12345_train_part_2'

# Helper function to load images based on metadata
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

# Load training images from both parts
X_train_1, y_train_1 = load_images(train_metadata_df, train_part_1_path)
X_train_2, y_train_2 = load_images(train_metadata_df, train_part_2_path)

# Concatenate the training parts
X_train = np.concatenate((X_train_1, X_train_2), axis=0)
y_train = np.concatenate((y_train_1, y_train_2), axis=0)

# Load test images
test_part_path = 'derm12345_test'
X_test, y_test = load_images(test_metadata_df, test_part_path)

# Encode labels as integers and then convert to one-hot encoding
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(label_encoder.classes_))
y_val = tf.keras.utils.to_categorical(y_val, num_classes=len(label_encoder.classes_))
y_test = tf.keras.utils.to_categorical(y_test, num_classes=len(label_encoder.classes_))

# Image Data Generator with augmentation for training
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

# Image Data Generator for validation and test (no augmentation)
data_gen_val_test = ImageDataGenerator(rescale=1./255)

# Creating data generators
gen_train = data_gen_train.flow(X_train, y_train, batch_size=32)
gen_val = data_gen_val_test.flow(X_val, y_val, batch_size=32)
gen_test = data_gen_val_test.flow(X_test, y_test, batch_size=32)

# Load the pre-trained NASNetMobile model and fine-tune
base_model = NASNetMobile(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model initially

# Create the model
model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

# Train the model
history = model.fit(
    gen_train,
    validation_data=gen_val,
    epochs=50,
    callbacks=[early_stopping, reduce_lr]
)

# Fine-tune the base model
base_model.trainable = True
for layer in base_model.layers[:100]:  # Freeze first 100 layers to avoid overfitting
    layer.trainable = False

# Recompile with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Fine-tune the model
history_fine = model.fit(
    gen_train,
    validation_data=gen_val,
    epochs=30,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate on test set
test_loss, test_acc = model.evaluate(gen_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Save the model
model.save('skin_lesion_classifier.h5')

# Function to load and predict new images
def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    class_label = label_encoder.inverse_transform([class_idx])[0]
    return class_label

# Example prediction
image_to_predict = "foto32.jpg"
print(f"Predicted Class: {predict_image(image_to_predict)}")





#copyright Cristian Ferrara 2024