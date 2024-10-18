import numpy as np
import tensorflow as tf
import pandas as pd

# Load the saved model
model = tf.keras.models.load_model('skin_lesion_classifier.h5')

# Load the metadata CSV to retrieve the classes
train_metadata_path = 'derm12345_metadata_train.csv'
train_metadata_df = pd.read_csv(train_metadata_path)

# Encode labels to get class names
class_labels = train_metadata_df['label'].unique()
class_labels.sort()  # Sorting to ensure consistency


# Function to load and predict new images with description
def predict_image_with_description(image_path):
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    class_label = class_labels[class_idx]

    # Dictionary for descriptions of different types of nevi
    descriptions = {
        'melanocytic_benign': "This is a benign melanocytic nevus, which is generally harmless.",
        'melanocytic_malignant': "This is a malignant melanoma. It is important to consult a dermatologist immediately.",
        'keratinocytic': "This is a benign keratinocytic lesion, usually harmless but should be monitored for any changes.",
        'vascular': "This is a benign vascular lesion. Typically, no treatment is needed unless it changes.",
        'actinic_keratosis': "This is an actinic keratosis, a precancerous condition. Medical advice is recommended.",
        'basal_cell_carcinoma': "This is a basal cell carcinoma, a type of skin cancer. It is important to get a medical evaluation.",
        'jd': "Dysplastic Junctional Nevus: This is a dysplastic junctional nevus, a type of melanocytic lesion that is generally benign but may carry an increased risk of developing into melanoma. It should be monitored for any changes.",
        'cd': "Dysplastic Compound Nevus: A benign but atypical mole that can sometimes resemble melanoma. Regular monitoring is advised.",
        'rd': "Dysplastic Recurrent Nevus: A recurrent nevus that appears after partial removal of a previous mole. It is generally benign but should be monitored for changes.",
        'ak': "Actinic Keratosis: A rough, scaly patch on the skin caused by years of sun exposure. It is a precancerous lesion, and medical treatment is recommended.",
        'bcc': "Basal Cell Carcinoma: A type of skin cancer that grows slowly and rarely spreads. It is important to have it treated to prevent further complications.",
        'mel': "Melanoma: A dangerous form of skin cancer that can spread rapidly. Immediate consultation with a dermatologist is crucial.",
        'bdb': "Banal Dermal Nevus: A benign type of mole that is typically located in the dermis layer of the skin. It is usually harmless but should be monitored for changes in shape or color.",
        'sk': "Seborrheic Keratosis: A common noncancerous skin growth that often appears as a brown, black, or pale growth on the face, chest, shoulders, or back. It is generally harmless and does not require treatment unless it becomes irritated.",
        # Add more descriptions as needed for other classes
    }

    # Get the description for the predicted class
    description = descriptions.get(class_label, "Unknown type of lesion. Consult a dermatologist for more information.")

    return f"Predicted Class: {class_label}\nDescription: {description}"


# Example usage
image_to_predict = "foto24.jpg"
print(predict_image_with_description(image_to_predict))


#Copyright Cristian Ferrara 2024