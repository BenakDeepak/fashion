import os
import numpy as np
import pickle
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.preprocessing import normalize

# Directory containing images
IMAGE_DIR = 'C:\\Users\\Deepak\\Desktop\\projects\\fashion\\data'
EMBEDDINGS_FILE = 'embeddings.pkl'
FILENAMES_FILE = 'filenames.pkl'
PROCESSED_IMAGES_FILE = 'processed_images.pkl'

# Load ResNet50 model with pre-trained weights, excluding top layers
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

# Function to extract features from an image
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

# Load processed images
def load_processed_images():
    if os.path.exists(PROCESSED_IMAGES_FILE):
        with open(PROCESSED_IMAGES_FILE, 'rb') as f:
            return set(pickle.load(f))
    return set()

# Save processed images
def save_processed_images(processed_images):
    with open(PROCESSED_IMAGES_FILE, 'wb') as f:
        pickle.dump(list(processed_images), f)

# Function to update embeddings when new images are added
def update_embeddings():
    processed_images = load_processed_images()

    # Load existing embeddings and filenames
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(FILENAMES_FILE):
        with open(EMBEDDINGS_FILE, 'rb') as f:
            old_embeddings = pickle.load(f)
        with open(FILENAMES_FILE, 'rb') as f:
            old_filenames = pickle.load(f)
    else:
        old_embeddings = np.empty((0, 2048))  # Assuming 2048 is the feature size
        old_filenames = []

    new_embeddings = []
    new_filenames = []

    # Convert old_filenames to a set for quick lookup
    existing_filenames = set(old_filenames)

    # Find new images to process
    for clothing_type in os.listdir(IMAGE_DIR):
        clothing_dir = os.path.join(IMAGE_DIR, clothing_type)
        if os.path.isdir(clothing_dir):
            for filename in os.listdir(clothing_dir):
                img_path = os.path.join(clothing_dir, filename)
                relative_filename = os.path.join(clothing_type, filename)
                if os.path.isfile(img_path) and relative_filename not in processed_images and relative_filename not in existing_filenames:
                    features = extract_features(img_path)
                    new_filenames.append(relative_filename)
                    new_embeddings.append(features)
                    processed_images.add(relative_filename)

    # If there are new embeddings, update the files
    if new_embeddings:
        new_embeddings = np.array(new_embeddings)
        new_embeddings = normalize(new_embeddings)  # Normalize embeddings

        # Combine old and new embeddings without reprocessing old files
        all_embeddings = np.vstack([old_embeddings, new_embeddings])
        all_filenames = old_filenames + new_filenames

        # Save the updated embeddings and filenames
        with open(EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump(all_embeddings, f)

        with open(FILENAMES_FILE, 'wb') as f:
            pickle.dump(all_filenames, f)

        save_processed_images(processed_images)

if __name__ == "__main__":
    # Update embeddings when new images are added
    update_embeddings()
