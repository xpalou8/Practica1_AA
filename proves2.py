import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Define constants
image_size = (200, 200)
orientations = 8
pixels_per_cell = (12, 12)
cells_per_block = (1, 1)

# Function to extract HOG features from an image
def extract_hog_features(image):
    fd, _ = hog(
        image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=True,
    )
    return fd

# Function to load and preprocess images
def load_and_preprocess_images(folder_path):
    X = []
    y = []

    for class_label, class_name in enumerate(os.listdir(folder_path)):
        class_folder = os.path.join(folder_path, class_name)
        if os.path.isdir(class_folder):
            for image_filename in os.listdir(class_folder):
                if image_filename.endswith(".jpg") or image_filename.endswith(".png"):
                    image_path = os.path.join(class_folder, image_filename)

                    # Load and resize the image
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    image = cv2.resize(image, image_size)

                    # Extract HOG features
                    hog_features = extract_hog_features(image)

                    X.append(hog_features)
                    y.append(class_label)
            print("***************************CARPETA TERMINADA***********************************")

    return X, y

# Load and preprocess training and testing data
X_train, y_train = load_and_preprocess_images("Perceptron/Practica1/dat/a2/data/train")
print("**TRAIN TERMINADO**")
X_test, y_test = load_and_preprocess_images("Perceptron/Practica1/dat/a2/data/test")
print("**TEST TERMINADO**")

print(X_train)
print(y_train)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a linear SVM model
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)
