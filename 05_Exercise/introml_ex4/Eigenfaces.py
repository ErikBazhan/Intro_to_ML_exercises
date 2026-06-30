from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


N = 64
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
TRAINED_CLASSIFIERS = {}
TRAINED_STANDARDIZATION = {}

# Do not add further imports.
# Implement PCA and feature standardization with NumPy only.
# Do not use sklearn.decomposition.PCA or other pre-built standardization helpers.


def _build_classifier(classifier_type):
    if classifier_type == "logistic": # Logistic Regression
        return LogisticRegression(max_iter=2000)
    if classifier_type == "gaussian_nb": # Gaussian Naive Bayes
        return GaussianNB()
    raise ValueError(f"Unknown classifier type: {classifier_type}")


def _uses_feature_scaling(classifier_type): 
    return classifier_type == "logistic" #Logistic Regression needs normalized PCA coordinates
                                         # Gaussian NB uses PCA features directly

def _list_class_directories(dataset_root): # Capture all classes
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    class_dirs = [path for path in sorted(dataset_root.iterdir()) if path.is_dir()]
    if not class_dirs:
        raise ValueError(
            f"Expected at least one class subdirectory in {dataset_root}. "
            "Use a structure like dataset/class_name/image.png."
        )
    return class_dirs


def create_database_from_folder(dataset_root, image_size=(N, N)): # Load images from classes and make matrix
    """
    Load a local image dataset from class subdirectories.

    Expected structure:
        dataset_root/
            class_a/
                img_01.png
            class_b/
                img_02.png
    """
    labels = []
    train = []
    class_dirs = _list_class_directories(dataset_root)

    target_height = None
    target_width = None
    if image_size is not None:
        target_width, target_height = image_size

    for class_dir in class_dirs:
        image_paths = [
            path for path in sorted(class_dir.iterdir()) if path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not image_paths:
            raise ValueError(f"Class directory contains no supported images: {class_dir}")

        for image_path in image_paths:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Could not load image: {image_path}")

            if image_size is not None:
                img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
            elif target_height is None or target_width is None:
                target_height, target_width = img.shape
            elif (img.shape[1], img.shape[0]) != (target_width, target_height):
                raise ValueError(
                    "All images must share the same size when image_size is None. "
                    f"Expected {(target_width, target_height)}, got {(img.shape[1], img.shape[0])} from {image_path}."
                )

            train.append(img.reshape(-1).astype(np.float64)) # 1D vector
            labels.append(class_dir.name)

    if not train:
        raise ValueError(f"No images found below {dataset_root}")

    train = np.asarray(train, dtype=np.float64)
    return np.asarray(labels), train, train.shape[0], target_height, target_width


def calculate_average_face(train):
    """
    Calculate the average image using all training images.
    """
    return np.mean(np.asarray(train, dtype=np.float64), axis=0)


def calculate_eigenfaces(train, avg, num_eigenfaces):
    """
    Calculate the principal directions of the centered training set using SVD.
    """
    U, S, Vt = np.linalg.svd(train - avg, full_matrices=False) # Singular value decomposition of data in direction of max. variance
    # U contains the left singular vectors (eigenfaces), S contains the singular values, and Vt contains the right singular vectors.
    
    return Vt[:num_eigenfaces] # Return the first num_eigenfaces principal components (eigenfaces).


def get_feature_representation(images, eigenfaces, avg, num_eigenfaces):
    """
    Project all images into the PCA space spanned by the first num_eigenfaces components.
    """
    centered_images = images - avg # Center the images by subtracting the average face.
    # eigenfaces[:num_eigenfaces] because we only want to use the first num_eigenfaces principal components for the projection.
    # .T for matrix multiplication
    # np.dot scalar product to project the centered images onto the eigenfaces.
    return  np.dot(centered_images, eigenfaces[:num_eigenfaces].T)



def calculate_feature_statistics(features):
    """
    Compute the mean and standard deviation of every PCA feature over the training set.

    Standardize every feature by centering it to zero mean
    and rescaling it to unit standard deviation. Implement this with NumPy only.
    """
    feature_mean = np.mean(features, axis=0) # Compute the mean of each feature across all training samples.
    feature_std = np.std(features, axis=0) # Compute the standard deviation of each feature across all training samples.
    feature_std = np.where(feature_std == 0, 1, feature_std) # Replace any zero standard deviations with 1 to avoid division by zero during standardization.
    
    return feature_mean, feature_std

def standardize_features(features, feature_mean, feature_std):
    """
    Standardize all features using the previously computed mean and standard deviation.

    Apply the same transformation to the training features and later to every test image.
    """
    return (features - feature_mean) / feature_std # Standardize the features by subtracting the mean and dividing by the standard deviation.


def reconstruct_image(img, eigenfaces, avg, num_eigenfaces, h, w):
    """
    Reconstruct an image from the first num_eigenfaces principal components.
    """
    image_vector = np.asarray(img, dtype=np.float64).reshape(1, -1) # convert the input image to a 1D vector
    coefficients = get_feature_representation(image_vector, eigenfaces, avg, num_eigenfaces) # get the PCA coefficients for the input image
    reconstruction = avg + np.dot(coefficients, eigenfaces[:num_eigenfaces]) # reconstruct the image by adding the average face to the linear combination of the eigenfaces weighted by the PCA coefficients
    return reconstruction.reshape(h, w)


def process_and_train(labels, train, num_images, h, w, classifier_type="logistic", num_eigenfaces=None):
    """
    Compute PCA features and train one classifier on top of them.
    For Logistic Regression, standardize the PCA features with your own helper functions.
    """

    # get average face, eigenfaces and PCA features for the training set
    avg_img = calculate_average_face(train)
    eigenfaces = calculate_eigenfaces(train, avg_img, num_eigenfaces)
    features = get_feature_representation(train, eigenfaces, avg_img, num_eigenfaces)
    
    # Build the classifier based on the specified type (logistic regression or Gaussian Naive Bayes)
    classifier = _build_classifier(classifier_type)
    if _uses_feature_scaling(classifier_type):
        feature_mean, feature_std = calculate_feature_statistics(features)
        features_for_training = standardize_features(features, feature_mean, feature_std)
        TRAINED_STANDARDIZATION[classifier_type] = (feature_mean, feature_std) # Store the mean and standard deviation for later use during classification of test images
    else:
        features_for_training = features
        TRAINED_STANDARDIZATION.pop(classifier_type, None) # Remove any existing standardization statistics for this classifier type if it doesn't use feature scaling.

    classifier.fit(features_for_training, labels) # Train the classifier on the PCA features and corresponding labels.
    TRAINED_CLASSIFIERS[classifier_type] = classifier # Store the trained classifier 

    return eigenfaces, num_eigenfaces, avg_img


def train_both_classifiers(labels, train, num_images, h, w, num_eigenfaces=None):
    """
    Train Logistic Regression and Gaussian Naive Bayes on the same PCA features.
    For Logistic Regression, standardize the PCA features with your own helper functions.
    """


    avg_img = calculate_average_face(train)
    eigenfaces = calculate_eigenfaces(train, avg_img, num_eigenfaces)
    features = get_feature_representation(train, eigenfaces, avg_img, num_eigenfaces)

    # Logistic Regression uses standardized PCA features.
    logistic_model = _build_classifier("logistic")
    feature_mean, feature_std = calculate_feature_statistics(features)
    logistic_features = standardize_features(features, feature_mean, feature_std)
    logistic_model.fit(logistic_features, labels)
    TRAINED_CLASSIFIERS["logistic"] = logistic_model
    TRAINED_STANDARDIZATION["logistic"] = (feature_mean, feature_std)

    # Gaussian Naive Bayes uses raw PCA features.
    gaussian_model = _build_classifier("gaussian_nb")
    gaussian_model.fit(features, labels)
    TRAINED_CLASSIFIERS["gaussian_nb"] = gaussian_model
    TRAINED_STANDARDIZATION.pop("gaussian_nb", None)

    return eigenfaces, num_eigenfaces, avg_img


def classify_image(img, eigenfaces, avg, num_eigenfaces, h, w, classifier_type="logistic"):
    """
    Predict the class label of one image from its PCA coefficients.
    If Logistic Regression is used, apply the same feature standardization as during training.
    """
    image_vector = np.asarray(img, dtype=np.float64).reshape(1, -1)
    features = get_feature_representation(image_vector, eigenfaces, avg, num_eigenfaces)

    if classifier_type not in TRAINED_CLASSIFIERS:
        return np.asarray([None], dtype=object)

    classifier = TRAINED_CLASSIFIERS[classifier_type]

    if _uses_feature_scaling(classifier_type):
        if classifier_type not in TRAINED_STANDARDIZATION:
            raise ValueError(
                f"Missing standardization statistics for classifier '{classifier_type}'."
            )
        feature_mean, feature_std = TRAINED_STANDARDIZATION[classifier_type]
        features = standardize_features(features, feature_mean, feature_std)

    return classifier.predict(features)
