import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize

def preprocess_image(image_path):
    img = imread(image_path)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = rgb2gray(img)
    img_resized = resize(img, (20, 20), anti_aliasing=True)
    img_normalized = img_resized / 255.0
    img_flattened = img_normalized.flatten()
    return img_flattened

def predict_digit(perceptron, image_path):
    preprocessed_image = preprocess_image(image_path)
    preprocessed_image = preprocessed_image.reshape(1, -1)
    predictions = perceptron.forward(preprocessed_image)
    predicted_class = np.argmax(predictions)
    return predicted_class
