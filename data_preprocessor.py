import numpy as np
from tensorflow.keras.datasets import mnist
from skimage.transform import resize

class DataPreprocessor:
    def load_data(self):
        (train_images, train_labels), (test_images, _) = mnist.load_data()
        return train_images, train_labels, test_images

    def preprocess_images(self, images):
        images_resized = np.array([resize(image, (20, 20)) for image in images]) / 255.0
        images_flattened = images_resized.reshape(images_resized.shape[0], -1)
        return images_flattened
