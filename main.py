from data_preprocessor import DataPreprocessor
from perceptron import Perceptron
from image_utils import predict_digit
import os

# Load and preprocess data
data_preprocessor = DataPreprocessor()
train_images, train_labels, test_images = data_preprocessor.load_data()
train_images_flattened = data_preprocessor.preprocess_images(train_images)
test_images_flattened = data_preprocessor.preprocess_images(test_images)

# Train the Perceptron
num_classes = 10
perceptron = Perceptron(input_size=400, output_size=num_classes)
perceptron.train(train_images_flattened, train_labels, epochs=100, learning_rate=0.01)

# Predict digits from images
script_directory = os.path.dirname(os.path.realpath(__file__))

image_filenames = [
    'Nine.jpg',
    'One.jpg',
    'Three.jpg'
]

image_paths = [os.path.join(script_directory, filename) for filename in image_filenames]

for image_path in image_paths:
    predicted_digit = predict_digit(perceptron, image_path)
    print(f"The predicted digit for image {image_path} is: {predicted_digit}")
