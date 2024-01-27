import numpy as np

class Perceptron:
    def __init__(self, input_size, output_size):
        # Initialize weights and bias
        self.weights = np.random.randn(input_size, output_size) * 0.01  # Small random values
        self.bias = np.zeros((1, output_size))

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def cross_entropy_loss(self, predicted, actual):
        m = actual.shape[0]
        log_likelihood = -np.log(predicted[range(m), actual])
        loss = np.sum(log_likelihood) / m
        return loss

    def forward(self, inputs):
        # Linear combination
        z = np.dot(inputs, self.weights) + self.bias
        # Activation
        return self.softmax(z)

    def backpropagation(self, inputs, actual, learning_rate):
        # Forward pass
        predicted = self.forward(inputs)

        # Calculate error
        error = predicted
        error[range(len(actual)), actual] -= 1  # derivative of cross-entropy with softmax

        # Compute gradients
        dweights = np.dot(inputs.T, error) / inputs.shape[0]
        dbias = np.mean(error, axis=0, keepdims=True)

        # Update weights and bias
        self.weights -= learning_rate * dweights
        self.bias -= learning_rate * dbias

    def train(self, inputs, labels, epochs, learning_rate):
        for epoch in range(epochs):
            self.backpropagation(inputs, labels, learning_rate)
            if epoch % 10 == 0:  # Print loss every 10 epochs
                loss = self.cross_entropy_loss(self.forward(inputs), labels)
                print(f'Epoch {epoch}, Loss: {loss}')


