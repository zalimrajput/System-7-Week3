
import numpy as np

# 1. AND gate dataset
# Two inputs and their expected outputs
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])

# 2. Initialize weights and bias
weights = np.zeros(2)
bias = 0.0

# 3. Set learning rate and epochs
learning_rate = 0.1
epochs = 20

# 4. Step activation function
def activation_function(value):
    return 1 if value >= 0 else 0

# 5. Train the perceptron
for epoch in range(epochs):

    total_errors = 0

    for inputs, expected_output in zip(X, y):

        # Forward propagation
        weighted_sum = np.dot(inputs, weights) + bias

        prediction = activation_function(weighted_sum)

        # Calculate error
        error = expected_output - prediction

        # Update weights and bias
        weights += learning_rate * error * inputs
        bias += learning_rate * error

        if error != 0:
            total_errors += 1

    print(f"Epoch {epoch + 1}: Errors = {total_errors}")

    if total_errors == 0:
        break

# 6. Test the trained perceptron
print("\nFinal Weights:", weights)
print("Final Bias:", bias)

print("\nAND Gate Predictions:")

for inputs in X:
    weighted_sum = np.dot(inputs, weights) + bias
    prediction = activation_function(weighted_sum)

    print(f"Input: {inputs} -> Prediction: {prediction}")
