import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import time

start_time = time.time()

load_start_time = time.time()
# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
load_end_time = time.time()

print(f'Loading time: {load_end_time - load_start_time} seconds') #Loading time: 10.00931191444397 seconds

# Create a simple neural network
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'), # Substitute with 'Dense(1, activation='relu')' for 33.39% accuracy, compared to 97.75%
    Dense(10, activation='softmax')
])

# An even easier model to replicate in c++ is as follows:
#model = Sequential([
#    Flatten(input_shape=(28, 28)),
#    Dense(10, activation='softmax')
#])
# This achives an accuracy of 92.35% for a total run time of 25.75 seconds

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')

end_time = time.time()
print(f'Total time: {end_time - start_time} seconds')