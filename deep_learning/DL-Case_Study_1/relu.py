import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0

# Model
model = Sequential([
    Input(shape=(784,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)

test_loss, test_acc = model.evaluate(x_test, y_test)
print("ReLU Test Loss:", test_loss)
print("ReLU Test Accuracy:", test_acc)

# Graph 1: Accuracy
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(["Train", "Validation"])
plt.title("ReLU Accuracy")
plt.savefig("plots/relu_accuracy.png")
plt.show()

# Graph 2: Loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(["Train", "Validation"])
plt.title("ReLU Loss")
plt.savefig("plots/relu_loss.png")
plt.show()

# Graph 3: Loss vs Epochs
plt.figure()
plt.plot(history.history['loss'])
plt.title("ReLU Loss vs Epochs")
plt.savefig("plots/relu_loss_epochs.png")
plt.show()
