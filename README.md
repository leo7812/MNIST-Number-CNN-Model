# MNIST Handwritten Digit Recognizer
This project contains a Python script that builds, trains, and evaluates a Convolutional Neural Network (CNN) to recognize handwritten digits (0-9) using computer webcam. After training, the script can take a user-provided image of a digit and predict what number it is.

How It Works
The script performs the following steps:

Loads Data: Loads the MNIST dataset from downloaded local CSV files (mnist_train.csv and mnist_test.csv).

Builds Model: Constructs a CNN using TensorFlow and Keras. The model is designed to learn the features of handwritten digits.

Trains Model: Trains the model on 60,000 images of handwritten digits and validates it on a separate set of 10,000 images.

Evaluates Performance: After training, it calculates the final accuracy on the test dataset and displays a plot of the training history.

Predicts Custom Image: It takes an image file named number.png, processes it to match the MNIST format, and uses the trained model to predict the digit.

Model Architecture
The neural network is a sequential model with the following layers:

Conv2D (32 filters, ReLU activation)

MaxPooling2D

Dropout (rate of 0.25 to prevent overfitting)

Conv2D (64 filters, ReLU activation)

MaxPooling2D

Dropout (rate of 0.25)

Flatten

Dense (128 neurons, ReLU activation)

Dropout (rate of 0.5)

Dense (10 output neurons, Softmax activation for classification)

Training Results
The model is trained for 2-3 epochs using the Adam optimizer and sparse_categorical_crossentropy as the loss function. It consistently achieves a high accuracy ( >98%) on the test dataset.
