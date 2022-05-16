# Load the pre-processed dataset and create data splits
# Build our own CNN model then train and evaluate it with the data splits

import json as js
import numpy as np
import tensorflow as tf
from keras import models
from sklearn.model_selection import train_test_split  # Traditional Machine-Learning library
from tensorflow import keras  # Use Keras front-end of TensorFlow for modelling

JSON_PATH = "../preprocess/data.json"  # Pre-processed dataset
MODEL_PATH = "models/model.h5"  # Keras model
TEST_PATH = "models/tested_model.h5"
TRAIN_PATH = "models/validated_model.h5"

BATCH_SIZE = 24  # Number of samples the network will see before updating
DROPOUT = 0.25  # Drops 25% of neurons in the dense layer, forces adaptation
EPOCHS = 50  # The amount of times the network models the dataset for training
LEARNING_RATE = 0.0001  # Optimisation Algorithm - Adam
NUM_KEYWORDS = 22  # Number of mappings in the dataset


def load_dataset(data_path):
    # Load the pre-processed dataset
    with open(data_path, "r") as fp:
        data = js.load(fp)

    # Extract inputs and targets (Pylists to Numpy arrays)
    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])

    return X, y


def get_data_splits(data_path, test_size=0.1, test_validation=0.1):
    # 10% of dataset used for testing purposes
    # 90% of dataset used with X_train and Y_train
    X, y = load_dataset(data_path)

    # Create 'train, validation and test' data splits (2-Dimensional array)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train,
                                                                    test_size=test_validation)

    # Convert inputs from 2-Dimensional to 3-Dimensional array
    X_train = X_train[..., np.newaxis]  # (# segments, 13, 1)
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape, learning_rate=LEARNING_RATE, error="sparse_categorical_crossentropy"):  # SCC
    # Initialise network
    model = keras.Sequential()  # Create a sequential model, convolutional neural network (feed-forward)

    # Conv layer 1
    # (# number of filters, # kernel size, # activation, # input shape of first layer, # overfitting)
    model.add(keras.layers.Conv2D(64, (3, 3), activation="relu",
                                  input_shape=input_shape,
                                  kernel_regularizer=keras.regularizers.l2(0.001)))

    # Normalise, MaxPooling layer 1
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))

    # Conv layer 2
    model.add(keras.layers.Conv2D(32, (3, 3), activation="relu",
                                  kernel_regularizer=keras.regularizers.l2(0.001)))

    # Normalise, MaxPooling layer 2
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))

    # Conv layer 3
    model.add(keras.layers.Conv2D(32, (2, 2), activation="relu",
                                  kernel_regularizer=keras.regularizers.l2(0.001)))

    # Normalise, MaxPooling layer 3
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same"))

    # Flatten output of convolutional layers, forward-feed into dense layer then dropout percentage of neurons
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(DROPOUT))

    # SoftMax classifier
    model.add(keras.layers.Dense(NUM_KEYWORDS, activation="softmax"))  # Prediction score [0.1, 0.7, 0.2]

    # Compile the model by Keras
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=error,
        metrics=["accuracy"]
    )

    # Model statistics overview
    model.summary()
    return model


def main():
    print("Loading data splits..")
    # Load data splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = get_data_splits(JSON_PATH)

    print("Building CNN model..")
    # Build CNN model
    # Equally spaced segments = total amount of sample sets / hop length
    # (#segments, # coefficients 13, # information channel of an image, 1)
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])  # CNN takes a 3-Dimensional input
    model = build_model(input_shape, LEARNING_RATE)

    print("Training initial CNN model..")
    # Train CNN model
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(X_validation, y_validation))

    print("Saving initial CNN model..")
    # Save CNN model
    model.save(MODEL_PATH)

    print("Loading test CNN model..")
    # Load CNN model
    test = models.load_model(MODEL_PATH)

    print("Evaluating test CNN model..")
    # Evaluate to test the CNN model
    results = test.evaluate(X_test, tf.cast(y_test, tf.float32), batch_size=BATCH_SIZE)
    test_error, test_accuracy = test.evaluate(X_test, y_test)
    print(f"Test Error: {test_error}, Test Accuracy: {test_accuracy}")

    print("Saving test CNN model..")
    # Save tested CNN model
    test.save(TEST_PATH)

    print("Loading train CNN model..")
    # Load CNN test model
    train = models.load_model(TEST_PATH)

    print("Combining data splits..")
    # Fully train the CNN test model
    complete_train_X = np.concatenate((X_train, X_validation, X_test))
    complete_train_Y = np.concatenate((y_train, y_validation, y_test))

    print("Shuffling combined data split for data augmentation..")
    complete_train_dataset = tf.data.Dataset.from_tensor_slices((complete_train_X, complete_train_Y))\
        .repeat(count=-1)\
        .shuffle(100000).batch(BATCH_SIZE)

    print("Fully training the train CNN model..")
    history = train.fit(
        complete_train_dataset,
        steps_per_epoch=len(complete_train_X) // BATCH_SIZE,
        epochs=10
    )

    print("Fully trained model evaluation..")
    # Evaluate to test the trained CNN model
    results = train.evaluate(X_test, tf.cast(y_test, tf.float32), batch_size=BATCH_SIZE)
    test_error, test_accuracy = train.evaluate(X_test, y_test)
    print(f"Test Error: {test_error}, Test Accuracy: {test_accuracy}")

    print("Fully trained CNN model..")
    # Save fully-trained CNN model
    train.save(TRAIN_PATH)

    print("Finished.")
    exit(0)


if __name__ == "__main__":
    main()
