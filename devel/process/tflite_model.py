# Load the processed and validated model
# Convert fully trained model into a tflite model for C/CPP

import numpy as np
import tensorflow as tf
from train_model import get_data_splits

JSON_PATH = "../preprocess/data.json"  # Pre-processed dataset
MODEL_PATH = "models/validated_model.h5"  # Fully trained Keras model

print("Loading combined data splits..")
# Load data splits
X_train, X_validation, X_test, y_train, y_validation, y_test = get_data_splits(JSON_PATH)
complete_train_X = np.concatenate((X_train, X_validation, X_test))
# Casting to FLOAT32 as tflite models class FLOAT64 as NOTYPE
complete_train_X = complete_train_X.astype(np.float32)


def main():
    print("Loading validated Keras model..")
    # Load final model from training
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Initialising tflite converter..")
    # Initialise tflite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Set optimization flag
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        for i in range(0, len(complete_train_X), 100):
            yield [complete_train_X[i:i + 100]]

    print("Running tflite converter..")
    # Enforce integer-only quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Representative dataset to ensure correct quantization
    converter.representative_dataset = representative_dataset
    tflite_quant_model = converter.convert()

    print("Opening converted tflite model..")
    open("converted_quant_model.tflite", "wb").write(tflite_quant_model)

    print("Finished.")
    exit(0)


if __name__ == "__main__":
    main()
