# Check TensorFlow is installed, supporting GPU usage and list the available devices configured
# GPUs are more favourable and practical than CPUS when utilising TensorFlow for ML

import tensorflow as tf
from tensorflow.python.client import device_lib


def test_environment():
    # Validate the TensorFlow installation and availability of GPU support with CUDA
    try:
        version = tf.__version__
        print(f"TensorFlow v{version} successfully installed.")
        if tf.test.is_built_with_cuda():
            print("The installed version of TensorFlow includes GPU support.")
        else:
            print("The installed version of TensorFlow does not include GPU support.")
    except ImportError:
        print("ERROR: Failed to import the TensorFlow module.")


def test_operation():
    # Test if operations are assigned to GPU
    tf.debugging.set_log_device_placement(True)

    # Create some tensors
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)

    """
    tf.Tensor(
        [[22. 28.]
        [49. 64.]], shape=(2, 2), dtype=float32)
    """
    print(c)


if __name__ == '__main__':
    print(device_lib.list_local_devices())
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    test_environment()
    test_operation()
