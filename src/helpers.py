"""Helper functions for environment configuration and data handling"""

import tensorflow as tf


def configure_gpu(gpu_num=0, memory_growth=True):
    """Configure GPU settings.

    Args:
        gpu_num (int): The GPU device number to use.
        memory_growth (bool): Whether to enable memory growth for the selected GPU.
    """

    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:

            # Restrict to specified GPU only
            tf.config.set_visible_devices(gpus[gpu_num], 'GPU')

            # Set memory growth for the selected GPU
            tf.config.experimental.set_memory_growth(gpus[gpu_num], memory_growth)

            print(f'Using GPU {gpu_num} with memory growth enabled')

        except RuntimeError as e:
            print(f'GPU configuration error: {e}')
    else:
        print('No GPU devices found')


def prepare_tf_dataset(dataset_path, batch_size, shuffle=True, shuffle_buffer_size=5000):
    """Prepare a TensorFlow dataset from a given file path.

    Args:
        dataset_path (str): Path to the dataset file.
        batch_size (int): Batch size for the dataset.
        shuffle (bool): Whether to shuffle the dataset.
        shuffle_buffer_size (int): Buffer size for shuffling the dataset.

    Returns:
        tf.data.Dataset: A TensorFlow dataset ready for training or evaluation.
    """

    # Load finite dataset from disk
    dataset = tf.data.Dataset.load(str(dataset_path))

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=shuffle_buffer_size,
            reshuffle_each_iteration=True # Ensures a new shuffle order every epoch
        )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset