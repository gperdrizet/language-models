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