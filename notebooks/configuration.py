"""Global configuration"""

import sys
from pathlib import Path
import tensorflow as tf

# Paths
PROJECT_ROOT = project_root = Path('..').resolve()
sys.path.insert(0, str(project_root))

MODEL_DIRECTORY    = PROJECT_ROOT / 'models'
DATA_DIRECTORY     = PROJECT_ROOT / 'data'
TRAIN_DATASET_PATH = DATA_DIRECTORY / 'training_dataset'
VAL_DATASET_PATH   = DATA_DIRECTORY / 'validation_dataset'
TEST_DATASET_PATH  = DATA_DIRECTORY / 'test_dataset.pkl'

# Global configuration dictionary
GLOBAL_CONFIG = {
    'dataset'            : 'Helsinki-NLP/opus-100',      # Hugging Face dataset repo
    'tokenizer'          : 'Helsinki-NLP/opus-mt-en-fr', # Hugging Face tokenizer repo
    'num_tokens'         : 59514,   # Vocabulary size for the tokenizer
    'max_seq_length'     : 100,     # Maximum tokens per sentence after tokenization
    'max_encoder_length' : 101,     # Padding length for encoder (extra space for EOS)
    'max_decoder_length' : 101,     # Padding length for decoder (extra space for BOS/EOS)
    'num_train_samples'  : 1000000, # Max training set size (after application of length threshold)
    'validation_split'   : 0.1,     # Fraction of training data to use for validation
    'test_set_size'      : 1000,    # Number of translation pairs in test set
}

# Output signature for TensorFlow datasets
TF_OUTPUT_SIGNATURE = (
    {
        'encoder_input': tf.TensorSpec(shape=(GLOBAL_CONFIG['max_encoder_length'],), dtype=tf.int32),
        'decoder_input': tf.TensorSpec(shape=(GLOBAL_CONFIG['max_decoder_length'],), dtype=tf.int32)
    },
    tf.TensorSpec(shape=(GLOBAL_CONFIG['max_decoder_length'],), dtype=tf.int32)
)