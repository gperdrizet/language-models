"""Global configuration"""

GLOBAL_CONFIG = {
    'dataset'            : 'Helsinki-NLP/opus-100',      # Hugging Face dataset repo
    'tokenizer'          : 'Helsinki-NLP/opus-mt-en-fr', # Hugging Face tokenizer repo
    'max_seq_length'     : 100,     # Maximum tokens per sentence after tokenization
    'max_encoder_length' : 102,     # Padding length for encoder (allows for special tokens)
    'max_decoder_length' : 104,     # Padding length for decoder (extra space for BOS/EOS)
    'num_train_samples'  : 1000000, # Max training set size (after application of length threshold)
    'test_set_size'      : 1000,    # Number of translation pairs in test set
    'shuffle_buffer_size': 12800    # Buffer size for shuffling the training dataset
}