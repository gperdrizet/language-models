"""
Custom loss functions for sequence-to-sequence models.
"""

import tensorflow as tf


def masked_sparse_categorical_crossentropy(y_true, y_pred, pad_token_id=59513):
    """
    Sparse categorical crossentropy that masks padding positions.
    
    Args:
        y_true: Ground truth labels of shape (batch, seq_len)
        y_pred: Model predictions (logits) of shape (batch, seq_len, vocab_size)
        pad_token_id: ID of padding token to mask (default: 59513 for MarianTokenizer)
    
    Returns:
        Scalar loss value (averaged over non-padding positions)
    """
    # Create mask: True for real tokens, False for padding
    mask = tf.cast(tf.not_equal(y_true, pad_token_id), tf.float32)
    
    # Compute loss per position
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction='none'  # Don't reduce yet, need to apply mask first
    )
    loss = loss_fn(y_true, y_pred)
    
    # Apply mask
    masked_loss = loss * mask
    
    # Average over non-padding positions only
    return tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)


def masked_accuracy(y_true, y_pred, pad_token_id=59513):
    """
    Accuracy metric that masks padding positions.
    
    Args:
        y_true: Ground truth labels of shape (batch, seq_len)
        y_pred: Model predictions (logits) of shape (batch, seq_len, vocab_size)
        pad_token_id: ID of padding token to mask (default: 59513 for MarianTokenizer)
    
    Returns:
        Scalar accuracy value (averaged over non-padding positions)
    """
    # Create mask: True for real tokens, False for padding
    mask = tf.cast(tf.not_equal(y_true, pad_token_id), tf.float32)
    
    # Get predicted token IDs
    predictions = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    
    # Compare with ground truth
    matches = tf.cast(tf.equal(predictions, tf.cast(y_true, tf.int32)), tf.float32)
    
    # Apply mask
    masked_matches = matches * mask
    
    # Average over non-padding positions only
    return tf.reduce_sum(masked_matches) / tf.reduce_sum(mask)
