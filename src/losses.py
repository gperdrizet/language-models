"""
Custom loss functions for sequence-to-sequence models.
"""

import tensorflow as tf


def masked_sparse_categorical_crossentropy(y_true, y_pred, pad_token_id):
    mask = tf.cast(tf.not_equal(y_true, pad_token_id), tf.float32)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)
    
    masked_loss = loss * mask
    return tf.reduce_sum(masked_loss) / tf.maximum(tf.reduce_sum(mask), 1e-9)


def masked_accuracy(y_true, y_pred, pad_token_id):
    mask = tf.cast(tf.not_equal(y_true, pad_token_id), tf.float32)
    predictions = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    matches = tf.cast(tf.equal(predictions, tf.cast(y_true, tf.int32)), tf.float32)
    
    masked_matches = matches * mask
    return tf.reduce_sum(masked_matches) / tf.maximum(tf.reduce_sum(mask), 1e-9)