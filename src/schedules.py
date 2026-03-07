"""Learning rate schedules for transformer training."""

import tensorflow as tf


class TransformerSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate schedule from 'Attention is All You Need' (Vaswani et al. 2017).
    
    Formula: lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    
    This creates:
    - Linear warmup for first warmup_steps (usually 4000)
    - Inverse square root decay after warmup
    
    Args:
        d_model: Model dimension (used to scale learning rate)
        warmup_steps: Number of warmup steps (default: 4000)
    
    Example:
        With d_model=256 and warmup_steps=4000:
        - Step 1: ~0.000004 (very small)
        - Step 4000: ~0.0006 (peak)
        - Step 16000: ~0.0003 (half of peak)
    """
    
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
    
    def __call__(self, step):
        # Cast step to float32
        step = tf.cast(step, tf.float32)
        
        # Avoid division by zero at step 0
        step = tf.maximum(step, 1.0)
        
        # Compute both terms
        arg1 = tf.math.rsqrt(step)  # step^(-0.5)
        arg2 = step * (self.warmup_steps ** -1.5)
        
        # Learning rate formula
        lr = tf.math.rsqrt(self.d_model) * tf.minimum(arg1, arg2)
        
        return lr
    
    def get_config(self):
        return {
            'd_model': int(self.d_model.numpy()),
            'warmup_steps': int(self.warmup_steps.numpy())
        }
