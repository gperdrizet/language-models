"""Learning rate schedules for transformer training."""

import tensorflow as tf


class TransformerSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate schedule with linear warmup and exponential decay.
    
    This creates:
    - Linear warmup from initial_lr to peak_lr over warmup_steps
    - Exponential decay after warmup: peak_lr * (decay_rate ^ steps_after_warmup)
    
    Args:
        initial_lr: Starting learning rate during warmup (default: 1e-6)
        peak_lr: Maximum learning rate reached after warmup (default: 0.01)
        warmup_steps: Number of steps for linear warmup (default: 1404)
        decay_rate: Exponential decay rate per step after warmup (default: 0.99)
    
    Example:
        With initial_lr=1e-6, peak_lr=0.01, warmup_steps=1404, decay_rate=0.99:
        - Step 1: 1e-6 (initial)
        - Step 702 (halfway): ~0.005 (halfway to peak)
        - Step 1404: 0.01 (peak)
        - Step 1404 + N: 0.01 * (0.99^N) (exponential decay)
    """
    
    def __init__(self, initial_lr=1e-6, peak_lr=0.01, warmup_steps=1404, decay_rate=0.99):
        super().__init__()
        
        self.initial_lr = tf.cast(initial_lr, tf.float32)
        self.peak_lr = tf.cast(peak_lr, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.decay_rate = tf.cast(decay_rate, tf.float32)
    
    def __call__(self, step):
        # Cast step to float32
        step = tf.cast(step, tf.float32)
        
        # Linear warmup phase
        warmup_lr = self.initial_lr + (self.peak_lr - self.initial_lr) * (step / self.warmup_steps)
        
        # Exponential decay phase
        steps_after_warmup = step - self.warmup_steps
        decay_lr = self.peak_lr * tf.pow(self.decay_rate, steps_after_warmup)
        
        # Use warmup before warmup_steps, decay after
        lr = tf.where(step < self.warmup_steps, warmup_lr, decay_lr)
        
        return lr
    
    def get_config(self):
        return {
            'initial_lr': float(self.initial_lr.numpy()),
            'peak_lr': float(self.peak_lr.numpy()),
            'warmup_steps': int(self.warmup_steps.numpy()),
            'decay_rate': float(self.decay_rate.numpy())
        }
