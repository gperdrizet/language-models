"""Learning rate schedules for transformer training."""

import tensorflow as tf


class TransformerSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate schedule with linear warmup and cosine annealing decay.
    
    This creates:
    - Linear warmup from initial_lr to peak_lr over warmup_steps
    - Cosine annealing decay from peak_lr to min_lr over remaining steps
    
    The cosine decay keeps the learning rate high for longer and smoothly
    decreases to min_lr by the end of training, avoiding premature decay.
    
    Args:
        initial_lr: Starting learning rate during warmup (default: 1e-6)
        peak_lr: Maximum learning rate reached after warmup (default: 0.01)
        min_lr: Minimum learning rate at end of training (default: 1e-7)
        warmup_steps: Number of steps for linear warmup (default: 1404)
        total_steps: Total training steps including warmup (default: 35100)
    
    Example:
        With initial_lr=1e-9, peak_lr=0.0003, min_lr=1e-7, warmup_steps=7032, total_steps=175800:
        - Step 0-7032: Linear warmup from 1e-9 to 0.0003
        - Step 7032-175800: Cosine decay from 0.0003 to 1e-7
        - At 50% through decay: LR ≈ 0.00015 (still high)
        - At 75% through decay: LR ≈ 0.00005 (gradual decrease)
    """
    
    def __init__(self, initial_lr=1e-6, peak_lr=0.01, min_lr=1e-7, warmup_steps=1404, total_steps=35100):
        super().__init__()
        
        self.initial_lr = tf.cast(initial_lr, tf.float32)
        self.peak_lr = tf.cast(peak_lr, tf.float32)
        self.min_lr = tf.cast(min_lr, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.total_steps = tf.cast(total_steps, tf.float32)
        self.decay_steps = self.total_steps - self.warmup_steps
    
    def __call__(self, step):
        # Cast step to float32
        step = tf.cast(step, tf.float32)
        
        # Linear warmup phase
        warmup_lr = self.initial_lr + (self.peak_lr - self.initial_lr) * (step / self.warmup_steps)
        
        # Cosine annealing decay phase
        steps_after_warmup = step - self.warmup_steps
        progress = tf.minimum(steps_after_warmup / self.decay_steps, 1.0)  # Clamp to [0, 1]
        cosine_decay = 0.5 * (1.0 + tf.cos(tf.constant(3.14159265359) * progress))
        decay_lr = self.min_lr + (self.peak_lr - self.min_lr) * cosine_decay
        
        # Use warmup before warmup_steps, cosine decay after
        lr = tf.where(step < self.warmup_steps, warmup_lr, decay_lr)
        
        return lr
    
    def get_config(self):
        return {
            'initial_lr': float(self.initial_lr.numpy()),
            'peak_lr': float(self.peak_lr.numpy()),
            'min_lr': float(self.min_lr.numpy()),
            'warmup_steps': int(self.warmup_steps.numpy()),
            'total_steps': int(self.total_steps.numpy())
        }
