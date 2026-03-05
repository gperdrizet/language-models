"""
Training callbacks for neural machine translation models.

This module provides custom Keras callbacks for monitoring and improving
model training, including BLEU score evaluation.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from sacrebleu.metrics import BLEU


class BLEUCallback(tf.keras.callbacks.Callback):
    """
    Callback to compute BLEU score and checkpoint best model.
    
    Builds inference models from current training weights, generates translations,
    and computes BLEU score on a sample of the training data. Restores best weights
    at the end of training.
    """
    
    def __init__(
        self,
        pairs,
        tokenizer,
        max_encoder_len,
        max_decoder_len,
        translate_fn,
        build_inference_fn,
        checkpoint_dir=None,
        sample_size=100,
        latent_dim=256,
        restore_best_weights=True
    ):
        """
        Initialize BLEU callback.
        
        Args:
            pairs: List of (source, target) translation pairs
            tokenizer: Tokenizer for encoding/decoding
            max_encoder_len: Maximum encoder sequence length
            max_decoder_len: Maximum decoder sequence length
            translate_fn: Translation function (translate_lstm or translate_attention)
            build_inference_fn: Function to build inference models
            checkpoint_dir: Directory to save checkpoints (e.g., '../models/lstm/checkpoints')
            sample_size: Number of pairs to evaluate per epoch
            latent_dim: Latent dimension used in model
            restore_best_weights: Whether to restore best weights after training
        """
        super().__init__()
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len
        self.translate_fn = translate_fn
        self.build_inference_fn = build_inference_fn
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.sample_size = min(sample_size, len(pairs))
        self.latent_dim = latent_dim
        self.restore_best_weights = restore_best_weights
        
        # Create checkpoint directory if specified
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Track BLEU scores and best model weights
        self.bleu_scores = []
        self.bleu = BLEU()
        self.best_bleu = 0.0
        self.best_weights = None
        self.best_epoch = 0
        
        # Fixed sample for consistent evaluation across epochs
        np.random.seed(315)
        self.sample_indices = np.random.choice(len(pairs), size=self.sample_size, replace=False)
        
    def on_epoch_end(self, epoch, logs=None):
        """Evaluate BLEU score at the end of each epoch."""

        # Build inference models to generate translations with current weights
        encoder_model, decoder_model = self.build_inference_fn(self.model, self.latent_dim)
        
        # Translate sample sentences and collect references
        hypotheses = []
        references = []

        for idx in self.sample_indices:

            en_text, fr_ref = self.pairs[idx]
            fr_hyp = self.translate_fn(
                en_text, encoder_model, decoder_model,
                self.tokenizer, self.max_encoder_len, self.max_decoder_len
            )
            hypotheses.append(fr_hyp)
            references.append(fr_ref)
        
        # Compute corpus BLEU
        result = self.bleu.corpus_score(hypotheses, [references])
        score = result.score
        self.bleu_scores.append(score)
        
        # Log BLEU score to TensorBoard
        if logs is not None:
            logs['bleu_score'] = score
        
        # Checkpoint if this is the best BLEU score so far
        if score > self.best_bleu:

            self.best_bleu = score
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch
            
            # Save checkpoint if directory specified
            if self.checkpoint_dir:
                checkpoint_filename = f'model_epoch_{epoch+1:02d}_best_bleu_{score:.2f}.h5'
                checkpoint_path = self.checkpoint_dir / checkpoint_filename
                self.model.save_weights(str(checkpoint_path))
                # print(f' - BLEU: {score:.2f} (best) - saved {checkpoint_filename}')
            # else:
            #     print(f' - BLEU: {score:.2f} (best)')

        # else:
        #     print(f' - BLEU: {score:.2f} (best: {self.best_bleu:.2f})')
    
    def on_train_end(self, logs=None):
        """Restore best weights after training completes."""

        if self.restore_best_weights and self.best_weights is not None:

            print(f'Restoring best model weights from epoch {self.best_epoch + 1} (BLEU: {self.best_bleu:.2f})')
            self.model.set_weights(self.best_weights)
