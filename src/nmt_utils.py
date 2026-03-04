"""
Neural machine translation model utilities.

This module provides model building, inference, and evaluation utilities
for encoder-decoder neural machine translation models.
"""

import numpy as np
import tensorflow as tf
from sacrebleu.metrics import BLEU
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Embedding, Bidirectional, Concatenate, Attention
)


def build_bidirectional_model(num_tokens, max_encoder_len, max_decoder_len, latent_dim=256):
    """
    Build encoder-decoder with bidirectional encoder.
    
    Args:
        num_tokens: Vocabulary size
        max_encoder_len: Maximum encoder sequence length
        max_decoder_len: Maximum decoder sequence length
        latent_dim: Latent dimension for LSTM layers
    
    Returns:
        Compiled Keras model
    """
    # ── Encoder ──────────────────────────────────────────────
    encoder_inputs = Input(shape=(max_encoder_len,), name='encoder_input')
    encoder_embedding = Embedding(num_tokens, latent_dim, mask_zero=True, name='encoder_embedding')
    encoder_embedded = encoder_embedding(encoder_inputs)
    
    # Bidirectional LSTM encoder
    encoder_lstm = Bidirectional(
        LSTM(latent_dim, return_state=True, name='encoder_lstm'),
        name='bidirectional_encoder'
    )
    
    # Get outputs and states from bidirectional LSTM
    # Returns: outputs, forward_h, forward_c, backward_h, backward_c
    outputs, fwd_h, fwd_c, bwd_h, bwd_c = encoder_lstm(encoder_embedded)
    
    # Concatenate forward and backward states
    state_h = Concatenate(name='concat_h')([fwd_h, bwd_h])
    state_c = Concatenate(name='concat_c')([fwd_c, bwd_c])
    encoder_states = [state_h, state_c]
    
    # ── Decoder ──────────────────────────────────────────────
    decoder_inputs = Input(shape=(max_decoder_len,), name='decoder_input')
    decoder_embedding = Embedding(num_tokens, latent_dim, mask_zero=True, name='decoder_embedding')
    decoder_embedded = decoder_embedding(decoder_inputs)
    decoder_lstm = LSTM(latent_dim * 2, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(decoder_embedded, initial_state=encoder_states)
    decoder_dense = Dense(num_tokens, activation='softmax', name='output')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_attention_model(num_tokens, max_encoder_len, max_decoder_len, latent_dim=256):
    """
    Build encoder-decoder with attention mechanism.
    
    Args:
        num_tokens: Vocabulary size
        max_encoder_len: Maximum encoder sequence length
        max_decoder_len: Maximum decoder sequence length
        latent_dim: Latent dimension for LSTM layers
    
    Returns:
        Compiled Keras model
    """
    # ── Encoder ──────────────────────────────────────────────
    encoder_inputs = Input(shape=(max_encoder_len,), name='encoder_input')
    encoder_embedding = Embedding(num_tokens, latent_dim, mask_zero=True, name='encoder_embedding')
    encoder_embedded = encoder_embedding(encoder_inputs)
    
    # Bidirectional LSTM encoder - return_sequences=True for attention
    encoder_lstm = Bidirectional(
        LSTM(latent_dim, return_sequences=True, return_state=True, name='encoder_lstm'),
        name='bidirectional_encoder'
    )
    
    # Get outputs (all timesteps) and states
    encoder_outputs, fwd_h, fwd_c, bwd_h, bwd_c = encoder_lstm(encoder_embedded)
    
    # Concatenate forward and backward states for decoder initialization
    state_h = Concatenate(name='concat_h')([fwd_h, bwd_h])
    state_c = Concatenate(name='concat_c')([fwd_c, bwd_c])
    encoder_states = [state_h, state_c]
    
    # ── Decoder ──────────────────────────────────────────────
    decoder_inputs = Input(shape=(max_decoder_len,), name='decoder_input')
    decoder_embedding = Embedding(num_tokens, latent_dim, mask_zero=True, name='decoder_embedding')
    decoder_embedded = decoder_embedding(decoder_inputs)
    
    # Decoder LSTM (latent_dim * 2 because bidirectional encoder)
    decoder_lstm = LSTM(latent_dim * 2, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(decoder_embedded, initial_state=encoder_states)
    
    # ── Attention ────────────────────────────────────────────
    # Keras Attention layer implements Luong-style (dot-product) attention
    # Query: decoder outputs, Key/Value: encoder outputs
    attention_layer = Attention(name='attention')
    context = attention_layer([decoder_outputs, encoder_outputs])
    
    # Concatenate attention context with decoder outputs
    decoder_combined = Concatenate(name='concat_attention')([context, decoder_outputs])
    
    # Output layer
    decoder_dense = Dense(num_tokens, activation='softmax', name='output')
    decoder_outputs = decoder_dense(decoder_combined)
    
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_inference_models_lstm(model, latent_dim=256):
    """
    Build separate encoder and decoder models for LSTM inference.
    
    Args:
        model: Trained bidirectional LSTM model
        latent_dim: Latent dimension used in training
    
    Returns:
        Tuple of (encoder_model, decoder_model)
    """
    # ── Encoder inference model ──────────────────────────────────
    encoder_input_layer = model.get_layer('encoder_input').input
    encoder_embedding_layer = model.get_layer('encoder_embedding')
    bidirectional_layer = model.get_layer('bidirectional_encoder')
    concat_h = model.get_layer('concat_h')
    concat_c = model.get_layer('concat_c')

    # Reconstruct encoder: input -> embedding -> BiLSTM -> concatenated states
    encoder_embedded = encoder_embedding_layer(encoder_input_layer)
    _, fwd_h, fwd_c, bwd_h, bwd_c = bidirectional_layer(encoder_embedded)
    state_h = concat_h([fwd_h, bwd_h])
    state_c = concat_c([fwd_c, bwd_c])
    
    # Encoder model: takes input sequence, outputs initial decoder states
    encoder_model = Model(encoder_input_layer, [state_h, state_c])

    # ── Decoder inference model ──────────────────────────────────
    decoder_state_input_h = Input(shape=(latent_dim * 2,), name='decoder_state_h')
    decoder_state_input_c = Input(shape=(latent_dim * 2,), name='decoder_state_c')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    # Single token input (shape=(1,) for one token at a time)
    decoder_input_layer = Input(shape=(1,), name='decoder_inf_input')
    
    # Reuse trained decoder layers
    decoder_embedding_layer = model.get_layer('decoder_embedding')
    decoder_lstm_layer = model.get_layer('decoder_lstm')
    decoder_dense_layer = model.get_layer('output')

    # Reconstruct decoder: token -> embedding -> LSTM -> softmax
    decoder_embedded = decoder_embedding_layer(decoder_input_layer)
    decoder_outputs, state_h, state_c = decoder_lstm_layer(
        decoder_embedded, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense_layer(decoder_outputs)

    # Decoder model: takes token + states, outputs probabilities + new states
    decoder_model = Model(
        [decoder_input_layer] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
    )
    
    return encoder_model, decoder_model


def build_inference_models_attention(model, max_encoder_len, latent_dim=256):
    """
    Build separate encoder and decoder models for attention-based inference.
    
    Args:
        model: Trained attention model
        max_encoder_len: Maximum encoder sequence length
        latent_dim: Latent dimension used in training
    
    Returns:
        Tuple of (encoder_model, decoder_model)
    """
    # ── Encoder inference model ────────────────────────────────────────────
    encoder_input_layer = model.get_layer('encoder_input').input
    encoder_embedding_layer = model.get_layer('encoder_embedding')
    bidirectional_layer = model.get_layer('bidirectional_encoder')
    concat_h = model.get_layer('concat_h')
    concat_c = model.get_layer('concat_c')

    # Reconstruct encoder
    encoder_embedded = encoder_embedding_layer(encoder_input_layer)
    encoder_outputs, fwd_h, fwd_c, bwd_h, bwd_c = bidirectional_layer(encoder_embedded)
    state_h = concat_h([fwd_h, bwd_h])
    state_c = concat_c([fwd_c, bwd_c])
    
    # Encoder model now returns encoder_outputs (for attention) AND initial states
    encoder_model = Model(encoder_input_layer, [encoder_outputs, state_h, state_c])

    # ── Decoder inference model ────────────────────────────────────────────
    decoder_input_layer = Input(shape=(1,), name='decoder_inf_input')
    decoder_state_input_h = Input(shape=(latent_dim * 2,), name='decoder_state_h')
    decoder_state_input_c = Input(shape=(latent_dim * 2,), name='decoder_state_c')
    encoder_outputs_input = Input(shape=(max_encoder_len, latent_dim * 2), name='encoder_outputs_input')
    
    # Reuse trained layers
    decoder_embedding_layer = model.get_layer('decoder_embedding')
    decoder_lstm_layer = model.get_layer('decoder_lstm')
    attention_layer = model.get_layer('attention')
    concat_attention_layer = model.get_layer('concat_attention')
    decoder_dense_layer = model.get_layer('output')

    # Reconstruct decoder with attention
    decoder_embedded = decoder_embedding_layer(decoder_input_layer)
    decoder_outputs, state_h, state_c = decoder_lstm_layer(
        decoder_embedded, initial_state=[decoder_state_input_h, decoder_state_input_c]
    )
    
    # Apply attention using encoder outputs
    context = attention_layer([decoder_outputs, encoder_outputs_input])
    decoder_combined = concat_attention_layer([context, decoder_outputs])
    decoder_outputs = decoder_dense_layer(decoder_combined)

    # Decoder model: takes token + states + encoder_outputs, returns probs + new states
    decoder_model = Model(
        [decoder_input_layer, decoder_state_input_h, decoder_state_input_c, encoder_outputs_input],
        [decoder_outputs, state_h, state_c]
    )
    
    return encoder_model, decoder_model


def translate_lstm(input_text, encoder_model, decoder_model, tokenizer, max_encoder_len, max_decoder_len):
    """
    Translate text using greedy decoding with LSTM model.
    
    Args:
        input_text: Source text to translate
        encoder_model: Encoder inference model
        decoder_model: Decoder inference model
        tokenizer: Tokenizer for encoding/decoding
        max_encoder_len: Maximum encoder sequence length
        max_decoder_len: Maximum decoder sequence length
    
    Returns:
        Translated text
    """
    # Tokenize and pad the input sentence
    input_tokens = tokenizer(
        input_text,
        padding='max_length',
        max_length=max_encoder_len,
        truncation=True,
        return_tensors='np'
    )['input_ids']
    
    # Encode: run encoder once to get initial states
    states = encoder_model.predict(input_tokens, verbose=0)
    
    # Start with pad token (acts as BOS for this tokenizer)
    target_seq = np.array([[tokenizer.pad_token_id]])
    decoded_tokens = []

    # Autoregressive decoding loop
    for _ in range(max_decoder_len):

        # Get next token probabilities and updated states
        output_tokens, h, c = decoder_model.predict([target_seq] + states, verbose=0)
        
        # Greedy selection: pick highest probability token
        sampled_token_id = np.argmax(output_tokens[0, -1, :])
        
        # Stop if end-of-sequence token
        if sampled_token_id == tokenizer.eos_token_id:
            break
            
        # Append token and prepare for next iteration
        decoded_tokens.append(sampled_token_id)
        target_seq = np.array([[sampled_token_id]])
        states = [h, c]
    
    # Decode token IDs back to text
    return tokenizer.decode(decoded_tokens, skip_special_tokens=True)


def translate_attention(input_text, encoder_model, decoder_model, tokenizer, max_encoder_len, max_decoder_len):
    """
    Translate text using greedy decoding with attention model.
    
    Args:
        input_text: Source text to translate
        encoder_model: Encoder inference model
        decoder_model: Decoder inference model
        tokenizer: Tokenizer for encoding/decoding
        max_encoder_len: Maximum encoder sequence length
        max_decoder_len: Maximum decoder sequence length
    
    Returns:
        Translated text
    """
    # Tokenize and pad the input sentence
    input_tokens = tokenizer(
        input_text,
        padding='max_length',
        max_length=max_encoder_len,
        truncation=True,
        return_tensors='np'
    )['input_ids']
    
    # Encode: get encoder outputs (for attention) and initial states
    encoder_outputs, state_h, state_c = encoder_model.predict(input_tokens, verbose=0)
    states = [state_h, state_c]
    
    # Start with pad token (acts as BOS for this tokenizer)
    target_seq = np.array([[tokenizer.pad_token_id]])
    decoded_tokens = []

    # Autoregressive decoding loop
    for _ in range(max_decoder_len):

        # Get next token probabilities and updated states
        # Note: encoder_outputs is passed every step for attention
        output_tokens, h, c = decoder_model.predict(
            [target_seq, states[0], states[1], encoder_outputs], verbose=0
        )
        
        # Greedy selection: pick highest probability token
        sampled_token_id = np.argmax(output_tokens[0, -1, :])
        
        # Stop if end-of-sequence token
        if sampled_token_id == tokenizer.eos_token_id:
            break
            
        # Append token and prepare for next iteration
        decoded_tokens.append(sampled_token_id)
        target_seq = np.array([[sampled_token_id]])
        states = [h, c]
    
    # Decode token IDs back to text
    return tokenizer.decode(decoded_tokens, skip_special_tokens=True)


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
        self.sample_size = min(sample_size, len(pairs))
        self.latent_dim = latent_dim
        self.restore_best_weights = restore_best_weights
        
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
        
        # Checkpoint if this is the best BLEU score so far
        if score > self.best_bleu:

            self.best_bleu = score
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch
            print(f' - BLEU: {score:.2f} (best)')

        else:
            print(f' - BLEU: {score:.2f} (best: {self.best_bleu:.2f})')
    
    def on_train_end(self, logs=None):
        """Restore best weights after training completes."""

        if self.restore_best_weights and self.best_weights is not None:

            print(f'Restoring best model weights from epoch {self.best_epoch + 1} (BLEU: {self.best_bleu:.2f})')
            self.model.set_weights(self.best_weights)
