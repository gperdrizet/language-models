"""
Neural machine translation model building and inference utilities.

This module provides functions for building encoder-decoder architectures,
creating inference models, and translating text.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Embedding, Bidirectional, Concatenate, Attention
)

# Import masked loss/accuracy to ignore padding positions
from .losses import masked_sparse_categorical_crossentropy, masked_accuracy
from .schedules import TransformerSchedule

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


# ── Transformer model ────────────────────────────────────────


def get_positional_encoding(seq_len, d_model):

    positions = tf.cast(tf.range(seq_len), tf.float32)[:, tf.newaxis]
    dims = tf.cast(tf.range(d_model), tf.float32)[tf.newaxis, :]
    
    angle_rates = 1.0 / tf.pow(10000.0, (2.0 * (dims // 2.0)) / tf.cast(d_model, tf.float32))
    angle_rads = positions * angle_rates
    
    indices = tf.range(d_model)
    is_even = tf.equal(indices % 2, 0)
    is_even = tf.broadcast_to(is_even[tf.newaxis, :], tf.shape(angle_rads))
    
    pos_encoding = tf.where(is_even, tf.sin(angle_rads), tf.cos(angle_rads))

    return pos_encoding


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        pos_enc = get_positional_encoding(max_len, d_model)
        # Register as non-trainable weight to preserve GPU device placement
        self.pos_encoding = self.add_weight(
            name='pos_encoding',
            shape=(max_len, d_model),
            initializer=tf.keras.initializers.Constant(pos_enc),
            trainable=False
        )

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:seq_len, :]


def feed_forward_network(d_model, d_ff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(d_ff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model // num_heads, 
            dropout=dropout_rate
        )
        self.ffn = feed_forward_network(d_model, d_ff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=False, mask=None):
        attn_input = self.layernorm1(x)
        attn_output = self.mha(
            query=attn_input, 
            value=attn_input, 
            key=attn_input, 
            attention_mask=mask, 
            training=training
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = x + attn_output

        ffn_input = self.layernorm2(out1)
        ffn_output = self.ffn(ffn_input)
        ffn_output = self.dropout2(ffn_output, training=training)
        return out1 + ffn_output


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)

        self.self_mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model // num_heads, 
            dropout=dropout_rate
        )

        self.cross_mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model // num_heads, 
            dropout=dropout_rate
        )

        self.ffn = feed_forward_network(d_model, d_ff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training=False, dec_padding_mask=None, enc_padding_mask=None):
        attn1_input = self.layernorm1(x)
        attn1 = self.self_mha(
            query=attn1_input, 
            value=attn1_input, 
            key=attn1_input, 
            attention_mask=dec_padding_mask, 
            use_causal_mask=True, 
            training=training
        )
        attn1 = self.dropout1(attn1, training=training)
        out1 = x + attn1

        attn2_input = self.layernorm2(out1)
        attn2 = self.cross_mha(
            query=attn2_input, 
            value=enc_output, 
            key=enc_output, 
            attention_mask=enc_padding_mask, 
            training=training
        )
        attn2 = self.dropout2(attn2, training=training)
        out2 = out1 + attn2

        ffn_input = self.layernorm3(out2)
        ffn_output = self.ffn(ffn_input)
        ffn_output = self.dropout3(ffn_output, training=training)
        return out2 + ffn_output


class Encoder(tf.keras.layers.Layer):
    def __init__(self, n_layers, d_model, num_heads, d_ff, vocab_size, max_len, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_len, d_model)
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(n_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.final_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=False, mask=None):
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)

        for enc_layer in self.enc_layers:
            x = enc_layer(x, training=training, mask=mask)

        return self.final_layernorm(x)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, n_layers, d_model, num_heads, d_ff, vocab_size, max_len, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_len, d_model)
        self.dec_layers = [
            DecoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(n_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.final_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, enc_output, training=False, dec_padding_mask=None, enc_padding_mask=None):
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)

        for dec_layer in self.dec_layers:
            x = dec_layer(x, enc_output, training=training, dec_padding_mask=dec_padding_mask, enc_padding_mask=enc_padding_mask)

        return self.final_layernorm(x)


def create_padding_mask(seq, pad_token_id):
    # Shape: (batch, 1, 1, seq_len) for broadcast in MultiHeadAttention
    mask = tf.cast(tf.math.not_equal(seq, pad_token_id), tf.bool)

    return mask[:, tf.newaxis, tf.newaxis, :]


def create_decoder_padding_mask(seq, pad_token_id):
    seq_len = tf.shape(seq)[1]
    positions = tf.range(seq_len)[tf.newaxis, :]
    positions = tf.broadcast_to(positions, tf.shape(seq))
    
    is_not_pad = tf.not_equal(seq, pad_token_id)
    is_position_zero = tf.equal(positions, 0)
    mask = tf.logical_or(is_not_pad, is_position_zero)

    return mask[:, tf.newaxis, tf.newaxis, :]


class Transformer(tf.keras.Model):
    def __init__(self, n_layers, d_model, num_heads, d_ff, input_vocab_size, 
                 target_vocab_size, max_encoder_len, max_decoder_len, 
                 dropout_rate=0.1, pad_token_id=59513, **kwargs):

        super().__init__(**kwargs)
        self.pad_token_id = pad_token_id
        
        self.encoder = Encoder(
            n_layers, d_model, num_heads, d_ff, input_vocab_size, max_encoder_len, dropout_rate
        )
        self.decoder = Decoder(
            n_layers, d_model, num_heads, d_ff, target_vocab_size, max_decoder_len, dropout_rate
        )
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, training=False):
        encoder_input, decoder_input = inputs

        enc_padding_mask = create_padding_mask(encoder_input, self.pad_token_id)
        dec_padding_mask = create_decoder_padding_mask(decoder_input, self.pad_token_id)

        enc_output = self.encoder(encoder_input, training=training, mask=enc_padding_mask)
        dec_output = self.decoder(
            decoder_input, enc_output, training=training, 
            dec_padding_mask=dec_padding_mask, enc_padding_mask=enc_padding_mask
        )

        return self.final_layer(dec_output)

def build_transformer_model(num_tokens, max_encoder_len, max_decoder_len, num_heads=3,
                           d_model=256, n_layers=4, d_ff=1024, dropout_rate=0.1, 
                           warmup_steps=4000, initial_lr=1e-6, peak_lr=0.01, 
                           min_lr=1e-7, total_steps=35100, use_warmup=False, pad_token_id=59513):
    """
    Build and compile transformer model for neural machine translation.
    
    Args:
        num_tokens: Vocabulary size
        max_encoder_len: Maximum encoder sequence length
        max_decoder_len: Maximum decoder sequence length
        num_heads: Number of attention heads (default: 3)
        d_model: Model dimension (default: 256)
        n_layers: Number of encoder/decoder layers (default: 4)
        d_ff: Feed-forward dimension (default: 1024, typically 4 × d_model)
        dropout_rate: Dropout rate (default: 0.1)
        warmup_steps: Warmup steps for learning rate schedule (default: 4000)
        initial_lr: Initial learning rate during warmup (default: 1e-6)
        peak_lr: Peak learning rate after warmup (default: 0.01)
        min_lr: Minimum learning rate at end of training (default: 1e-7)
        total_steps: Total training steps for cosine annealing (default: 35100)
        use_warmup: If True, use TransformerSchedule with warmup; if False, use fixed LR (default: False)
        pad_token_id: Padding token ID (default: 59513 for MarianTokenizer)
    
    Returns:
        Compiled transformer model
    """
    
    model = Transformer(
        n_layers=n_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        input_vocab_size=num_tokens,
        target_vocab_size=num_tokens,
        max_encoder_len=max_encoder_len,
        max_decoder_len=max_decoder_len,
        dropout_rate=dropout_rate,
        pad_token_id=pad_token_id
    )
    
    # Choose learning rate schedule
    if use_warmup:
        learning_rate = TransformerSchedule(
            initial_lr=initial_lr,
            peak_lr=peak_lr,
            min_lr=min_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps
        )
        print(f'Using cosine annealing: initial_lr={initial_lr:.1e}, peak_lr={peak_lr}, ' \
              f'min_lr={min_lr:.1e}, warmup_steps={warmup_steps}, total_steps={total_steps}')

    else:
        learning_rate = 0.0003  # Fixed learning rate (no warmup, no scaling)
        print(f'Using fixed learning rate: {learning_rate}')
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    
    # Import masked loss/accuracy to ignore padding positions
    from .losses import masked_sparse_categorical_crossentropy, masked_accuracy
    
    # Create named wrapper for masked accuracy metric (avoids "<lambda>" in TensorBoard)
    def accuracy_metric(y_true, y_pred):
        return masked_accuracy(y_true, y_pred, pad_token_id)

    accuracy_metric.__name__ = 'accuracy'
    
    model.compile(
        optimizer=optimizer,
        loss=lambda y_true, y_pred: masked_sparse_categorical_crossentropy(y_true, y_pred, pad_token_id),
        metrics=[accuracy_metric]
    )
    
    return model


def translate_transformer(input_text, model, tokenizer, max_encoder_len, max_decoder_len):
    encoder_input = tokenizer(
        input_text,
        padding='max_length',
        truncation=True,
        max_length=max_encoder_len,
        return_tensors='np'
    )['input_ids']

    encoder_input = tf.constant(encoder_input, dtype=tf.int32)
    decoder_input = tf.constant([[tokenizer.eos_token_id]], dtype=tf.int32)

    for _ in range(max_decoder_len - 1):
        predictions = model([encoder_input, decoder_input], training=False)
        predicted_id = tf.argmax(predictions[:, -1:, :], axis=-1, output_type=tf.int32)

        if predicted_id.numpy()[0, 0] == tokenizer.eos_token_id:
            break

        decoder_input = tf.concat([decoder_input, predicted_id], axis=-1)

    return tokenizer.decode(decoder_input[0], skip_special_tokens=True)

def build_inference_models_transformer(model, latent_dim):
    """
    Dummy function for BLEUCallback compatibility.
    
    Transformers don't need separate inference models because:
    - Same model works for both training and inference
    - Causal masking handles autoregressive generation
    - No hidden states to maintain between steps
    
    Args:
        model: Trained transformer model (unused)
        latent_dim: Latent dimension (unused)
    
    Returns:
        Tuple of (None, None) for encoder_model, decoder_model
    """
    return None, None
