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
    """
    Generate positional encoding matrix using sine/cosine functions.
    
    Args:
        seq_len: Maximum sequence length
        d_model: Model dimension
    
    Returns:
        Positional encoding matrix of shape (seq_len, d_model)
    """
    # Create position indices
    positions = np.arange(seq_len)[:, np.newaxis]
    
    # Create dimension indices  
    dims = np.arange(d_model)[np.newaxis, :]
    
    # Compute angle rates
    angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / d_model)
    angle_rads = positions * angle_rates
    
    # Apply sin to even indices, cos to odd indices
    pos_encoding = np.zeros((seq_len, d_model))
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    return pos_encoding.astype(np.float32)


class PositionalEncoding(tf.keras.layers.Layer):
    """
    Adds positional encoding to input embeddings.
    
    Args:
        max_len: Maximum sequence length
        d_model: Model dimension
    """
    
    def __init__(self, max_len, d_model):
        super().__init__()
        # Convert to TensorFlow constant so it can be sliced with TF tensors
        pos_encoding_np = get_positional_encoding(max_len, d_model)
        self.pos_encoding = tf.constant(pos_encoding_np, dtype=tf.float32)
    
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:seq_len, :]


def feed_forward_network(d_model, d_ff):
    """
    Position-wise feed-forward network.
    
    Args:
        d_model: Model dimension
        d_ff: Feed-forward dimension (typically 4 * d_model)
    
    Returns:
        Sequential model with two dense layers and ReLU activation
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(d_ff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    """
    Single transformer encoder layer with self-attention and feed-forward network.
    
    Args:
        d_model: Model dimension
        d_ff: Feed-forward dimension
        dropout_rate: Dropout rate
    """
    
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super().__init__()
        
        # Simple dot-product self-attention (same as LSTM attention)
        self.attention = tf.keras.layers.Attention(dropout=dropout_rate)
        self.ffn = feed_forward_network(d_model, d_ff)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x, training, mask=None):
        # Self-attention: query=key=value=x
        # Pass mask as [query_mask, value_mask] list if provided
        mask_list = [mask, mask] if mask is not None else None
        attn_output = self.attention([x, x], mask=mask_list, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


class DecoderLayer(tf.keras.layers.Layer):
    """
    Single transformer decoder layer with masked self-attention, cross-attention, and feed-forward network.
    
    Args:
        d_model: Model dimension
        d_ff: Feed-forward dimension
        dropout_rate: Dropout rate
    """
    
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super().__init__()
        
        # Simple dot-product attention (same as LSTM attention)
        self.self_attention = tf.keras.layers.Attention(dropout=dropout_rate)
        self.cross_attention = tf.keras.layers.Attention(dropout=dropout_rate)
        self.ffn = feed_forward_network(d_model, d_ff)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x, enc_output, training, dec_padding_mask=None, enc_padding_mask=None):
        # Masked self-attention: decoder attends to previous positions
        # Use causal masking (look-ahead) automatically
        dec_mask_list = [dec_padding_mask, dec_padding_mask] if dec_padding_mask is not None else None
        attn1 = self.self_attention([x, x], mask=dec_mask_list, use_causal_mask=True, training=training)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)
        
        # Cross-attention: decoder attends to encoder (same as LSTM attention)
        # Query from decoder, value/key from encoder
        cross_mask_list = [dec_padding_mask, enc_padding_mask] if enc_padding_mask is not None else None
        attn2 = self.cross_attention([out1, enc_output], mask=cross_mask_list, training=training)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)
        
        # Feed-forward network
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)
        
        return out3


class Encoder(tf.keras.layers.Layer):
    """
    Transformer encoder: embedding + positional encoding + N encoder layers.
    
    Args:
        n_layers: Number of encoder layers
        d_model: Model dimension
        d_ff: Feed-forward dimension
        vocab_size: Vocabulary size
        max_len: Maximum sequence length
        dropout_rate: Dropout rate
    """
    
    def __init__(self, n_layers, d_model, d_ff, vocab_size, max_len, dropout_rate=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_len, d_model)
        
        self.enc_layers = [
            EncoderLayer(d_model, d_ff, dropout_rate)
            for _ in range(n_layers)
        ]
        
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x, training, mask=None):
        # Embedding + positional encoding
        x = self.embedding(x)
        # Note: Removed sqrt(d_model) scaling for training stability without warmup
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        
        # Pass through encoder layers
        for enc_layer in self.enc_layers:
            x = enc_layer(x, training, mask)
        
        return x


class Decoder(tf.keras.layers.Layer):
    """
    Transformer decoder: embedding + positional encoding + N decoder layers.
    
    Args:
        n_layers: Number of decoder layers
        d_model: Model dimension
        d_ff: Feed-forward dimension
        vocab_size: Vocabulary size
        max_len: Maximum sequence length
        dropout_rate: Dropout rate
    """
    
    def __init__(self, n_layers, d_model, d_ff, vocab_size, max_len, dropout_rate=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_len, d_model)
        
        self.dec_layers = [
            DecoderLayer(d_model, d_ff, dropout_rate)
            for _ in range(n_layers)
        ]
        
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x, enc_output, training, dec_padding_mask=None, enc_padding_mask=None):
        # Embedding + positional encoding
        x = self.embedding(x)
        # Note: Removed sqrt(d_model) scaling for training stability without warmup
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        
        # Pass through decoder layers
        for dec_layer in self.dec_layers:
            x = dec_layer(x, enc_output, training, dec_padding_mask, enc_padding_mask)
        
        return x


def create_padding_mask(seq):
    """
    Create mask for padding tokens (zeros).
    
    Args:
        seq: Input sequence of shape (batch, seq_len)
    
    Returns:
        Padding mask of shape (batch, seq_len) with dtype bool.
        For use with layers.Attention as [query_mask, value_mask].
    """
    # Mark padding positions (value == 0) as True (masked out)
    mask = tf.cast(tf.math.equal(seq, 0), tf.bool)
    return mask


class Transformer(Model):
    """
    Complete transformer model for sequence-to-sequence tasks.
    
    Args:
        n_layers: Number of encoder/decoder layers
        d_model: Model dimension
        d_ff: Feed-forward dimension
        input_vocab_size: Source vocabulary size
        target_vocab_size: Target vocabulary size
        max_encoder_len: Maximum encoder sequence length
        max_decoder_len: Maximum decoder sequence length
        dropout_rate: Dropout rate
    """
    
    def __init__(self, n_layers, d_model, d_ff, input_vocab_size, 
                 target_vocab_size, max_encoder_len, max_decoder_len, dropout_rate=0.1):
        super().__init__()
        
        self.encoder = Encoder(n_layers, d_model, d_ff, 
                              input_vocab_size, max_encoder_len, dropout_rate)
        
        self.decoder = Decoder(n_layers, d_model, d_ff,
                              target_vocab_size, max_decoder_len, dropout_rate)
        
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    
    def call(self, inputs, training=False):
        encoder_input, decoder_input = inputs
        
        # Create padding masks (shape: batch, seq_len)
        enc_padding_mask = create_padding_mask(encoder_input)
        dec_padding_mask = create_padding_mask(decoder_input)
        
        # Encoder
        enc_output = self.encoder(encoder_input, training, enc_padding_mask)
        
        # Decoder (uses causal masking internally via use_causal_mask=True)
        dec_output = self.decoder(decoder_input, enc_output, training, 
                                 dec_padding_mask, enc_padding_mask)
        
        # Final linear layer
        output = self.final_layer(dec_output)
        
        return output


def build_transformer_model(num_tokens, max_encoder_len, max_decoder_len, 
                           d_model=256, n_layers=4, d_ff=1024, dropout_rate=0.1, learning_rate=0.001):
    """
    Build and compile transformer model for neural machine translation.
    
    Args:
        num_tokens: Vocabulary size
        max_encoder_len: Maximum encoder sequence length
        max_decoder_len: Maximum decoder sequence length
        d_model: Model dimension (default: 256)
        n_layers: Number of encoder/decoder layers (default: 4)
        d_ff: Feed-forward dimension (default: 1024, typically 4 × d_model)
        dropout_rate: Dropout rate (default: 0.1)
        learning_rate: Learning rate for Adam optimizer (default: 0.001)
    
    Returns:
        Compiled transformer model
    """
    model = Transformer(
        n_layers=n_layers,
        d_model=d_model,
        d_ff=d_ff,
        input_vocab_size=num_tokens,
        target_vocab_size=num_tokens,
        max_encoder_len=max_encoder_len,
        max_decoder_len=max_decoder_len,
        dropout_rate=dropout_rate
    )
    
    # Use specified learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    return model


def translate_transformer(input_text, model, tokenizer, max_encoder_len, max_decoder_len):
    """
    Translate text using trained transformer with greedy decoding.
    
    Args:
        input_text: Source text to translate
        model: Trained transformer model
        tokenizer: Tokenizer for encoding/decoding
        max_encoder_len: Maximum encoder sequence length
        max_decoder_len: Maximum decoder sequence length
    
    Returns:
        Translated text
    """
    # Tokenize input
    encoder_input = tokenizer(
        input_text,
        padding='max_length',
        truncation=True,
        max_length=max_encoder_len,
        return_tensors='np'
    )['input_ids']
    
    # Convert to TensorFlow tensor
    encoder_input = tf.constant(encoder_input, dtype=tf.int64)
    
    # Initialize decoder input with PAD token (BOS)
    decoder_input = tf.constant([[tokenizer.pad_token_id]], dtype=tf.int32)
    
    # Autoregressive generation
    for _ in range(max_decoder_len - 1):
        # Predict next token
        predictions = model([encoder_input, decoder_input], training=False)
        
        # Get last token prediction
        predicted_id = tf.argmax(predictions[:, -1:, :], axis=-1, output_type=tf.int32)
        
        # Stop if EOS token
        if predicted_id.numpy()[0, 0] == tokenizer.eos_token_id:
            break
        
        # Append to decoder input
        decoder_input = tf.concat([decoder_input, predicted_id], axis=-1)
    
    # Decode tokens to text
    output_text = tokenizer.decode(decoder_input[0], skip_special_tokens=True)
    
    return output_text


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
