# Project Guidelines

## Code style

This project uses **Jupyter notebooks** for neural machine translation research with TensorFlow/Keras. Follow patterns from [01-encoder-decoder-NMT-LSTM.ipynb](../notebooks/01-encoder-decoder-NMT-LSTM.ipynb) and [02-encoder-decoder-LSTM-attention.ipynb](../notebooks/02-encoder-decoder-LSTM-attention.ipynb).

**TensorFlow configuration pattern (must precede TensorFlow import):**
```python
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Select GPU

import tensorflow as tf

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

**Reproducibility:** Always set `np.random.seed(315)` and `tf.random.set_seed(315)` early in notebooks.

**Notebook editing:** Use notebook editing tools (like `edit_notebook_file`) rather than modifying JSON on disk. Editing notebook JSON directly may miss changes the user has made in the active notebook viewer and doesn't let them see changes after they have been made.

**Import organization:** Consolidate all imports (except `src` module imports) in the main imports cell at the top of notebooks. This includes:
- Standard library: `os`, `sys`, `logging`, `datetime`
- Third-party: `numpy`, `tensorflow`, `matplotlib`, `datasets`, `transformers`, `sacrebleu`
- Keras components: `layers`, `callbacks`
- Keep `src` imports in their respective usage cells for context

### General patterns
- **Docstrings**: Google style with `Args:` and `Returns:` sections
- **Type hints**: Not required but use in function signatures when helpful
- **Variable names**: Descriptive over terse (`latent_dim` not `ld`)
- **Constants**: Use lowercase (not ALL_CAPS) for module-level constants and configuration variables
  - Example: `image_size = 256` not `IMAGE_SIZE = 256`
  - Exception: Python dunder constants like `__version__` follow Python conventions
- **Line length**: ~88 chars (Black-style) but not enforced

### Python conventions
```python
# Good: Clear, documented utility function
def load_df2k_ost(split='train', max_images=None):
    """
    Load DF2K_OST dataset from HuggingFace Hub.
    
    Args:
        split: 'train' or 'validation'
        max_images: Maximum number of images to load (None = all)
    
    Returns:
        numpy array of shape (N, 256, 256, 3) with values in [0, 1]
    """
```

### Code formatting for readability
- **Blank lines in control blocks**: Add blank lines within `try/except` and `if/else` blocks to visually separate logical groups
  - After variable initialization before control flow starts
  - Between the end of a loop/operation and a following print/return statement
  - After the last statement in a `try` block before the `except`
  - After control blocks before the next section

Example:
```python
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        print(f'GPU memory growth enabled for {len(gpus)} GPU(s)')

    except RuntimeError as e:
        print(f'GPU configuration error: {e}')
```

## Markdown and comments style

- Use sentence case for all titles, headings, list elements, etc
- Do not use emojis or decorative symbols (✓ ✗ ★)
- Functional symbols like → for transformations/translations are acceptable
- Use standard operators: `<=` not `≤`, `-` not `—` (em dash)
- Keep documentation factual - avoid comparative language ("better", "improved", "enhancement")

## Architecture

**Encoder-decoder models with training/inference separation:**
- **Training model:** Teacher forcing with full sequences, compiled with loss/metrics
- **Inference models:** Separate encoder/decoder for autoregressive generation (no teacher forcing)
- Rebuild inference models from trained weights using `build_inference_models()`

**Critical dimension pattern for bidirectional encoders:**
- Encoder LSTM: `latent_dim` per direction
- Decoder LSTM: `latent_dim * 2` (concatenated forward+backward states)
- See `build_bidirectional_model()` and `build_attention_model()` for examples

**Decoder input alignment:**
- Training: `decoder_input[0] = pad_token_id` (BOS), `decoder_target[0] = first_real_token`
- Inference: Start with `np.array([[tokenizer.pad_token_id]])`
- This pattern ensures training/inference consistency

## Build and test

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run notebooks:** Use Jupyter or VS Code notebook interface. Notebooks are self-contained with inline documentation.

**No automated tests** - validation uses BLEU scores computed via `BLEUCallback` during training.

## Project conventions

**Tokenization:** Use Hugging Face `MarianTokenizer` for subword tokenization (example: `Helsinki-NLP/opus-mt-en-fr`). The tokenizer handles both source and target languages.
- Vocabulary size: ~60,000 tokens (59,514 for Helsinki-NLP/opus-mt-en-fr)

**Sequence length configuration:**
- `max_seq_length = 20` (filter dataset)
- `max_encoder_len = 22` (padding for special tokens)
- `max_decoder_len = 24` (extra space for BOS)
- Note: Use lowercase for configuration variables, not ALL_CAPS

**Model checkpointing:** `BLEUCallback` tracks best BLEU score and restores weights at end of training. This handles overfitting where validation loss diverges but translation quality improves.

**Translation:** Greedy decoding in `translate()` function - decoder outputs fed back autoregressively until `eos_token_id`.

**ASCII diagrams:** Notebooks include detailed architecture diagrams using ASCII art. Maintain this pattern for new architectures.

## Integration points

- **Datasets:** `load_dataset('Helsinki-NLP/opus-100', 'en-fr')` for parallel text
- **Tokenizer:** `MarianTokenizer.from_pretrained()` - multilingual SentencePiece model
- **Evaluation:** `sacrebleu.metrics.BLEU` for corpus-level scoring (not sentence-level)

## Model sharing

**Upload utility:** `utils/upload_models_to_hub.py` uploads trained models to Hugging Face Hub for sharing and transfer learning.

Usage:
```bash
python utils/upload_models_to_hub.py [--model lstm|attention|both]
```

Requirements:
- `HF_TOKEN` in `.env` file (Hugging Face access token)
- Training checkpoints in `models/checkpoints/{lstm,lstm-attention}/` directory

The script automatically finds and uploads the latest checkpoint (highest epoch number) from the training runs. The script automatically generates model cards with architecture details, training configuration, usage examples, and fine-tuning instructions.

## Security

HF_TOKEN stored in `.env` file (excluded from git via `.gitignore`). Public datasets from Hugging Face require no authentication.
