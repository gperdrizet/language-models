# Neural Machine Translation with LSTMs

Educational notebooks demonstrating encoder-decoder architectures for neural machine translation (English-French) with TensorFlow/Keras.

## Overview

This project contains three Jupyter notebooks that progressively introduce neural machine translation concepts:

1. **01-encoder-decoder-NMT-LSTM.ipynb** - Bidirectional LSTM encoder-decoder baseline
2. **02-encoder-decoder-LSTM-attention.ipynb** - Adding Luong attention mechanism
3. **03-fine-tuning-activity.ipynb** - Transfer learning activity (English-German fine-tuning)

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run Notebooks

Open notebooks in Jupyter or VS Code and execute cells sequentially. Notebooks are self-contained with inline documentation.

### Training

- **Notebook 01 & 02:** Train from scratch (~5 hours on GPU for full 15 epochs)
- **Notebook 03:** Fine-tune pre-trained model (~1-2 hours for 5 epochs)

### Visualize Training

```bash
tensorboard --logdir logs/
```

## Project Structure

```
.
├── notebooks/           # Jupyter notebooks
├── src/                # Reusable model functions and callbacks
├── utils/              # Utility scripts (model upload, etc.)
├── models/             # Saved models and checkpoints (generated)
├── logs/               # TensorBoard logs (generated)
└── data/               # Dataset cache (generated)
```

## Model Upload

After training, upload models to Hugging Face Hub for sharing:

```bash
python utils/upload_models_to_hub.py
```

See [utils/README.md](utils/README.md) for detailed instructions.

## Features

- **Subword tokenization:** MarianTokenizer (SentencePiece) for handling rare words
- **Bidirectional encoder:** Captures context from both directions
- **Attention mechanism:** Luong-style attention for better long-range dependencies
- **BLEU evaluation:** Automatic corpus-level scoring during training
- **TensorBoard logging:** Loss, accuracy, and BLEU score visualization
- **Transfer learning:** Fine-tune for new language pairs in 5 epochs

## Dataset

OPUS-100 English-French parallel corpus from Hugging Face:
- 100,000 sentence pairs
- Filtered to ≤20 tokens per sentence
- 10% validation split

## Architecture Details

**LSTM model:**
- Bidirectional LSTM encoder (256 units per direction)
- LSTM decoder (512 units)
- Teacher forcing during training
- Separate encoder/decoder for inference

**Attention model:**
- Same as LSTM + Luong attention layer
- Attention weights over encoder hidden states
- Improved handling of long sentences

## License

MIT