# Neural Machine Translation with LSTMs

Educational notebooks demonstrating encoder-decoder architectures for neural machine translation (English-French) with TensorFlow/Keras.

Models are avalible on Hugging Face:
- [gperdrizet/english-french-LSTM](https://huggingface.co/gperdrizet/english-french-LSTM)
- [gperdrizet/english-french-LSTM-attention](https://huggingface.co/gperdrizet/english-french-LSTM-attention)

## Overview

This project contains three Jupyter notebooks that progressively introduce neural machine translation concepts:

1. **01-encoder-decoder-NMT-LSTM.ipynb** - Bidirectional LSTM encoder-decoder baseline
2. **02-encoder-decoder-LSTM-attention.ipynb** - Adding Luong attention mechanism
3. **03-fine-tuning-activity.ipynb** - Transfer learning activity (English-German fine-tuning)

## Quick start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run notebooks

Open notebooks in Jupyter or VS Code and execute cells sequentially. Notebooks are self-contained with inline documentation.

### Training

- **Notebook 01 & 02:** Train from scratch (~5 hours on GPU for full 15 epochs)
- **Notebook 03:** Fine-tune pre-trained model (~1-2 hours for 5 epochs)

### Visualize training

```bash
tensorboard --logdir logs/
```

## Project structure

```
.
├── notebooks/   # Jupyter notebooks
├── src/         # Reusable model functions and callbacks
├── utils/       # Utility scripts (model upload, etc.)
├── models/      # Saved models and checkpoints (generated)
├── logs/        # TensorBoard logs (generated)
└── data/        # Dataset cache (generated)
```

## Model upload

After training, upload models to Hugging Face Hub:

```bash
python utils/upload_models_to_hub.py
```

The upload script automatically:
1. Finds the latest training checkpoint
2. Rebuilds models and converts to SavedModel format
3. Generates model card (README)
4. Uploads to Hugging Face Hub

The script works directly from checkpoints saved during training.

See [utils/README.md](utils/README.md) for detailed instructions.

## Features

- **Subword tokenization:** MarianTokenizer (SentencePiece) for handling rare words
- **Bidirectional encoder:** Captures context from both directions
- **Attention mechanism:** Luong-style attention over encoder hidden states
- **BLEU evaluation:** Automatic corpus-level scoring during training
- **TensorBoard logging:** Loss, accuracy, and BLEU score visualization
- **Transfer learning:** Fine-tune for new language pairs in 5 epochs

## Dataset

OPUS-100 English-French parallel corpus from Hugging Face:
- 100,000 sentence pairs
- Filtered to <=20 tokens per sentence
- 10% validation split

## Architecture details

**LSTM model:**
- Bidirectional LSTM encoder (256 units per direction)
- LSTM decoder (512 units)
- Teacher forcing during training
- Separate encoder/decoder for inference

**Attention model:**
- Same as LSTM + Luong attention layer
- Attention weights over encoder hidden states
- Handles long-range dependencies

## License

MIT