# Neural machine translation with LSTMs

Educational notebooks demonstrating encoder-decoder architectures for neural machine translation (English-French) with TensorFlow/Keras.

Models are avalible on Hugging Face:
- [gperdrizet/english-french-LSTM](https://huggingface.co/gperdrizet/english-french-LSTM)
- [gperdrizet/english-french-LSTM-attention](https://huggingface.co/gperdrizet/english-french-LSTM-attention)

## Overview

This project introduces students to transformer models by building concepts sequentially, starting with RNN-based encoder-decoder architectures. By understanding how attention mechanisms solve the limitations of LSTMs, students gain the foundation needed to understand modern transformer architectures.

The project contains three Jupyter notebooks that progressively introduce neural machine translation concepts:

1. **01-encoder-decoder-NMT-LSTM.ipynb** - Bidirectional LSTM encoder-decoder baseline
2. **02-encoder-decoder-LSTM-attention.ipynb** - Adding Luong attention mechanism
3. **03-fine-tuning-activity.ipynb** - Transfer learning activity (English-German fine-tuning)

## Getting started

### Running in a devcontainer

This project is configured to run in a devcontainer with all dependencies pre-installed.

**Option 1: Local VS Code with Docker**

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/language-models.git
   cd language-models
   ```
3. Open in VS Code:
4. When prompted, click "Reopen in Container" (or use Command Palette: "Dev Containers: Reopen in Container")

**Option 2: GitHub Codespaces**

1. Fork the repository on GitHub
2. Click "Code" → "Codespaces" → "Create codespace on main"
3. Wait for the environment to build (first build takes ~5 minutes)

### Quick start

Once in the devcontainer:

1. Open any notebook in the `notebooks/` directory
2. Select the Python kernel when prompted
3. Run cells sequentially

### Training

- **Notebook 01 & 02:** Train from scratch (~5 hours on GPU for full 15 epochs)
- **Notebook 03:** Fine-tune pre-trained model (~1-2 hours for 5 epochs)

### Visualize training

Use the TensorBoard VS Code extension to visualize training metrics:

1. Open the Command Palette (Ctrl+Shift+P / Cmd+Shift+P)
2. Run "Python: Launch TensorBoard"
3. Select the `logs/` directory when prompted
4. View loss, accuracy, and BLEU score plots in the TensorBoard panel

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

After training, upload models to Hugging Face Hub.

### Prerequisites

1. Create repositories on Hugging Face Hub:
   - `YOUR_USERNAME/english-french-LSTM`
   - `YOUR_USERNAME/english-french-LSTM-attention`

2. Add your Hugging Face access token to `.env`:
   ```bash
   HF_TOKEN=your_token_here
   ```
   (Get your token from https://huggingface.co/settings/tokens)

### Upload command

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

## Dataset

[OPUS-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100) English-French parallel corpus from Hugging Face:
- 100,000 sentence pairs
- Filtered to <=20 tokens per sentence
- 10% validation split

> Zhang, B., Williams, P., Titov, I., & Sennrich, R. (2020). **Improving massively multilingual neural machine translation and zero-shot translation.** *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL).* https://arxiv.org/abs/2004.11867