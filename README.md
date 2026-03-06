# Language models

This project introduces students to transformer models by building concepts sequentially, starting with RNN-based encoder-decoder architectures. By understanding how attention mechanisms solve the limitations of LSTMs, students gain the foundation needed to understand modern transformer architectures.

The project contains four Jupyter notebooks that progressively introduce neural machine translation concepts:

| Notebook | Description |
|----------|-------------|
| [01-encoder-decoder-LSTM.ipynb](notebooks/01-encoder-decoder-LSTM.ipynb) | Bidirectional LSTM encoder-decoder baseline |
| [02-encoder-decoder-LSTM-attention.ipynb](notebooks/02-encoder-decoder-LSTM-attention.ipynb) | Adding Luong attention mechanism |
| [03-encoder-decoder-transformer.ipynb](notebooks/03-encoder-decoder-transformer.ipynb) | Transformer architecture with self-attention |
| [04-fine-tuning-activity.ipynb](notebooks/04-fine-tuning-activity.ipynb) | Transfer learning activity (English-German fine-tuning) |

The resulting trained models are available on Hugging Face:

| Model | Description |
|-------|-------------|
| [gperdrizet/english-french-LSTM](https://huggingface.co/gperdrizet/english-french-LSTM) | Recurrent only encoder-decoder |
| [gperdrizet/english-french-LSTM-attention](https://huggingface.co/gperdrizet/english-french-LSTM-attention) | Recurrent encoder-decoder with attention |
| [gperdrizet/english-french-transformer](https://huggingface.co/gperdrizet/english-french-transformer) | Transformer based encoder-decoder |

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

- **Notebooks 01, 02, & 03:** Train from scratch (~5 hours on GPU for full training)
- **Notebook 04:** Fine-tune pre-trained model (~1-2 hours for 5 epochs)

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

If you want to upload your own models to Hugging Face after training, do the following:

### Prerequisites

1. Create repositories on Hugging Face Hub:
   - `YOUR_USERNAME/english-french-LSTM`
   - `YOUR_USERNAME/english-french-LSTM-attention`

2. Add your Hugging Face access token to `.env` file:
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

## References

**Attention mechanism:**

> Bahdanau, D., Cho, K., & Bengio, Y. (2015). **Neural machine translation by jointly learning to align and translate.** *Proceedings of the 3rd International Conference on Learning Representations (ICLR).* https://arxiv.org/abs/1409.0473

**Encoder-decoder architecture:**

> Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). **Learning phrase representations using RNN encoder-decoder for statistical machine translation.** *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP).* https://arxiv.org/abs/1406.1078

> Sutskever, I., Vinyals, O., & Le, Q. V. (2014). **Sequence to sequence learning with neural networks.** *Advances in Neural Information Processing Systems, 27.* https://arxiv.org/abs/1409.3215

**OPUS-100 dataset:**

> Zhang, B., Williams, P., Titov, I., & Sennrich, R. (2020). **Improving massively multilingual neural machine translation and zero-shot translation.** *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL).* https://arxiv.org/abs/2004.11867

**Transformer architecture:**

> Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). **Attention is all you need.** *Advances in Neural Information Processing Systems, 30.* https://arxiv.org/abs/1706.03762

**Transfer learning for neural machine translation:**

> Zoph, B., Yuret, D., May, J., & Knight, K. (2016). **Transfer learning for low-resource neural machine translation.** *Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (EMNLP).* https://arxiv.org/abs/1604.02201