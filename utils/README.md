# Model Upload Utility

This directory contains utility scripts for managing the neural machine translation models.

## upload_models_to_hub.py

Uploads trained English-French NMT models to Hugging Face Hub for sharing and fine-tuning.

**The script automatically finds and uploads the latest checkpoint** (highest epoch number) from the training runs.

### Prerequisites

1. **Hugging Face account and token:**
   - Create an account at [huggingface.co](https://huggingface.co)
   - Generate a token at [Settings → Access Tokens](https://huggingface.co/settings/tokens)
   - Add token to `.env` file in project root:
     ```
     HF_TOKEN=your_token_here
     ```

2. **Trained models:**
   - Train models using notebooks 01 and/or 02
   - Checkpoints are automatically saved during training at:
     - `models/checkpoints/lstm/model_epoch_XX_val_loss_Y.YYYY.h5`
     - `models/checkpoints/lstm-attention/model_epoch_XX_val_loss_Y.YYYY.h5`
   - The script will upload the checkpoint with the highest epoch number

### Usage

Upload both models:
```bash
python utils/upload_models_to_hub.py
```

Upload only the LSTM model:
```bash
python utils/upload_models_to_hub.py --model lstm
```

Upload only the attention model:
```bash
python utils/upload_models_to_hub.py --model attention
```

Force re-upload even if files exist:
```bash
python utils/upload_models_to_hub.py --force
```

### What it does

1. Validates that model files exist locally
2. Creates/updates Hugging Face repository
3. Generates comprehensive model card (README.md) with:
   - Model architecture details
   - Training configuration
   - Usage instructions
   - Fine-tuning guide
   - Limitations and citations
4. Uploads model weights and README to Hugging Face Hub

### Output

Models will be uploaded to:
- LSTM: `gperdrizet/english-french-LSTM`
- Attention: `gperdrizet/english-french-LSTM-attention`

View online at `https://huggingface.co/<repo_id>`
