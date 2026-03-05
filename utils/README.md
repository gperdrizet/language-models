# Model upload utility

This directory contains utility scripts for managing the neural machine translation models.

## upload_models_to_hub.py

Converts training checkpoints to SavedModel format and uploads to Hugging Face Hub.

**The script automatically:**
1. Finds the latest checkpoint (highest epoch number) from training
2. Rebuilds model architecture and loads checkpoint weights
3. Builds inference models (encoder/decoder) from training model
4. Saves all as SavedModel format
5. Generates comprehensive model card
6. Uploads to Hugging Face Hub

### Prerequisites

1. **Hugging Face account and token:**
   - Create an account at [huggingface.co](https://huggingface.co)
   - Generate a token at [Settings > Access Tokens](https://huggingface.co/settings/tokens)
   - Add token to `.env` file in project root:
     ```
     HF_TOKEN=your_token_here
     ```

2. **Trained models:**
   - Train models using notebooks 01 and/or 02
   - Checkpoints are automatically saved during training to:
     - `models/checkpoints/lstm/model_epoch_XX_val_loss_Y.YYYY.h5`
     - `models/checkpoints/lstm-attention/model_epoch_XX_val_loss_Y.YYYY.h5`

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

1. **Searches for checkpoints:** Finds the latest checkpoint by epoch number
2. **Builds models:** Rebuilds training model architecture and loads checkpoint weights
3. **Creates inference models:** Builds encoder/decoder for deployment
4. **Converts to SavedModel:** Saves all models in TensorFlow SavedModel format
5. **Generates model card:** Creates comprehensive README with:
   - Model architecture details
   - Training configuration
   - Usage instructions for loading and translation
   - Fine-tuning guide
   - Limitations and citations
6. **Uploads to Hugging Face:** Uploads all files (training model, encoder, decoder, config, tokenizer, README)

### Output

Models will be uploaded to:
- LSTM: `gperdrizet/english-french-LSTM`
- Attention: `gperdrizet/english-french-LSTM-attention`

View online at `https://huggingface.co/<repo_id>`

SavedModel format provides:
- **Cross-version compatibility** across TensorFlow versions
- **Production deployment** support (TF Serving, TF Lite, TF.js)

Users can then load models directly with:
```python
from huggingface_hub import snapshot_download
import tensorflow as tf

model_path = snapshot_download(repo_id='gperdrizet/english-french-LSTM')
encoder = tf.keras.models.load_model(f'{model_path}/encoder_model')
decoder = tf.keras.models.load_model(f'{model_path}/decoder_model')
```
