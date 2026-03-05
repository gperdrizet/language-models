"""
Upload trained NMT models to Hugging Face Hub.

This script finds and uploads the latest checkpoint (highest epoch number) of 
the English-French LSTM and LSTM-attention models to Hugging Face Hub for 
sharing and fine-tuning.

Usage:
    python utils/upload_models_to_hub.py [--model lstm|attention|both]

Requirements:
    - HF_TOKEN environment variable set (or in .env file)
    - Training checkpoints in models/checkpoints/ directory
"""

import os
import argparse
import re
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Model configurations
MODELS = {
    'lstm': {
        'repo_id': 'gperdrizet/english-french-LSTM',
        'checkpoint_dir': 'models/checkpoints/lstm',
        'description': 'Bidirectional LSTM encoder-decoder for English-French translation',
        'architecture': 'Bidirectional LSTM encoder + LSTM decoder',
    },
    'attention': {
        'repo_id': 'gperdrizet/english-french-LSTM-attention',
        'checkpoint_dir': 'models/checkpoints/lstm-attention',
        'description': 'LSTM encoder-decoder with Luong attention for English-French translation',
        'architecture': 'Bidirectional LSTM encoder + LSTM decoder + Luong attention',
    }
}


def find_latest_checkpoint(checkpoint_dir):
    """Find the checkpoint with the highest epoch number."""
    
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        return None
    
    # Find all .h5 checkpoint files
    checkpoints = list(checkpoint_path.glob('model_epoch_*.h5'))
    
    if not checkpoints:
        return None
    
    # Extract epoch number from filenames and find the maximum
    # Pattern: model_epoch_XX_val_loss_Y.YYYY.h5
    epoch_pattern = re.compile(r'model_epoch_(\d+)_val_loss_[\d.]+\.h5')
    
    best_checkpoint = None
    max_epoch = -1
    
    for checkpoint in checkpoints:
        match = epoch_pattern.match(checkpoint.name)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                best_checkpoint = checkpoint
    
    return best_checkpoint


def create_model_card(model_config, model_name):
    """Generate a model card (README.md) for the model."""
    
    return f"""---
tags:
- neural-machine-translation
- lstm
- attention
- english
- french
- opus-100
- seq2seq
license: mit
---

# {model_config['description']}

This model was trained for English-to-French neural machine translation using the OPUS-100 dataset.

## Model Details

- **Architecture:** {model_config['architecture']}
- **Training data:** OPUS-100 English-French parallel corpus (100,000 sentence pairs, filtered to ≤20 tokens)
- **Tokenizer:** MarianTokenizer (Helsinki-NLP/opus-mt-en-fr)
- **Vocabulary size:** ~8,000 subword tokens
- **Latent dimension:** 256
- **Max sequence length:** 22 (encoder), 24 (decoder)

## Training Configuration

- **Optimizer:** Adam (learning rate: 0.001)
- **Loss function:** Sparse categorical crossentropy
- **Batch size:** 32
- **Epochs:** 15
- **Validation split:** 10%

## Usage

This model can be fine-tuned for other language pairs using transfer learning. See the fine-tuning activity notebook for examples.

### Loading the model

```python
from huggingface_hub import hf_hub_download
import tensorflow as tf

# Download model weights
weights_path = hf_hub_download(
    repo_id='{model_config['repo_id']}',
    filename='model_weights.h5'
)

# Build model architecture (see training notebook)
# Then load weights:
model.load_weights(weights_path)
```

## Fine-tuning for other language pairs

This model can be fine-tuned for other European language pairs (e.g., English-German, English-Spanish) with minimal additional training:

1. Build a new model with the target vocabulary
2. Load these pre-trained weights
3. Reinitialize embedding layers for new vocabulary
4. Fine-tune for 3-5 epochs

See the accompanying fine-tuning notebook for a complete example.

## Limitations

- Trained only on short sentences (≤20 tokens)
- Performance degrades on longer sequences
- Best suited for European language pairs with similar syntax
- Uses greedy decoding (no beam search)

## Citation

If you use this model, please cite:

```
@misc{{english-french-{model_name},
  author = {{George Perdrizet}},
  title = {{English-French Neural Machine Translation}},
  year = {{2026}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/{model_config['repo_id']}}}}},
}}
```

## Model Card Authors

George Perdrizet

## Model Card Contact

gperdrizet on GitHub
"""


def upload_model(model_name, token, force=False):
    """Upload a model to Hugging Face Hub."""
    
    if model_name not in MODELS:
        print(f"ERROR: Unknown model '{model_name}'. Choose from: {list(MODELS.keys())}")
        return False
    
    config = MODELS[model_name]
    
    # Find the latest checkpoint
    print(f"\nSearching for latest checkpoint in {config['checkpoint_dir']}...")
    model_path = find_latest_checkpoint(config['checkpoint_dir'])
    
    # Check if checkpoint was found
    if model_path is None:
        print(f"ERROR: No checkpoints found in {config['checkpoint_dir']}")
        print(f"Please train the model first using the corresponding notebook.")
        return False
    
    print(f"✓ Found latest checkpoint: {model_path.name}")
    
    print(f"\n{'='*60}")
    print(f"Uploading {model_name} model to Hugging Face Hub")
    print(f"{'='*60}")
    print(f"Repository: {config['repo_id']}")
    print(f"Checkpoint: {model_path}")
    print(f"Model size: {model_path.stat().st_size / (1024*1024):.2f} MB")
    
    try:
        # Initialize Hugging Face API
        api = HfApi()
        
        # Create repository if it doesn't exist
        print(f"\nCreating/updating repository...")
        create_repo(
            repo_id=config['repo_id'],
            token=token,
            exist_ok=True,
            repo_type='model'
        )
        
        # Generate and upload model card
        print(f"Generating model card...")
        model_card = create_model_card(config, model_name)
        
        # Create temporary README file
        readme_path = model_path.parent / 'README.md'
        with open(readme_path, 'w') as f:
            f.write(model_card)
        
        # Upload README
        print(f"Uploading model card...")
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo='README.md',
            repo_id=config['repo_id'],
            token=token
        )
        
        # Upload model weights
        print(f"Uploading model weights (this may take a few minutes)...")
        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo='model_weights.h5',
            repo_id=config['repo_id'],
            token=token
        )
        
        print(f"\n✓ Successfully uploaded {model_name} model!")
        print(f"  View at: https://huggingface.co/{config['repo_id']}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error uploading model: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Upload trained NMT models to Hugging Face Hub'
    )
    parser.add_argument(
        '--model',
        choices=['lstm', 'attention', 'both'],
        default='both',
        help='Which model to upload (default: both)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force upload even if files already exist'
    )
    
    args = parser.parse_args()
    
    # Check for HF_TOKEN
    token = os.getenv('HF_TOKEN')
    if not token:
        print("ERROR: HF_TOKEN environment variable not set")
        print("Please create a .env file with your Hugging Face token:")
        print("  HF_TOKEN=your_token_here")
        print("\nGet your token at: https://huggingface.co/settings/tokens")
        return
    
    print("Hugging Face Model Upload Utility")
    print("=" * 60)
    
    # Upload requested models
    if args.model in ['lstm', 'both']:
        success_lstm = upload_model('lstm', token, args.force)
    else:
        success_lstm = True
    
    if args.model in ['attention', 'both']:
        success_attention = upload_model('attention', token, args.force)
    else:
        success_attention = True
    
    # Summary
    print("\n" + "=" * 60)
    print("Upload Summary")
    print("=" * 60)
    
    if args.model in ['lstm', 'both']:
        status = "✓ Success" if success_lstm else "✗ Failed"
        print(f"LSTM model: {status}")
    
    if args.model in ['attention', 'both']:
        status = "✓ Success" if success_attention else "✗ Failed"
        print(f"Attention model: {status}")
    
    if (args.model == 'both' and success_lstm and success_attention) or \
       (args.model == 'lstm' and success_lstm) or \
       (args.model == 'attention' and success_attention):
        print("\n✓ All uploads completed successfully!")
    else:
        print("\n✗ Some uploads failed. Check error messages above.")


if __name__ == '__main__':
    main()
