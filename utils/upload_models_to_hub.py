"""
Upload trained NMT models to Hugging Face Hub.

This script converts training checkpoints to SavedModel format and uploads them to 
Hugging Face Hub. It:
1. Finds the best checkpoint (highest BLEU score) from training_metrics.json
2. Rebuilds model architecture and loads checkpoint weights
3. Builds inference models (encoder/decoder)
4. Saves all as SavedModel format
5. Generates model card and uploads

Usage:
    python utils/upload_models_to_hub.py [--model lstm|attention|both]

Requirements:
    - HF_TOKEN environment variable set (or in .env file)
    - Training checkpoints in models/<MODEL>/checkpoints/ directory
    - training_metrics.json in models/<MODEL>/ directory
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv
import tensorflow as tf

# Add project root to path for imports
sys.path.append('.')

# Load environment variables
load_dotenv()

# Model configurations  
MODELS = {
    'lstm': {
        'repo_id': 'gperdrizet/english-french-LSTM',
        'model_dir': 'models/english-french-LSTM',
        'checkpoint_dir': 'models/lstm/checkpoints',
        'metrics_dir': 'models/lstm',  # Where training_metrics.json is stored
        'description': 'Bidirectional LSTM encoder-decoder for English-French translation',
        'architecture': 'Bidirectional LSTM encoder + LSTM decoder',
        'model_type': 'lstm',
    },
    'attention': {
        'repo_id': 'gperdrizet/english-french-LSTM-attention',
        'model_dir': 'models/english-french-LSTM-attention',
        'checkpoint_dir': 'models/lstm-attention/checkpoints',
        'metrics_dir': 'models/lstm-attention',  # Where training_metrics.json is stored
        'description': 'LSTM encoder-decoder with Luong attention for English-French translation',
        'architecture': 'Bidirectional LSTM encoder + LSTM decoder + Luong attention',
        'model_type': 'attention',
    }
}

# Default configuration (used when building models from checkpoints)
DEFAULT_CONFIG = {
    'vocab_size': 59514,  # MarianTokenizer vocab size
    'max_encoder_len': 22,
    'max_decoder_len': 24,
    'latent_dim': 256,
    'num_samples': 100000,
    'tokenizer': 'Helsinki-NLP/opus-mt-en-fr'
}


def find_best_checkpoint(model_dir, checkpoint_dir):
    """Find the checkpoint with the best BLEU score from training_metrics.json.
    
    Args:
        model_dir: Model directory containing training_metrics.json
        checkpoint_dir: Directory containing checkpoint files
    
    Returns:
        tuple: (checkpoint_path, epoch, bleu_score) or (None, None, None)
    """
    # First, try to load from training_metrics.json
    metrics_path = Path(model_dir) / 'training_metrics.json'
    
    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
            
            checkpoint_file = metrics['checkpoint_file']
            best_epoch = metrics['best_epoch']
            best_bleu = metrics['best_bleu']
            
            checkpoint_path = Path(checkpoint_dir) / checkpoint_file
            
            if checkpoint_path.exists():
                return checkpoint_path, best_epoch, best_bleu
            else:
                print(f"Warning: Checkpoint file {checkpoint_file} not found at expected location")
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Warning: Could not read training_metrics.json: {e}")
    
    # Fallback: find the latest checkpoint by epoch number
    print("Falling back to finding latest checkpoint by epoch number...")
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        return None, None, None
    
    # Find all .h5 checkpoint files
    checkpoints = list(checkpoint_path.glob('model_epoch_*.h5'))
    
    if not checkpoints:
        return None, None, None
    
    # Extract epoch number from filenames and find the maximum
    # Pattern: model_epoch_XX_best_bleu_YY.YY.h5 or model_epoch_XX_val_loss_Y.YYYY.h5
    epoch_pattern = re.compile(r'model_epoch_(\d+)_')
    bleu_pattern = re.compile(r'best_bleu_([\d.]+)\.h5')
    
    best_checkpoint = None
    max_epoch = -1
    best_bleu = None
    
    for checkpoint in checkpoints:
        match = epoch_pattern.search(checkpoint.name)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
                best_checkpoint = checkpoint
                
                # Try to extract BLEU score from filename
                bleu_match = bleu_pattern.search(checkpoint.name)
                if bleu_match:
                    best_bleu = float(bleu_match.group(1))
    
    return best_checkpoint, max_epoch if best_checkpoint else None, best_bleu


def check_savedmodel_exists(model_dir):
    """Check if SavedModel already exists in model directory."""
    
    model_path = Path(model_dir)
    
    if not model_path.exists():
        return False
    
    # Check for required SavedModel directories
    required_dirs = ['training_model', 'encoder_model', 'decoder_model']
    
    for model_dir_name in required_dirs:
        saved_model_pb = model_path / model_dir_name / 'saved_model.pb'
        if not saved_model_pb.exists():
            return False
    
    # Check for config.json
    if not (model_path / 'config.json').exists():
        return False
    
    return True


def build_models_from_checkpoint(checkpoint_path, model_type, model_dir):
    """
    Build training and inference models from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_type: 'lstm' or 'attention'
        model_dir: Directory to save models
    
    Returns:
        config dict with model metadata
    """
    from src import (
        build_bidirectional_model,
        build_attention_model,
        build_inference_models_lstm,
        build_inference_models_attention
    )
    from transformers import MarianTokenizer
    
    print(f"\n{'='*60}")
    print("Building models from checkpoint")
    print(f"{'='*60}")
    
    # Use default config
    config = DEFAULT_CONFIG.copy()
    config['architecture'] = MODELS[model_type]['architecture']
    config['best_epoch'] = checkpoint_path[1]  # epoch number from find_best_checkpoint
    if checkpoint_path[2] is not None:
        config['best_bleu'] = checkpoint_path[2]  # BLEU score
    
    # Load tokenizer to save with models
    print("Loading tokenizer...")
    tokenizer = MarianTokenizer.from_pretrained(config['tokenizer'])
    
    # Build training model
    print("Building training model architecture...")
    if model_type == 'lstm':
        training_model = build_bidirectional_model(
            config['vocab_size'],
            config['max_encoder_len'],
            config['max_decoder_len'],
            latent_dim=config['latent_dim']
        )
    else:  # attention
        training_model = build_attention_model(
            config['vocab_size'],
            config['max_encoder_len'],
            config['max_decoder_len'],
            latent_dim=config['latent_dim']
        )
    
    # Load checkpoint weights
    print(f"Loading checkpoint weights from {checkpoint_path[0].name}...")
    training_model.load_weights(str(checkpoint_path[0]))
    
    # Build inference models
    print("Building inference models...")
    if model_type == 'lstm':
        encoder_model, decoder_model = build_inference_models_lstm(
            training_model,
            latent_dim=config['latent_dim']
        )
    else:  # attention
        encoder_model, decoder_model = build_inference_models_attention(
            training_model,
            config['max_encoder_len'],
            latent_dim=config['latent_dim']
        )
    
    # Create model directory
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    
    # Save models as SavedModel
    print("\nSaving models as SavedModel format...")
    print("  - Saving training model...")
    training_model.save(str(model_path / 'training_model'))
    
    print("  - Saving encoder model...")
    encoder_model.save(str(model_path / 'encoder_model'))
    
    print("  - Saving decoder model...")
    decoder_model.save(str(model_path / 'decoder_model'))
    
    # Save tokenizer
    print("  - Saving tokenizer...")
    tokenizer.save_pretrained(str(model_path))
    
    # Save config
    print("  - Saving config...")
    with open(model_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Calculate sizes
    def get_dir_size(path):
        """Calculate total size of a directory in MB."""
        total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        return total_size / (1024*1024)
    
    training_size = get_dir_size(model_path / 'training_model')
    encoder_size = get_dir_size(model_path / 'encoder_model')
    decoder_size = get_dir_size(model_path / 'decoder_model')
    
    print(f"\nModels saved successfully:")
    print(f"  - training_model/: {training_size:.2f} MB")
    print(f"  - encoder_model/: {encoder_size:.2f} MB")
    print(f"  - decoder_model/: {decoder_size:.2f} MB")
    
    return config


def create_model_card(model_config, model_name):
    """Generate a model card (README.md) for the model."""
    
    # Load template from file
    template_path = Path(__file__).parent / 'model_cards' / 'model_card_template.md'
    with open(template_path, 'r') as f:
        template = f.read()
    
    # Replace placeholders with actual values
    model_card = template.replace('{{description}}', model_config['description'])
    model_card = model_card.replace('{{architecture}}', model_config['architecture'])
    model_card = model_card.replace('{{repo_id}}', model_config['repo_id'])
    model_card = model_card.replace('{{model_name}}', model_name)
    
    return model_card


def upload_model(model_name, token, force=False):
    """Upload a model to Hugging Face Hub."""
    
    if model_name not in MODELS:
        print(f"ERROR: Unknown model '{model_name}'. Choose from: {list(MODELS.keys())}")
        return False
    
    model_config = MODELS[model_name]
    
    # Check if SavedModel already exists
    savedmodel_exists = check_savedmodel_exists(model_config['model_dir'])
    
    if savedmodel_exists and not force:
        print(f"\nSavedModel already exists in {model_config['model_dir']}")
        print("  Using existing models.")
        
        # Load config from existing file
        config_path = Path(model_config['model_dir']) / 'config.json'
        with open(config_path) as f:
            config = json.load(f)
        config['description'] = model_config['description']
        config['repo_id'] = model_config['repo_id']
    
    elif not savedmodel_exists or force:
        # Need to build from checkpoint
        print(f"\n{'='*60}")
        print(f"Preparing {model_name} model for upload")
        print(f"{'='*60}")
        
        # Find best checkpoint (by BLEU score)
        print(f"\nSearching for best checkpoint...")
        print(f"  Metrics dir: {model_config['metrics_dir']}")
        print(f"  Checkpoint dir: {model_config['checkpoint_dir']}")
        
        checkpoint_path, epoch, bleu_score = find_best_checkpoint(
            model_config['metrics_dir'],
            model_config['checkpoint_dir']
        )
        
        if checkpoint_path is None:
            print(f"ERROR: No checkpoints found in {model_config['checkpoint_dir']}")
            print(f"Please train the model first (run notebook 01 or 02).")
            return False
        
        if bleu_score is not None:
            print(f"Found best checkpoint: {checkpoint_path.name}")
            print(f"  Epoch: {epoch}, BLEU score: {bleu_score:.2f}")
        else:
            print(f"Found checkpoint: {checkpoint_path.name} (epoch {epoch})")
            print(f"  Note: BLEU score not available")
        
        # Build models from checkpoint
        config = build_models_from_checkpoint(
            (checkpoint_path, epoch, bleu_score),
            model_config['model_type'],
            model_config['model_dir']
        )
        config['description'] = model_config['description']
        config['repo_id'] = model_config['repo_id']
    
    # Calculate directory sizes
    model_path = Path(model_config['model_dir'])
    
    def get_dir_size(path):
        """Calculate total size of a directory in MB."""
        total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        return total_size / (1024*1024)
    
    training_size = get_dir_size(model_path / 'training_model')
    encoder_size = get_dir_size(model_path / 'encoder_model')
    decoder_size = get_dir_size(model_path / 'decoder_model')
    total_size = training_size + encoder_size + decoder_size
    
    print(f"\n{'='*60}")
    print(f"Uploading {model_name} model to Hugging Face Hub")
    print(f"{'='*60}")
    print(f"Repository: {model_config['repo_id']}")
    print(f"Total size: {total_size:.2f} MB")
    
    try:
        # Initialize Hugging Face API
        api = HfApi()
        
        # Create repository if it doesn't exist
        print(f"\nCreating/updating repository...")
        create_repo(
            repo_id=model_config['repo_id'],
            token=token,
            exist_ok=True,
            repo_type='model'
        )
        
        # Generate and upload model card
        print(f"Generating model card...")
        model_card = create_model_card(config, model_name)
        
        # Write README to model directory (will be uploaded with folder)
        readme_path = model_path / 'README.md'
        with open(readme_path, 'w') as f:
            f.write(model_card)
        
        # Upload entire model folder (includes encoder, decoder, config, tokenizer, README)
        print(f"Uploading model files (this may take a few minutes)...")
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=model_config['repo_id'],
            token=token,
            repo_type='model'
        )
        
        print(f"\nSuccessfully uploaded {model_name} model!")
        print(f"  View at: https://huggingface.co/{model_config['repo_id']}")
        
        return True
        
    except Exception as e:
        print(f"\nError uploading model: {e}")
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
        status = "Success" if success_lstm else "Failed"
        print(f"LSTM model: {status}")
    
    if args.model in ['attention', 'both']:
        status = "Success" if success_attention else "Failed"
        print(f"Attention model: {status}")
    
    if (args.model == 'both' and success_lstm and success_attention) or \
       (args.model == 'lstm' and success_lstm) or \
       (args.model == 'attention' and success_attention):
        print("\nAll uploads completed successfully!")
    else:
        print("\nSome uploads failed. Check error messages above.")


if __name__ == '__main__':
    main()