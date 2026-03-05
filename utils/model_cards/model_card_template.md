---
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

# {{description}}

This model was trained for English-to-French neural machine translation using the OPUS-100 dataset.

## Model details

- **Architecture:** {{architecture}}
- **Repository:** [gperdrizet/language-models](https://github.com/gperdrizet/language-models)
- **Training data:** OPUS-100 English-French parallel corpus (100,000 sentence pairs, filtered to <=20 tokens)
- **Tokenizer:** MarianTokenizer (Helsinki-NLP/opus-mt-en-fr)
- **Vocabulary size:** ~60,000 subword tokens
- **Latent dimension:** 256
- **Max sequence length:** 22 (encoder), 24 (decoder)

## Training configuration

- **Optimizer:** Adam (learning rate: 0.001)
- **Loss function:** Sparse categorical crossentropy
- **Batch size:** 32
- **Epochs:** 15
- **Validation split:** 10%

## Usage

### Loading and using the model for translation

```python
from huggingface_hub import snapshot_download
from transformers import MarianTokenizer
import tensorflow as tf
import os

# Download all model files to cache
model_path = snapshot_download(repo_id='{{repo_id}}')

# Load inference models (SavedModel format)
encoder_model = tf.keras.models.load_model(os.path.join(model_path, 'encoder_model'))
decoder_model = tf.keras.models.load_model(os.path.join(model_path, 'decoder_model'))

# Load tokenizer
tokenizer = MarianTokenizer.from_pretrained(model_path)

# Translate (requires translate function from the training repo)
# Example:
# from src import translate_lstm  # or translate_attention for attention model
# translation = translate_lstm(input_text, encoder_model, decoder_model, tokenizer, 22, 24)
```

### For deployment/web apps

Models are saved in TensorFlow SavedModel format for:
- **Production deployment** (TF Serving, TF Lite, TF.js)
- **Cross-version compatibility** across TensorFlow versions

### Fine-tuning for other language pairs

Load the training model to continue training or fine-tune:

```python
# Load training model
training_model = tf.keras.models.load_model(os.path.join(model_path, 'training_model'))

# Continue training with new data
training_model.fit(new_encoder_input, new_decoder_target, epochs=5)
```

This model can be fine-tuned for other European language pairs (e.g., English-German, English-Spanish) with minimal additional training.

See the accompanying fine-tuning notebook for a complete example.

## Limitations

- Trained only on short sentences (<=20 tokens)
- Performance degrades on longer sequences
- Best suited for European language pairs with similar syntax
- Uses greedy decoding (no beam search)

## Citation

If you use this model, please cite:

```
@misc{english-french-{{model_name}},
  author = {George Perdrizet},
  title = {English-French Neural Machine Translation},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/{{repo_id}}}},
}
```

## Model card authors

[George Perdrizet](https://github.com/gperdrizet)
