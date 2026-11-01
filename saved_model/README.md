---
language: en
license: mit
library_name: tensorflow
pipeline_tag: text-generation
tags:
  - character-level
  - text-generation
  - gru
  - rnn
  - tensorflow
  - keras
  - news
---

# Character-Level News Writer (GRU)

A small GRU language model (1M parameters) that writes **one character at a time**. It was never given a word list: it learned spelling, spacing, punctuation and the shape of a news story from ~2,700 Pakistani business and sports news articles.

**Try it:** [Character-level News Writer](https://huggingface.co/spaces/Ismailalshofi/char-news-writer), a Space where the model runs in your browser (the GRU re-implemented in plain JavaScript, checked against TensorFlow) · **Code:** [GitHub](https://github.com/ismailalshofi/character-based-language-model)

## Samples

Real outputs, temperature 0.5 and 0.8:

> **HONG KONG:** Asian stocks started on the final over as dealers could be able to see if they will allow the first day of the last settlement. The country's cricket team has recovered from a more than half the second set to beat South Africa...

> **KARACHI:** Former West Indies Bailey delivery immossed the winner in 2052, and then been planned to the statement on Friday despite himself to consolidate their production cuts in a new target of 127 for six...

The style is right: datelines, prices, scores, team names, "said on Monday". The facts are not, and business and sports get mixed in the same sentence, because the model only ever saw letters, never meaning.

## Model

| Component | Details |
|---|---|
| Embedding | 106 characters → 128-dim vectors |
| GRU | 512 units, dropout 0.2 |
| Dense | 512 → 106 logits |
| Parameters | 1,054,058 (4 MB) |

## Training

| | |
|---|---|
| Data | 2,692 news articles (1,284 business, 1,408 sports), 4.48M characters after cleaning |
| Cleaning | HTML and URLs removed, smart quotes and dashes normalized, whitespace collapsed. Case, digits and punctuation kept |
| Sequences | 100 characters, 90/10 train/validation split |
| Optimizer | Adam, batch 128, 30 epochs |
| Final validation loss | **1.096** (≈ 1.58 bits per character), accuracy 67.4% |

The validation loss was still going down at epoch 30, so early stopping never triggered: more epochs would likely help.

![Training curves](training_plot.png)

## Usage

The model is a subclassed Keras model, so loading it needs the `CharModel` class. This is the complete code (TensorFlow 2.15):

```python
import json

import tensorflow as tf
from huggingface_hub import hf_hub_download

REPO_ID = "Ismailalshofi/char-based-language-model"


class CharModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units, dropout_rate=0.2):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.gru = tf.keras.layers.GRU(
            rnn_units, return_sequences=True, return_state=True, dropout=dropout_rate, recurrent_dropout=0.0
        )
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = self.embedding(inputs, training=training)
        x = self.dropout(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)
        return (x, states) if return_state else x


# local_dir keeps the original file names (on Windows the hub cache uses symlinks TensorFlow can't read)
files = {name: hf_hub_download(REPO_ID, name, local_dir="model") for name in ("model.keras", "vocab.json")}
with open(files["vocab.json"], encoding="utf-8") as f:
    vocab = json.load(f)
get_ids = tf.keras.layers.StringLookup(vocabulary=vocab, mask_token=None)
get_chars = tf.keras.layers.StringLookup(vocabulary=get_ids.get_vocabulary(), invert=True, mask_token=None)
model = tf.keras.models.load_model(files["model.keras"], custom_objects={"CharModel": CharModel}, compile=False)

# Never sample the [UNK] token
unk_mask = tf.one_hot(get_ids(["[UNK]"])[0], len(get_ids.get_vocabulary()), on_value=-1e9, off_value=0.0)


def generate(seed, length=300, temperature=0.8):
    ids = get_ids(tf.strings.unicode_split([seed], "UTF-8")).to_tensor()
    states, text = None, seed
    for _ in range(length):
        logits, states = model(ids, states=states, return_state=True)
        ids = tf.random.categorical(logits[:, -1, :] / temperature + unk_mask, num_samples=1)
        text += get_chars(ids)[0, 0].numpy().decode("utf-8")
    return text


print(generate("KARACHI:"))
```

**Temperature:** around 0.5 is safe and repetitive, 0.8 is a good balance, above 1.0 the model starts inventing words.

## Limitations

- **Style, not facts.** It writes text that looks like news. Every name, number and event in it is made up.
- **Narrow domain.** Pakistani business and sports news from one source. Other topics or seeds produce weaker text.
- **Encoding artifacts.** Some apostrophes in the source data were stored as `´` (e.g. `Lord´s`), and the model learned them.
- **English only**, 106 characters. Anything outside that vocabulary becomes `[UNK]`.

## Related

This model is the starting point of a [controlled study of character vs subword tokenization](https://github.com/ismailalshofi/tokenization-study), which trains the same Transformer with both tokenizers and compares them fairly in bits per character.

## Author

**Ismail Alshofi** · [GitHub](https://github.com/ismailalshofi) · [Kaggle](https://www.kaggle.com/ismailalshofi) · [Medium](https://medium.com/@ialshofi)
