"""Try the character-level GRU language model in the browser."""

import json

import gradio as gr
import tensorflow as tf
from huggingface_hub import hf_hub_download

REPO_ID = "Ismail99/char-based-language-model"


class CharModel(tf.keras.Model):
    """Same class as in training. Keras needs it to rebuild the saved model."""

    def __init__(self, vocab_size, embedding_dim, rnn_units, dropout_rate=0.2):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.gru = tf.keras.layers.GRU(
            rnn_units,
            return_sequences=True,
            return_state=True,
            dropout=dropout_rate,
            recurrent_dropout=0.0,
        )
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = self.embedding(inputs, training=training)
        x = self.dropout(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)
        if return_state:
            return x, states
        return x


def load():
    # A plain local folder, not the hub cache: on Windows the cache uses symlinks,
    # which TensorFlow can't read through
    files = {name: hf_hub_download(REPO_ID, name, local_dir="model") for name in ("model.keras", "vocab.json")}
    with open(files["vocab.json"], encoding="utf-8") as f:
        vocab = json.load(f)
    get_ids = tf.keras.layers.StringLookup(vocabulary=vocab, mask_token=None)
    get_chars = tf.keras.layers.StringLookup(vocabulary=get_ids.get_vocabulary(), invert=True, mask_token=None)
    model = tf.keras.models.load_model(files["model.keras"], custom_objects={"CharModel": CharModel}, compile=False)
    return model, get_ids, get_chars


MODEL, GET_IDS, GET_CHARS = load()
# Never sample the [UNK] token
UNK_MASK = tf.one_hot(GET_IDS(["[UNK]"])[0], len(GET_IDS.get_vocabulary()), on_value=-1e9, off_value=0.0)


def generate(seed, length, temperature):
    """Stream the text as it is written, one character at a time."""
    seed = seed or "\n"
    ids = GET_IDS(tf.strings.unicode_split([seed], "UTF-8")).to_tensor()
    states = None
    text = seed
    for i in range(int(length)):
        logits, states = MODEL(ids, states=states, return_state=True)
        logits = logits[:, -1, :] / temperature + UNK_MASK
        next_id = tf.random.categorical(logits, num_samples=1)
        text += GET_CHARS(next_id)[0, 0].numpy().decode("utf-8")
        ids = next_id
        if i % 5 == 0:
            yield text
    yield text


with gr.Blocks(title="Character-level news writer") as demo:
    gr.Markdown(
        "# Character-level news writer\n"
        "A small GRU (1M parameters) that writes **one character at a time**. It never saw a word list: "
        "it learned spelling, spacing and punctuation from ~2,700 Pakistani business and sports news articles.\n\n"
        "Give it the start of a dateline or a sentence, then press **Write**. "
        "Low temperature is safe and repetitive, high temperature is creative and full of made-up words.\n\n"
        "[Model card](https://huggingface.co/Ismail99/char-based-language-model) · "
        "[Code on GitHub](https://github.com/ismailalshofi/character-based-language-model)"
    )
    with gr.Row():
        with gr.Column():
            seed = gr.Textbox(label="Start with", value="KARACHI:")
            length = gr.Slider(50, 600, value=300, step=10, label="Characters to write")
            temperature = gr.Slider(0.2, 1.5, value=0.8, step=0.1, label="Temperature")
            button = gr.Button("Write", variant="primary")
        output = gr.Textbox(label="Generated text", lines=12)
    gr.Examples(
        [["KARACHI:", 300, 0.8], ["Oil prices", 300, 0.8], ["HONG KONG:", 300, 0.5], ["The match", 300, 1.1]],
        inputs=[seed, length, temperature],
    )
    button.click(generate, inputs=[seed, length, temperature], outputs=output)

if __name__ == "__main__":
    demo.launch()
