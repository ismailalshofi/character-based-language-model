"""Export the trained GRU to plain files the browser demo can read.

    python space/export_weights.py

Writes space/weights.bin (all weights as little-endian float32, back to back) and
space/model.json (the vocabulary and where each weight sits in the .bin file).
It also re-implements one forward pass in NumPy and checks it against TensorFlow,
because the JavaScript in index.html does exactly the same maths.
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent


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


def gru_numpy(ids, w):
    """Keras GRU with reset_after=True, gates in the order z, r, h."""
    units = w["recurrent_kernel"].shape[0]
    h = np.zeros(units, dtype=np.float32)
    for i in ids:
        x = w["embedding"][i]
        xg = x @ w["kernel"] + w["bias"][0]
        hg = h @ w["recurrent_kernel"] + w["bias"][1]
        z = 1 / (1 + np.exp(-(xg[:units] + hg[:units])))
        r = 1 / (1 + np.exp(-(xg[units : 2 * units] + hg[units : 2 * units])))
        hh = np.tanh(xg[2 * units :] + r * hg[2 * units :])
        h = z * h + (1 - z) * hh
    return h @ w["dense_kernel"] + w["dense_bias"]


def main():
    saved = ROOT / "saved_model"
    model = tf.keras.models.load_model(saved / "model.keras", custom_objects={"CharModel": CharModel}, compile=False)
    vocab = json.loads((saved / "vocab.json").read_text(encoding="utf-8"))
    lookup = tf.keras.layers.StringLookup(vocabulary=vocab, mask_token=None)
    # Index 0 is [UNK], then the characters in training order
    chars = lookup.get_vocabulary()

    assert model.gru.cell.reset_after, "the browser code assumes reset_after=True"
    kernel, recurrent_kernel, bias = model.gru.get_weights()
    dense_kernel, dense_bias = model.dense.get_weights()
    weights = {
        "embedding": model.embedding.get_weights()[0],
        "kernel": kernel,
        "recurrent_kernel": recurrent_kernel,
        "bias": bias,
        "dense_kernel": dense_kernel,
        "dense_bias": dense_bias,
    }

    # Check the NumPy maths against TensorFlow before trusting it
    seed = "KARACHI: Oil prices rose on Monday"
    ids = lookup(tf.strings.unicode_split(seed, "UTF-8")).numpy()
    expected = model(ids[None, :])[0, -1].numpy()
    got = gru_numpy(ids, weights)
    error = float(np.abs(expected - got).max())
    print(f"NumPy vs TensorFlow, max logit difference: {error:.2e}")
    assert error < 1e-3

    layout, offset, blobs = {}, 0, []
    for name, array in weights.items():
        array = np.ascontiguousarray(array, dtype="<f4")
        layout[name] = {"offset": offset, "shape": list(array.shape)}
        offset += array.size
        blobs.append(array.tobytes())
    (OUT / "weights.bin").write_bytes(b"".join(blobs))
    (OUT / "model.json").write_text(
        json.dumps({"chars": chars, "weights": layout, "check": {"seed": seed, "logits": expected.tolist()}}),
        encoding="utf-8",
    )
    print(f"Wrote weights.bin ({offset:,} floats) and model.json ({len(chars)} characters)")


if __name__ == "__main__":
    main()
