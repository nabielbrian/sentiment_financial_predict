# predict.py — FINAL (Keras 3 safe + classes.txt override + auto-detect fallback)
import json
import pathlib
import itertools
from typing import List, Optional

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from keras import ops as K
from keras import layers

# --------------------------
# PATHS
# --------------------------
MODEL_PATH = pathlib.Path("model_impr2.keras")
TV_CFG_PATH = pathlib.Path("tv_cfg_impr2.json")
VOCAB_PATH = pathlib.Path("vocab_impr2.txt")
CLASSES_FPATH = pathlib.Path("classes.txt")  # opsional: 1 label per baris

# Default kalau tidak ada classes.txt (hanya placeholder, akan dioverride auto-detect)
DEFAULT_CLASS_NAMES = ["negative", "neutral", "positive"]

# --------------------------
# HELPERS
# --------------------------
def _load_vocab(vocab_file: pathlib.Path) -> List[str]:
    if not vocab_file.exists():
        return []
    with vocab_file.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def _load_class_names_from_file() -> Optional[List[str]]:
    if CLASSES_FPATH.exists():
        with CLASSES_FPATH.open("r", encoding="utf-8") as f:
            names = [x.strip() for x in f if x.strip()]
            if names:
                return names
    return None

def _allow_tv_keys(d: dict) -> dict:
    allowed = {
        "max_tokens","standardize","split","ngrams",
        "output_mode","output_sequence_length",
        "pad_to_max_tokens","sparse"
    }
    return {k: v for k, v in d.items() if k in allowed}

def _get_expected_seq_len(model: tf.keras.Model) -> Optional[int]:
    ishape = model.input_shape
    if isinstance(ishape, list):  # multi-input
        ishape = ishape[0]
    try:
        return int(ishape[-1]) if ishape[-1] is not None else None
    except Exception:
        return None

# --------------------------
# LOADERS (cached)
# --------------------------
@st.cache_resource(show_spinner=False)
def load_base_model(model_path: pathlib.Path) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path, compile=False)

@st.cache_resource(show_spinner=False)
def load_text_vectorizer(
    cfg_path: pathlib.Path,
    vocab_path: pathlib.Path,
    force_seq_len: Optional[int] = None,
) -> TextVectorization:
    default_cfg = {
        "standardize": "lower_and_strip_punctuation",
        "split": "whitespace",
        "output_mode": "int",
        "output_sequence_length": 40,  # akan dioverride oleh force_seq_len bila ada
        "sparse": False,
    }
    cfg = {}
    if cfg_path.exists():
        try:
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
            cfg = raw.get("config", raw) if isinstance(raw, dict) else {}
        except Exception:
            cfg = {}

    merged = {**default_cfg, **cfg}
    if force_seq_len is not None:
        merged["output_sequence_length"] = int(force_seq_len)

    tv = TextVectorization(**_allow_tv_keys(merged))
    vocab = _load_vocab(vocab_path)
    if vocab:
        tv.set_vocabulary(vocab)
    return tv

@st.cache_resource(show_spinner=False)
def build_inference_model() -> tf.keras.Model:
    base = load_base_model(MODEL_PATH)
    exp_len = _get_expected_seq_len(base)  # misal: 48

    tv = load_text_vectorizer(TV_CFG_PATH, VOCAB_PATH, force_seq_len=exp_len)

    # input: 1 string per contoh → TV keluarkan token ids (batch, seq_len)
    text_in = tf.keras.Input(shape=(), dtype=tf.string, name="text")
    ids = tv(text_in)                     # (None, seq_len)
    ids = K.cast(ids, "int32")            # Keras 3-safe cast

    logits = base(ids)
    if isinstance(logits, (list, tuple)):  # jaga-jaga kalau multi-output
        logits = logits[0]

    # Tentukan jumlah kelas & putuskan perlu softmax atau tidak
    try:
        last_dim = logits.shape[-1]
        num_classes = int(last_dim) if last_dim is not None else None
    except Exception:
        num_classes = None

    # Kalau bentuknya jelas klasifikasi (>=2 kelas), apply softmax
    if (num_classes is not None) and (num_classes >= 2) and (num_classes <= 100):
        probs = layers.Activation("softmax", name="probs")(logits)
    else:
        probs = logits  # kalau model sudah memberi probabilitas lain

    return tf.keras.Model(text_in, probs, name="inference_model")

# --------------------------
# AUTO-DETECT LABEL ORDER (fallback jika tidak ada classes.txt)
# --------------------------
def _autodetect_label_order(model: tf.keras.Model) -> List[str]:
    # jangkar sederhana untuk setiap sentimen
    anchors = {
        "positive": [
            "Customer demand is strong and margins improved this quarter.",
            "The company reported record profit and higher sales.",
            "Earnings beat expectations and guidance was raised."
        ],
        "negative": [
            "Revenue decreased sharply and the company posted a loss.",
            "Margins deteriorated and demand weakened.",
            "The outlook was cut after disappointing results."
        ],
        "neutral": [
            "The annual meeting was held on Monday.",
            "Management presented the quarterly report.",
            "The company announced its new product line."
        ],
    }

    def mean_probs(texts):
        arr = np.asarray(texts, dtype=object)
        p = model.predict(arr, verbose=0)  # (n, C)
        return p.mean(axis=0)              # (C,)

    pos_m = mean_probs(anchors["positive"])
    neg_m = mean_probs(anchors["negative"])
    neu_m = mean_probs(anchors["neutral"])

    # coba semua permutasi assignment label untuk index 0..C-1
    C = pos_m.shape[-1]
    best = None
    for labels_by_index in itertools.permutations(["negative","neutral","positive"], C):
        idx_pos = labels_by_index.index("positive")
        idx_neg = labels_by_index.index("negative")
        idx_neu = labels_by_index.index("neutral")
        score = pos_m[idx_pos] + neg_m[idx_neg] + neu_m[idx_neu]
        if (best is None) or (score > best[0]):
            best = (score, list(labels_by_index))
    return best[1] if best else DEFAULT_CLASS_NAMES

# --------------------------
# INFERENCE
# --------------------------
def _predict_one(model: tf.keras.Model, text: str):
    arr = np.asarray([text], dtype=object)
    preds = model.predict(arr, verbose=0)[0]
    return preds  # 1D array panjang C

# --------------------------
# STREAMLIT PAGE
# --------------------------
def run():
    st.header("🔮 Predict Sentiment (Positive / Neutral / Negative)")

    # Build model (once, cached)
    model = build_inference_model()

    # Tentukan urutan label final:
    src = "auto-detect"
    class_names = _load_class_names_from_file()
    if class_names is None:
        class_names = _autodetect_label_order(model)
    else:
        src = "classes.txt"

    with st.expander("⚙️ Assets & Config", expanded=False):
        st.write("- Model:", f"`{MODEL_PATH.name}`")
        st.write("- TV config:", f"`{TV_CFG_PATH.name}`")
        st.write("- Vocabulary:", f"`{VOCAB_PATH.name}`")
        st.write(f"- Label order: `{class_names}` (source: {src})")
        st.caption("Jika ingin mengunci urutan label, buat file `classes.txt` (satu label per baris).")

    text = st.text_area(
        "Masukkan teks/ulasan:",
        height=140,
        placeholder="contoh: Customer demand is strong and margins improved this quarter."
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        show_prob = st.checkbox("Tampilkan probabilitas kelas", value=True)
    with c2:
        predict_btn = st.button("Predict", use_container_width=True)

    if predict_btn:
        if not text.strip():
            st.warning("Masukkan teks terlebih dahulu.")
            return

        try:
            probs = _predict_one(model, text.strip())
            idx = int(np.argmax(probs))
            label = class_names[idx]
        except Exception as e:
            st.error("Gagal menjalankan inference.")
            st.exception(e)
            return

        st.success(f"Hasil prediksi: **{label}**")
        st.caption(f"debug: argmax index = {idx} | order = {class_names} | src = {src}")

        if show_prob and probs is not None and len(probs) == len(class_names):
            prob_dict = {cls: float(p) for cls, p in zip(class_names, probs)}
            st.write(prob_dict)
            st.bar_chart(prob_dict)

if __name__ == "__main__":
    run()
