# eda.py
import re
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from collections import Counter

# ---------------------------
# fungsi pendukung
# ---------------------------
STOPWORDS = {
    "the","and","to","of","in","on","for","a","an","is","are","was","were","be","been","being",
    "this","that","it","as","at","by","from","with","or","but","not","no","so","if","then","than",
    "into","over","under","up","down","out","very","can","could","should","would","will","may","might"
}

def tokenize(text):
    return [t.lower() for t in re.findall(r"[A-Za-z]+", str(text)) if t]

def top_tokens(series_text, n=10):
    c = Counter()
    for t in series_text:
        toks = [w for w in tokenize(t) if w not in STOPWORDS]
        c.update(toks)
    return pd.Series(dict(c.most_common(n)))

def tokenize_words(text):
    return [t.lower() for t in re.findall(r"[A-Za-z]+", str(text)) if t]

def get_bigrams(tokens):
    return [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]

def top_bigrams(series_text, n=10):
    c = Counter()
    for t in series_text:
        toks = [w for w in tokenize_words(t) if w not in STOPWORDS]
        bigs = get_bigrams(toks)
        c.update(bigs)
    return pd.Series(dict(c.most_common(n)))

# ---------------------------
# streamlit page
# ---------------------------
def run():
    st.header("📊 Exploratory Data Analysis (EDA)")

    # load dataset
    df = pd.read_csv("data.csv")
    st.subheader("Preview Dataset")
    st.write(df.head())

    df_vis = df.copy()

    # -----------------------------------
    # 1. Distribusi jumlah data per label
    # -----------------------------------
    st.subheader("1️⃣ Distribusi Label Sentimen")
    counts = df_vis["Sentiment"].value_counts()
    fig, ax = plt.subplots()
    counts.plot(kind="bar", color=["#66c2a5","#fc8d62","#8da0cb"], ax=ax)
    ax.set_title("Distribusi Label Sentimen")
    ax.set_xlabel("Label")
    ax.set_ylabel("Jumlah")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    st.pyplot(fig)

    st.markdown("""
    **Analisis:**  
    - Neutral mendominasi dataset (52%), Positive 31%, Negative hanya 14%.  
    - Terjadi ketidakseimbangan kelas yang signifikan.  
    - Model bisa bias ke kelas neutral → gunakan F1-macro untuk evaluasi dan pertimbangkan balancing techniques.
    """)

    # -----------------------------------
    # 2. Distribusi panjang teks
    # -----------------------------------
    st.subheader("2️⃣ Distribusi Panjang Teks (Word Count)")
    df_vis["word_count"] = df_vis["Sentence"].astype(str).str.split().str.len()
    fig, ax = plt.subplots(figsize=(7,4))
    ax.hist(df_vis["word_count"], bins=40, color="#66c2a5")
    ax.set_title("Distribusi Panjang Teks (Word Count)")
    ax.set_xlabel("Jumlah Kata per Kalimat")
    ax.set_ylabel("Frekuensi")
    st.pyplot(fig)
    st.markdown("""
    **Analisis:**  
    - Mayoritas teks memiliki panjang 10–25 kata.  
    - Puncak distribusi ada di 15–20 kata.  
    - Teks di atas 30 kata semakin jarang, hanya sedikit yang lebih dari 50 kata.  
    - Distribusi condong ke kanan: sebagian besar kalimat relatif singkat, sedangkan kalimat panjang adalah minoritas.
    """)

    # -----------------------------------
    # 3. Perbedaan panjang teks per label
    # -----------------------------------
    st.subheader("3️⃣ Distribusi Panjang Teks per Label")
    fig, ax = plt.subplots(figsize=(6,4))
    df_vis.boxplot(column="word_count", by="Sentiment", grid=False, ax=ax)
    ax.set_title("Distribusi Panjang Teks per Label")
    ax.set_xlabel("Label")
    ax.set_ylabel("Jumlah Kata")
    plt.suptitle("")
    st.pyplot(fig)
    st.markdown("""
    **Analisis:**  
    - Negative: median ~16 kata, mayoritas 12–25 kata, ada outlier >50 kata.  
    - Neutral: median ~20 kata, variasi paling lebar, ada outlier sampai >80 kata.  
    - Positive: median ~17 kata, mirip negative, ada outlier >50 kata.  
    - Secara umum, teks neutral cenderung lebih panjang, sementara positive & negative relatif lebih pendek.
    """)


    # -----------------------------------
    # 4. Token paling sering per label
    # -----------------------------------
    st.subheader("4️⃣ Token (Kata) Dominan per Label")
    labels = df_vis["Sentiment"].unique().tolist()
    for lab in labels:
        subset = df_vis.loc[df_vis["Sentiment"] == lab, "Sentence"]
        tt = top_tokens(subset, n=10)
        fig, ax = plt.subplots(figsize=(6,4))
        tt.sort_values().plot(kind="barh", color="#8da0cb", ax=ax)
        ax.set_title(f"Top Tokens - Label: {lab}")
        ax.set_xlabel("Frekuensi")
        ax.set_ylabel("Token")
        st.pyplot(fig)
    st.markdown("""
    **Analisis (Positive):**  
    - Token paling dominan adalah *eur*, menunjukkan konteks finansial.  
    - Kata lain seperti *sales*, *company*, *profit*, *net*, dan *year* sering muncul pada teks positif.  
    - Hal ini menegaskan bahwa kalimat positif umumnya berkaitan dengan kinerja keuangan, keuntungan, dan pertumbuhan perusahaan.
    """)

    # -----------------------------------
    # 5. Bigram paling sering per label
    # -----------------------------------
    st.subheader("5️⃣ Bigram (Pasangan Kata) Dominan per Label")
    for lab in labels:
        subset = df_vis.loc[df_vis["Sentiment"] == lab, "Sentence"]
        tb = top_bigrams(subset, n=10)
        fig, ax = plt.subplots(figsize=(6,4))
        tb.sort_values().plot(kind="barh", color="#fc8d62", ax=ax)
        ax.set_title(f"Top Bigrams - Label: {lab}")
        ax.set_xlabel("Frekuensi")
        ax.set_ylabel("Bigram")
        st.pyplot(fig)
    st.markdown("""
    **Analisis Bigram:**  
    - **Positive** → didominasi (eur, mn), (net, sales), (operating, profit); menandakan keberhasilan penjualan & profit.  
    - **Negative** → muncul (decreased, eur), (corresponding, period) selain bigram finansial umum; menggambarkan konteks penurunan.  
    - **Neutral** → fokus pada bigram faktual seperti (eur, mn), (eur, million), (company, s); cenderung informatif tanpa sentimen kuat.
    """)

