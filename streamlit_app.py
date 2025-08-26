import streamlit as st
import predict
import eda

# ----------- Styling simple -----------
st.markdown("""
    <style>
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #f1f5f9; /* biru muda pastel */
        }
        }
        h1, h2, h3 {
            color: #e6e6e6;
        }
        .sidebar-title {
            font-size: 20px;
            font-weight: bold;
            background: linear-gradient(90deg, #6EE7F9, #A78BFA, #F472B6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .sidebar-sub {
            font-size: 14px;
            color: #a3a3a3;
        }
    </style>
""", unsafe_allow_html=True)

# ----------- Sidebar Navigation -----------
with st.sidebar:
    st.markdown("<div class='sidebar-title'>📂 Page Navigation</div>", unsafe_allow_html=True)
    page = st.selectbox("Pilih Halaman", ("EDA", "Predict Sentiment"))
    st.markdown("<div class='sidebar-sub'>Gunakan menu ini untuk berpindah halaman</div>", unsafe_allow_html=True)
    st.write("")
    st.write("**About**")
    st.write(
    "Halaman ini menampilkan hasil eksplorasi data teks serta "
    "model prediksi sentimen untuk mengklasifikasikan kalimat menjadi "
    "positive, neutral, atau negative."
)

if page == "EDA":
    eda.run()
else:
    predict.run()

