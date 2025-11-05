import streamlit as st
import joblib
import os

@st.cache_resource
def load_rf():
    try:
        return joblib.load("model_randomforest.pkl")
    except Exception as e:
        st.error(f"Gagal memuat RandomForest: {type(e).__name__}: {e}")
        raise

@st.cache_resource
def load_lr():
    try:
        return joblib.load("model_linear.pkl")
    except Exception as e:
        st.error(f"Gagal memuat Linear Regression: {type(e).__name__}: {e}")
        raise

@st.cache_resource
def load_lstm():
    # import TensorFlow hanya saat perlu
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    try:
        return load_model("model_lstm.h5")
    except Exception as e:
        st.error(f"Gagal memuat LSTM: {type(e).__name__}: {e}")
        raise

# HAPUS pemanggilan load_models() saat import
# Panggil loader sesuai pilihan user di UI:
model_name = st.selectbox("Pilih model", ["Random Forest", "Linear Regression", "LSTM"])
if model_name == "Random Forest":
    model = load_rf()
elif model_name == "Linear Regression":
    model = load_lr()
else:
    model = load_lstm()
