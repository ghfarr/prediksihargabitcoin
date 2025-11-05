import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model  # ✅ penting untuk TF 2.13
from sklearn.metrics import mean_absolute_error, r2_score

# =====================================================
# Streamlit Page Setup (WAJIB paling atas)
# =====================================================
st.set_page_config(page_title="Bitcoin Price Predictor", layout="wide")

# =====================================================
# Load Models Once (Caching)
# =====================================================
@st.cache_resource
def load_models():
    rf = joblib.load("model_randomforest.pkl")
    lr = joblib.load("model_linear.pkl")
    lstm = load_model("model_lstm.h5")   # ✅ Aman di TF baru
    return rf, lr, lstm

rf_model, lr_model, lstm_model = load_models()

# =====================================================
# UI
# =====================================================
st.title("🔮 Prediksi Harga Bitcoin Menggunakan Machine Learning & Deep Learning")
st.markdown("""
Aplikasi ini memprediksi harga Bitcoin berdasarkan data historis menggunakan tiga model:
- **Random Forest**
- **Linear Regression**
- **LSTM (Deep Learning)**

Upload dataset CSV untuk melihat hasil prediksi.
""")

# =====================================================
# Upload Dataset
# =====================================================
uploaded_file = st.file_uploader("📂 Upload file CSV data harga Bitcoin", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Data yang Diupload")
    st.dataframe(df.head())

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    if all(col in df.columns for col in required_cols):

        X = df[["Open", "High", "Low", "Volume"]].values
        y_actual = df["Close"].values

        # =====================================================
        # Prediksi
        # =====================================================
        y_pred_rf = rf_model.predict(X)
        y_pred_lr = lr_model.predict(X)

        # LSTM needs 3D input
        X_lstm = np.reshape(X, (X.shape[0], 1, X.shape[1]))
        y_pred_lstm = lstm_model.predict(X_lstm).flatten()

        # =====================================================
        # Grafik
        # =====================================================
        st.subheader("📊 Perbandingan Hasil Prediksi (100 Data Terakhir)")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_actual[-100:], label="Actual", linewidth=2)
        ax.plot(y_pred_rf[-100:], label="Random Forest")
        ax.plot(y_pred_lr[-100:], label="Linear Regression")
        ax.plot(y_pred_lstm[-100:], label="LSTM")
        ax.set_title("Prediksi Harga Bitcoin")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Harga (USD)")
        ax.legend()
        st.pyplot(fig)

        # =====================================================
        # Evaluasi Model
        # =====================================================
        st.subheader("📈 Evaluasi Model")
        st.write(pd.DataFrame({
            "Model": ["Random Forest", "Linear Regression", "LSTM"],
            "MAE (Lebih kecil lebih baik)": [
                mean_absolute_error(y_actual, y_pred_rf),
                mean_absolute_error(y_actual, y_pred_lr),
                mean_absolute_error(y_actual, y_pred_lstm)
            ],
            "R² Score (Lebih besar lebih baik)": [
                r2_score(y_actual, y_pred_rf),
                r2_score(y_actual, y_pred_lr),
                r2_score(y_actual, y_pred_lstm)
            ]
        }))
    else:
        st.error(f"Dataset harus mengandung kolom: {', '.join(required_cols)}")
else:
    st.info("Silakan upload dataset CSV.")
