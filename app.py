import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Load Models
# =====================================================
rf_model = joblib.load(os.path.join(os.getcwd(), 'model_randomforest.pkl'))
lr_model = joblib.load(os.path.join(os.getcwd(), 'model_linear.pkl'))
lstm_model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'model_lstm.h5'))

# =====================================================
# Streamlit Page Setup
# =====================================================
st.set_page_config(page_title="Bitcoin Price Predictor", layout="wide")
st.title("Prediksi Harga Bitcoin Menggunakan Machine Learning & Deep Learning")
st.markdown("""
Aplikasi ini memprediksi harga Bitcoin berdasarkan data historis menggunakan tiga model:
-  **Random Forest**  
-  **Linear Regression**  
-  **LSTM (Deep Learning)**  

Upload dataset CSV Anda untuk melihat hasil prediksi dan perbandingan antar model.
""")

# =====================================================
#  Upload Dataset
# =====================================================
uploaded_file = st.file_uploader("Upload file CSV data harga Bitcoin", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data yang Diupload")
    st.dataframe(df.head())

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    if all(col in df.columns for col in required_cols):
        X = df[["Open", "High", "Low", "Volume"]].values
        y_actual = df["Close"].values

        # =====================================================
        # 🔹 Prediksi
        # =====================================================
        y_pred_rf = rf_model.predict(X)
        y_pred_lr = lr_model.predict(X)

        # LSTM expects 3D input
        X_lstm = np.reshape(X, (X.shape[0], 1, X.shape[1]))
        y_pred_lstm = lstm_model.predict(X_lstm).flatten()

        # =====================================================
        # 🔹 Tampilkan Grafik
        # =====================================================
        st.subheader("📊 Perbandingan Hasil Prediksi")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_actual[-100:], label="Actual", color='blue')
        ax.plot(y_pred_rf[-100:], label="Random Forest", color='green')
        ax.plot(y_pred_lr[-100:], label="Linear Regression", color='orange')
        ax.plot(y_pred_lstm[-100:], label="LSTM", color='red')
        ax.set_title("Perbandingan Prediksi Harga Bitcoin (100 Sample Terakhir)")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Harga Bitcoin (USD)")
        ax.legend()
        st.pyplot(fig)

        # =====================================================
        # 🔹 Evaluasi Akurasi
        # =====================================================
        from sklearn.metrics import mean_absolute_error, r2_score

        mae_rf = mean_absolute_error(y_actual, y_pred_rf)
        mae_lr = mean_absolute_error(y_actual, y_pred_lr)
        mae_lstm = mean_absolute_error(y_actual, y_pred_lstm)

        r2_rf = r2_score(y_actual, y_pred_rf)
        r2_lr = r2_score(y_actual, y_pred_lr)
        r2_lstm = r2_score(y_actual, y_pred_lstm)

        st.subheader("📈 Evaluasi Model")
        st.write(pd.DataFrame({
            "Model": ["Random Forest", "Linear Regression", "LSTM"],
            "MAE (Error)": [mae_rf, mae_lr, mae_lstm],
            "R² Score": [r2_rf, r2_lr, r2_lstm]
        }))
    else:
        st.error(f"Dataset harus mengandung kolom: {', '.join(required_cols)}")
else:
    st.info("Silakan upload dataset CSV terlebih dahulu.")
