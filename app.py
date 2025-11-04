import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

#LOAD MODELS
rf_model = joblib.load('model_randomforest.pkl')
lr_model = joblib.load('model_linear.pkl')
lstm_model = tf.keras.models.load_model('model_lstm.h5')

st.set_page_config(page_title="Bitcoin Price Predictor", layout="wide")

st.title("Bitcoin Price Prediction Dashboard")
st.markdown("Prediksi harga Bitcoin berdasarkan data historis menggunakan **Machine Learning & Deep Learning**.")

# FILE UPLOAD
uploaded_file = st.file_uploader("Upload file CSV data harga Bitcoin", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data yang Diupload")
    st.write(df.head())

    if all(col in df.columns for col in ["Open", "High", "Low", "Close", "Volume"]):
        # Pilihan model
        model_choice = st.selectbox("Pilih model prediksi:", 
                                    ["Random Forest", "Linear Regression", "LSTM"])

        # Preprocessing
        X = df[["Open", "High", "Low", "Volume"]].values
        y_actual = df["Close"].values

        # Prediksi
        if model_choice == "Random Forest":
            y_pred = rf_model.predict(X)
        elif model_choice == "Linear Regression":
            y_pred = lr_model.predict(X)
        else:
            y_pred = lstm_model.predict(X.reshape((X.shape[0], X.shape[1], 1))).flatten()

        # Visualisasi
        st.subheader("Hasil Prediksi")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_actual[-30:], label="Actual", color="blue")
        ax.plot(y_pred[-30:], label=f"Predicted ({model_choice})", color="red")
        ax.set_title("Perbandingan Harga Aktual vs Prediksi (30 sampel terakhir)")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Harga Bitcoin (USD)")
        ax.legend()
        st.pyplot(fig)

        # Unduh hasil prediksi
        output = pd.DataFrame({
            "Actual": y_actual,
            "Predicted": y_pred
        })
        csv = output.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download hasil prediksi (CSV)", csv, "hasil_prediksi.csv", "text/csv")

    else:
        st.error("❌ Kolom tidak lengkap. Pastikan kolom: Open, High, Low, Close, Volume ada di file CSV kamu.")
else:
    st.info("Silakan upload file CSV terlebih dahulu untuk memulai prediksi.")
