import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# Menonaktifkan pesan peringatan dari Pyplot
st.set_option('deprecation.showPyplotGlobalUse', False)

# Fungsi untuk memuat dan memproses data
@st.cache_data
def load_data(file_path):
    """Memuat data dari path file CSV."""
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df.set_index('Date', inplace=True)
    return df

# Fungsi untuk membuat dataset sekuensial untuk LSTM
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# Fungsi untuk memuat model yang sudah dilatih
@st.cache_resource
def load_prediction_model(model_path):
    """Memuat model Keras dari file .h5."""
    model = tf.keras.models.load_model(model_path)
    return model

# --- Konfigurasi Aplikasi Streamlit ---
st.set_page_config(page_title="Prediksi Harga Bitcoin", layout="wide")

# --- Judul dan Deskripsi ---
st.title("Aplikasi Prediksi Harga Bitcoin menggunakan LSTM")
st.markdown("""
Aplikasi ini menggunakan model *Long Short-Term Memory* (LSTM) untuk memprediksi harga penutupan (Close) Bitcoin untuk beberapa hari ke depan.
- **Sumber Data:** Menggunakan data historis harga Bitcoin dari Yahoo Finance.
- **Model:** LSTM yang dilatih pada data historis.
""")

# --- Sidebar untuk Input Pengguna ---
st.sidebar.header("Panel Kontrol")
st.sidebar.markdown("Pilih parameter di bawah ini untuk memulai prediksi.")

# Slider untuk memilih jumlah hari prediksi
n_days_to_predict = st.sidebar.slider("Jumlah Hari Prediksi", 1, 90, 30)
st.sidebar.info(f"Model akan memprediksi harga untuk **{n_days_to_predict} hari** ke depan.")

# --- Memuat Data dan Model ---
try:
    # Memuat data utama
    DATA_PATH = "BTC-USD.csv"
    df = load_data(DATA_PATH)
    
    # Memuat model
    MODEL_PATH = "model_lstm.h5" # Ganti dengan nama file model Anda
    model = load_prediction_model(MODEL_PATH)

    # --- Tampilan Utama ---
    
    # Menampilkan Data Mentah dalam expander
    with st.expander("Lihat Data Historis Bitcoin"):
        st.dataframe(df.tail()) # Menampilkan 5 baris terakhir

    # Visualisasi harga penutupan historis
    st.subheader("Grafik Harga Penutupan (Close) Historis")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Harga Penutupan'))
    fig_hist.update_layout(
        title="Pergerakan Harga Bitcoin (Historis)",
        xaxis_title="Tanggal",
        yaxis_title="Harga (USD)",
        template="plotly_white"
    )
    st.plotly_chart(fig_hist, use_container_width=True)


    # --- Proses Prediksi ---
    st.subheader(f"Hasil Prediksi untuk {n_days_to_predict} Hari ke Depan")

    # 1. Preprocessing data untuk prediksi
    closedf = df[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    closedf_scaled = scaler.fit_transform(np.array(closedf).reshape(-1, 1))

    # 2. Mengambil data 100 hari terakhir untuk memulai prediksi pertama
    time_step = 100
    x_input = closedf_scaled[-time_step:].reshape(1, -1)
    temp_input = list(x_input[0])

    # 3. Melakukan prediksi secara iteratif
    lst_output = []
    n_steps = time_step
    i = 0
    while(i < n_days_to_predict):
        if(len(temp_input) > time_step):
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1

    # 4. Mengembalikan hasil prediksi ke skala semula
    predicted_prices_scaled = np.array(lst_output)
    predicted_prices = scaler.inverse_transform(predicted_prices_scaled)

    # 5. Membuat tanggal untuk hasil prediksi
    last_date = df.index[-1]
    prediction_dates = pd.to_datetime([last_date + pd.DateOffset(days=x) for x in range(1, n_days_to_predict + 1)])

    # --- Menampilkan Hasil Prediksi ---

    # Membuat DataFrame untuk hasil prediksi
    df_pred = pd.DataFrame({'Tanggal': prediction_dates, 'Harga Prediksi': predicted_prices.flatten()})
    st.write("Tabel Harga Prediksi:")
    st.dataframe(df_pred)

    # Visualisasi Gabungan: Historis + Prediksi
    fig_pred = go.Figure()
    # Data historis (1 tahun terakhir untuk kejelasan)
    fig_pred.add_trace(go.Scatter(x=df.index[-365:], y=df['Close'][-365:], mode='lines', name='Data Historis'))
    # Data prediksi
    fig_pred.add_trace(go.Scatter(x=prediction_dates, y=predicted_prices.flatten(), mode='lines', name='Hasil Prediksi', line=dict(color='orange', dash='dash')))
    
    fig_pred.update_layout(
        title=f"Prediksi Harga Bitcoin untuk {n_days_to_predict} Hari ke Depan",
        xaxis_title="Tanggal",
        yaxis_title="Harga (USD)",
        template="plotly_dark"
    )
    st.plotly_chart(fig_pred, use_container_width=True)


except FileNotFoundError:
    st.error(f"Error: File tidak ditemukan. Pastikan file 'BTC-USD.csv' dan 'model_lstm.h5' berada di direktori yang sama dengan aplikasi.")
except Exception as e:
    st.error(f"Terjadi kesalahan saat memproses data atau model: {e}")
