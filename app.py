import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Load model
model = joblib.load('best_model_ecomerce_churn.pkl')

# --- HEADER ---
st.set_page_config(page_title="Prediksi Churn Pelanggan", layout="wide")
st.title('📊 Prediksi Churn Pelanggan E-Commerce')
st.markdown("""
Aplikasi ini memprediksi apakah pelanggan akan **churn** (berhenti bertransaksi) berdasarkan data input.
Gunakan sidebar untuk memilih metode input data.
""")

# --- SIDEBAR ---
st.sidebar.header("Pilih Metode Input Data")
input_method = st.sidebar.radio("Metode Input:", ['Manual', 'Upload File (CSV/XLSX)'])

# Placeholder untuk menampilkan hasil prediksi
hasil_prediksi_section = st.empty()

if input_method == 'Manual':
    st.sidebar.markdown("### Masukkan Data Pelanggan")
    
    # Dua kolom input di halaman utama
    col1, col2 = st.columns(2)
    
    with col1:
        Tenure = st.number_input('Tenure (bulan)', min_value=0, max_value=100, value=10)
        WarehouseToHome = st.number_input('Warehouse to Home (km)', min_value=0, max_value=100, value=10)
        NumberOfDeviceRegistered = st.number_input('Jumlah Perangkat Terdaftar', min_value=1, max_value=10, value=2)
        PreferedOrderCat = st.selectbox('Kategori Pesanan Favorit', ['Mobile Phone', 'Laptop & Accessory', 'Grocery', 'Mobile', 'Others', 'Fashion'])
        SatisfactionScore = st.selectbox('Skor Kepuasan (1-5)', [1, 2, 3, 4, 5])
        
    with col2:
        MaritalStatus = st.selectbox('Status Pernikahan', ['Single', 'Married', 'Divorced'])
        NumberOfAddress = st.number_input('Jumlah Alamat', min_value=1, max_value=10, value=2)
        Complain = st.selectbox('Pernah Komplain?', [0, 1])
        DaySinceLastOrder = st.number_input('Hari Sejak Transaksi Terakhir', min_value=0, max_value=365, value=30)
        CashbackAmount = st.number_input('Jumlah Cashback', min_value=0, max_value=1000, value=100)
    
    if st.sidebar.button('Prediksi'):
        input_data = pd.DataFrame([{
            'Tenure': Tenure,
            'WarehouseToHome': WarehouseToHome,
            'NumberOfDeviceRegistered': NumberOfDeviceRegistered,
            'PreferedOrderCat': PreferedOrderCat,
            'SatisfactionScore': SatisfactionScore,
            'MaritalStatus': MaritalStatus,
            'NumberOfAddress': NumberOfAddress,
            'Complain': Complain,
            'DaySinceLastOrder': DaySinceLastOrder,
            'CashbackAmount': CashbackAmount
        }])
        
        hasil = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
        
        if hasil == 1:
            hasil_prediksi_section.error(f'⚠️ Pelanggan diprediksi akan **CHURN** (Probabilitas: {prob:.2f})')
        else:
            hasil_prediksi_section.success(f'✅ Pelanggan diprediksi **TIDAK churn** (Probabilitas churn: {prob:.2f})')

else:
    st.sidebar.markdown("### Upload Data Pelanggan")
    file = st.sidebar.file_uploader("Upload file CSV atau Excel", type=["csv", "xlsx"])
    
    if file is not None:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        st.markdown("### 📄 Data yang diupload:")
        st.dataframe(df)
        
        if st.sidebar.button("Prediksi dari File"):
            predictions = model.predict(df)
            probabilities = model.predict_proba(df)[:, 1]

            df['Prediksi Churn'] = predictions
            df['Probabilitas Churn'] = probabilities.round(2)

            hasil_prediksi_section.success("✅ Prediksi berhasil dilakukan!")
            hasil_prediksi_section.dataframe(df)

            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Hasil Prediksi')
            st.download_button(
                label="⬇️ Unduh Hasil Prediksi (.xlsx)",
                data=output.getvalue(),
                file_name='hasil_prediksi_churn.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

