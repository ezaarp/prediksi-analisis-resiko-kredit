import streamlit as st
import pandas as pd
import numpy as np

# Import joblib 
try:
    import joblib
except ImportError:
    st.error("joblib tidak terinstall. Pastikan requirements.txt sudah benar.")
    st.stop()

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Dashboard Prediksi Loan Status",
    page_icon="💳",
)
st.title("Aplikasi Prediksi Loan Status")
st.write("Masukkan data peminjam untuk memprediksi kemungkinan gagal bayar pinjaman.")

st.markdown("---")
st.subheader("Evaluasi Model")

# Load model
@st.cache_resource
def load_models():
    try:
        model = joblib.load('model_logreg.pkl')
        scaler = joblib.load('scaler.pkl')
        kmeans = joblib.load('kmeans_final.pkl')
        pca = joblib.load('pca.pkl')
        return model, scaler, kmeans, pca, True
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, False

model, scaler, kmeans, pca, models_loaded = load_models()

if not models_loaded:
    st.error("❌ Model files tidak dapat dimuat. Pastikan semua file .pkl sudah ter-upload ke repository.")
    st.stop()

# Load data 
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('credit_risk_dataset.csv')
        # Imputasi 
        df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
        df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].mean())
        return df, True
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None, False

data, data_loaded = load_data()

if not data_loaded:
    st.error("❌ Dataset tidak dapat dimuat. Pastikan credit_risk_dataset.csv sudah ter-upload.")
    st.stop()

# Mapping label encoder 
home_ownership_map = {'RENT':0, 'OWN':1, 'MORTGAGE':2, 'OTHER':3}
loan_intent_map = {'DEBTCONSOLIDATION':0, 'EDUCATION':1, 'HOMEIMPROVEMENT':2, 'MEDICAL':3, 'PERSONAL':4, 'VENTURE':5}
loan_grade_map = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6}
default_on_file_map = {'N':0, 'Y':1}

# Fitur yang digunakan
features = [
    'person_age',
    'person_income',
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'person_emp_length',
    'cb_person_cred_hist_length',
    'person_home_ownership',
    'loan_intent',
    'loan_grade',
    'cb_person_default_on_file'
]

# Encode data untuk evaluasi
def preprocess_eval(df):
    df = df.copy()
    df['person_home_ownership'] = df['person_home_ownership'].map(home_ownership_map)
    df['loan_intent'] = df['loan_intent'].map(loan_intent_map)
    df['loan_grade'] = df['loan_grade'].map(loan_grade_map)
    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map(default_on_file_map)
    return df

if models_loaded and data_loaded:
    try:
        # Preprocessing
        data_transformed = data.copy()
        categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
        for col in categorical_cols:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            data_transformed[col] = le.fit_transform(data_transformed[col])
        
        # Scaling
        features_for_clustering = [
            'person_age', 'person_income', 'loan_amnt', 'loan_int_rate', 'loan_percent_income',
            'person_emp_length', 'cb_person_cred_hist_length', 'person_home_ownership',
            'loan_intent', 'loan_grade', 'cb_person_default_on_file'
        ]
        X_scaled = scaler.transform(data_transformed[features_for_clustering])
        
        # PCA Transform
        X_pca = pca.transform(X_scaled)
        pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        
        # Cluster menggunakan data PCA (tanpa feature names untuk menghindari error)
        cluster = kmeans.predict(X_pca)
        data_transformed['Cluster'] = cluster
        
        # Prepare final features
        X = data_transformed[features_for_clustering + ['Cluster']]
        y = data_transformed['loan_status']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        st.info("📊 **Evaluasi Model pada Test Set (20% data yang tidak digunakan untuk training)**")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.success(f"Accuracy: **{acc:.2f}**")
        with col2:
            st.info(f"Precision: **{prec:.2f}**")
        with col3:
            st.warning(f"Recall: **{rec:.2f}**")
        with col4:
            st.error(f"ROC AUC: **{roc_auc:.2f}**")

        # info ukuran dataset
        st.write(f"📈 **Dataset Info:** Total: {len(data):,} samples | Training: {len(X_train):,} | Test: {len(X_test):,}")

        plot_option = st.selectbox("Pilih grafik untuk ditampilkan:", ["Pilih", "ROC AUC Curve", "Confusion Matrix"])
        if plot_option == "ROC AUC Curve":
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
            ax2.plot([0, 1], [0, 1], linestyle='--', color='gray')
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.set_title('ROC Curve')
            ax2.legend()
            st.pyplot(fig2)
        elif plot_option == "Confusion Matrix":
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Error dalam evaluasi model: {str(e)}")

st.markdown("---")
st.subheader("Prediksi Loan Status Baru")

def user_input():
    col1, col2 = st.columns(2)
    with col1:
        person_age = st.number_input("Umur", min_value=18, max_value=100, value=25)
        person_income = st.number_input("Pendapatan Tahunan", min_value=0, value=50000)
        loan_amnt = st.number_input("Jumlah Pinjaman", min_value=0, value=10000)
        loan_int_rate = st.number_input("Suku Bunga (%)", min_value=0.0, max_value=30.0, value=10.0)
        loan_percent_income = st.number_input("Persentase Pinjaman dari Pendapatan", min_value=0.0, max_value=1.0, value=0.2)
        person_emp_length = st.number_input("Lama Bekerja (tahun)", min_value=0.0, value=2.0)
    with col2:
        cb_person_cred_hist_length = st.number_input("Panjang Riwayat Kredit", min_value=0, value=5)
        person_home_ownership = st.selectbox("Status Kepemilikan Rumah", ["RENT", "OWN", "MORTGAGE", "OTHER"])
        loan_intent = st.selectbox("Tujuan Pinjaman", ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"])
        loan_grade = st.selectbox("Grade Pinjaman", ["A", "B", "C", "D", "E", "F", "G"])
        cb_person_default_on_file = st.selectbox("Riwayat Gagal Bayar", ["N", "Y"])
    
    # Encode categorical
    home_encoded = home_ownership_map[person_home_ownership]
    intent_encoded = loan_intent_map[loan_intent]
    grade_encoded = loan_grade_map[loan_grade]
    default_encoded = default_on_file_map[cb_person_default_on_file]
    
    data = {
        'person_age': person_age,
        'person_income': person_income,
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'person_emp_length': person_emp_length,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'person_home_ownership': home_encoded,
        'loan_intent': intent_encoded,
        'loan_grade': grade_encoded,
        'cb_person_default_on_file': default_encoded
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input()

if st.button("Prediksi Loan Status") and models_loaded:
    try:
        # Scaling
        X_user_scaled = scaler.transform(input_df[features])
        # PCA Transform
        X_user_pca = pca.transform(X_user_scaled)
        # Cluster menggunakan data PCA (gunakan array numpy, bukan dataframe)
        user_cluster = kmeans.predict(X_user_pca)
        # Gabungkan fitur scaled dengan cluster
        X_user_final = np.concatenate([X_user_scaled, user_cluster.reshape(-1,1)], axis=1)
        # Predict
        pred = model.predict(X_user_final)[0]
        prob = model.predict_proba(X_user_final)[0,1]
        st.markdown("### Hasil Prediksi:")
        if pred == 1:
            st.error(f"**Gagal Bayar (Default)** dengan probabilitas {prob:.2f}")
        else:
            st.success(f"**LUNAS (Tidak Default)** dengan probabilitas {1-prob:.2f}")
    except Exception as e:
        st.error(f"Error dalam prediksi: {str(e)}") 