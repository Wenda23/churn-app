import json

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st
from xgboost import XGBClassifier

st.set_page_config(page_title="Prediksi Churn Pelanggan", page_icon="📉", layout="wide")


# ---------- Memuat model (sekali saja, lalu disimpan di cache) ----------
@st.cache_resource
def muat_model():
    model = XGBClassifier()
    model.load_model("model_xgb_churn.json")
    with open("kolom_fitur.json") as f:
        kolom = json.load(f)
    explainer = shap.TreeExplainer(model)
    return model, kolom, explainer


model, KOLOM_FITUR, explainer = muat_model()

# ---------- Judul ----------
st.title("📉 Prediksi Churn Pelanggan Telekomunikasi")
st.caption("Model XGBoost + Explainable AI (SHAP) · Dataset: Telco Customer Churn (IBM/Kaggle)")
st.write(
    "Isi profil pelanggan di panel kiri. Aplikasi akan menghitung **skor risiko churn** "
    "dan menjelaskan **faktor apa saja** yang membuat skornya tinggi atau rendah."
)

# ---------- Input profil pelanggan ----------
st.sidebar.header("Profil Pelanggan")

st.sidebar.subheader("Demografi")
gender = st.sidebar.radio("Jenis kelamin", ["Female", "Male"], horizontal=True)
senior = st.sidebar.radio("Lansia (≥ 65 tahun)?", ["No", "Yes"], horizontal=True)
partner = st.sidebar.radio("Punya pasangan?", ["No", "Yes"], horizontal=True)
dependents = st.sidebar.radio("Punya tanggungan?", ["No", "Yes"], horizontal=True)

st.sidebar.subheader("Akun & Tagihan")
tenure = st.sidebar.slider("Lama berlangganan (bulan)", 0, 72, 5)
contract = st.sidebar.selectbox("Jenis kontrak", ["Month-to-month", "One year", "Two year"])
payment = st.sidebar.selectbox(
    "Metode pembayaran",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
)
paperless = st.sidebar.radio("Tagihan paperless?", ["No", "Yes"], index=1, horizontal=True)
monthly = st.sidebar.number_input("Tagihan bulanan (USD)", 18.0, 120.0, 80.0, step=0.5)
total = st.sidebar.number_input(
    "Total tagihan (USD)", 0.0, 9000.0, float(round(tenure * monthly, 2)), step=10.0,
    help="Default: lama berlangganan × tagihan bulanan",
)

st.sidebar.subheader("Layanan")
phone = st.sidebar.radio("Layanan telepon?", ["No", "Yes"], index=1, horizontal=True)
multiple = st.sidebar.radio("Banyak saluran telepon?", ["No", "Yes"], horizontal=True,
                            disabled=(phone == "No"))
internet = st.sidebar.selectbox("Layanan internet", ["Fiber optic", "DSL", "No"])

tanpa_internet = internet == "No"
addon_label = {
    "OnlineSecurity": "Online Security",
    "OnlineBackup": "Online Backup",
    "DeviceProtection": "Device Protection",
    "TechSupport": "Tech Support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming Movies",
}
addon = {}
for kol, label in addon_label.items():
    pilihan = st.sidebar.checkbox(label, disabled=tanpa_internet)
    addon[kol] = int(pilihan and not tanpa_internet)


# ---------- Mengubah input menjadi fitur (sama persis dengan preprocessing di notebook) ----------
def ya(x):
    return int(x == "Yes")


baris = {
    "gender": int(gender == "Male"),
    "SeniorCitizen": ya(senior),
    "Partner": ya(partner),
    "Dependents": ya(dependents),
    "tenure": tenure,
    "PhoneService": ya(phone),
    "MultipleLines": int(phone == "Yes" and multiple == "Yes"),
    **addon,
    "PaperlessBilling": ya(paperless),
    "MonthlyCharges": monthly,
    "TotalCharges": total,
    "num_addons": sum(addon.values()),
    "InternetService_Fiber optic": int(internet == "Fiber optic"),
    "InternetService_No": int(internet == "No"),
    "Contract_One year": int(contract == "One year"),
    "Contract_Two year": int(contract == "Two year"),
    "PaymentMethod_Credit card (automatic)": int(payment == "Credit card (automatic)"),
    "PaymentMethod_Electronic check": int(payment == "Electronic check"),
    "PaymentMethod_Mailed check": int(payment == "Mailed check"),
}
X = pd.DataFrame([baris])[KOLOM_FITUR].astype(float)

# ---------- Prediksi ----------
skor = float(model.predict_proba(X)[0, 1])
if skor > 0.7:
    segmen, warna = "TINGGI", "🔴"
elif skor >= 0.3:
    segmen, warna = "SEDANG", "🟠"
else:
    segmen, warna = "RENDAH", "🟢"

kiri, kanan = st.columns([1, 2])
with kiri:
    st.subheader("Hasil Prediksi")
    st.metric("Skor risiko churn", f"{skor:.1%}")
    st.progress(skor)
    st.markdown(f"### {warna} Risiko {segmen}")
    st.caption(
        "Skor dipakai untuk **mengurutkan** risiko antar pelanggan. Karena model dilatih dengan "
        "penyeimbangan kelas, skor cenderung lebih tinggi dari peluang churn sebenarnya."
    )

# ---------- Penjelasan SHAP ----------
sv = explainer(X)
kontribusi = pd.Series(sv.values[0], index=KOLOM_FITUR)

with kanan:
    st.subheader("Kenapa skornya seperti ini? (SHAP)")
    plt.figure()
    shap.plots.waterfall(sv[0], max_display=10, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)
    st.caption("Merah = mendorong ke arah churn · Biru = menjauhkan dari churn")

# ---------- Rekomendasi otomatis ----------
st.subheader("💡 Rekomendasi Retensi")
saran = {
    "tenure": "Pelanggan masih baru: masukkan ke program onboarding (kontak di bulan ke-1, 3, dan 6).",
    "Contract_Two year": "Tawarkan insentif untuk beralih ke kontrak jangka panjang (1–2 tahun).",
    "Contract_One year": "Tawarkan insentif untuk beralih ke kontrak jangka panjang (1–2 tahun).",
    "InternetService_Fiber optic": "Pengguna Fiber optic: cek kepuasan terhadap harga dan kualitas jaringan.",
    "PaymentMethod_Electronic check": "Tawarkan insentif kecil untuk beralih ke pembayaran otomatis.",
    "OnlineSecurity": "Tawarkan Online Security gratis/diskon selama beberapa bulan.",
    "TechSupport": "Tawarkan Tech Support gratis/diskon selama beberapa bulan.",
    "MonthlyCharges": "Tagihan tergolong tinggi: pertimbangkan paket yang lebih sesuai kebutuhan.",
    "PaperlessBilling": "Pastikan tagihan elektronik mudah dipahami dan diterima tepat waktu.",
}
pendorong = kontribusi[kontribusi > 0].sort_values(ascending=False)
daftar = []
for fitur in pendorong.index:
    if fitur in saran and saran[fitur] not in daftar:
        daftar.append(saran[fitur])
    if len(daftar) == 3:
        break

if daftar:
    for s in daftar:
        st.markdown(f"- {s}")
else:
    st.success("Tidak ada faktor risiko menonjol. Pertahankan kualitas layanan.")

with st.expander("Lihat data fitur yang dikirim ke model"):
    st.dataframe(X.T.rename(columns={0: "Nilai"}))
