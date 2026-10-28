import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st
from xgboost import XGBClassifier

st.set_page_config(page_title="Cek Risiko Churn", page_icon="📶", layout="wide")

FOLDER = Path(__file__).parent


MERAH, BIRU = "#c8423b", "#2a78d6"
SEGMEN = {
    "Rendah": {"warna": "#2f8f5b", "latar": "#e8f4ed",
               "arti": "Pelanggan ini kemungkinan besar tetap berlangganan."},
    "Sedang": {"warna": "#b9770e", "latar": "#fbf1e1",
               "arti": "Ada beberapa tanda risiko. Sebaiknya dipantau."},
    "Tinggi": {"warna": "#c8423b", "latar": "#f9e7e5",
               "arti": "Pelanggan ini berisiko besar berhenti. Prioritaskan untuk dihubungi."},
}


def dolar(x):
    return f"${x:,.0f}"


# ---------------------------------------------------------------------
# Gaya tampilan
# ---------------------------------------------------------------------
st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 2.2rem;}
h1 {font-size: 2rem !important; letter-spacing: -0.02em;}
.sub {color: #5b6675; font-size: 1rem; margin: -0.6rem 0 1.2rem 0;}

.skor {border-radius: 14px; padding: 22px 24px 20px 24px;}
.skor .angka {font-size: 3.4rem; font-weight: 800; line-height: 1; letter-spacing: -0.03em;}
.skor .label {display: inline-block; font-weight: 700; font-size: 0.95rem;
              padding: 4px 12px; border-radius: 999px; color: #fff; margin-left: 10px;
              vertical-align: 0.9rem;}
.skor .arti {margin-top: 10px; font-size: 1.02rem; color: #1d2a3a;}

.skala {position: relative; margin: 18px 0 4px 0;}
.skala .jalur {display: flex; height: 10px; border-radius: 999px; overflow: hidden;}
.skala .jalur div {height: 100%;}
.skala .penanda {position: absolute; top: -5px; width: 4px; height: 20px;
                 background: #1d2a3a; border-radius: 2px; transform: translateX(-50%);}
.skala .angka-skala {position: relative; height: 1.1rem; font-size: 0.78rem;
                     color: #5b6675; margin-top: 6px;}
.skala .angka-skala span {position: absolute; transform: translateX(-50%);}
.skala .angka-skala span:first-child {transform: none;}
.skala .angka-skala span:last-child {transform: translateX(-100%);}

.faktor {display: flex; gap: 28px; margin-top: 6px;}
.faktor > div {flex: 1;}
.faktor h4 {font-size: 0.9rem; font-weight: 700; margin: 0 0 6px 0;}
.faktor ul {margin: 0; padding-left: 18px;}
.faktor li {margin-bottom: 3px;}

.saran {border-left: 3px solid #1f6f8b; padding: 2px 0 2px 14px; margin-bottom: 14px;}
.saran b {display: block; margin-bottom: 2px;}
.saran span {color: #3d4a5a;}

.kosong {border: 1.5px dashed #cfd6de; border-radius: 14px; padding: 48px 28px;
         text-align: center; color: #5b6675;}
.kosong b {display: block; color: #1d2a3a; font-size: 1.05rem; margin-bottom: 4px;}


.grafik {margin: 4px 0 18px 0;}
.br {display: grid; grid-template-columns: minmax(0, 2fr) minmax(0, 3fr);
     align-items: center; gap: 16px; padding: 5px 0;}
.bl {text-align: right; font-size: 0.95rem; line-height: 1.3;}
.bt {position: relative; height: 24px;}
.bt .mid {position: absolute; left: 50%; top: -6px; bottom: -6px; width: 1.5px;
          background: #1d2a3a;}
.bt .bar {position: absolute; top: 4px; height: 16px;}
.bt-label {display: flex; justify-content: space-between; font-size: 0.82rem; font-weight: 600;}
.cara-baca {background: #f2f4f7; border-radius: 12px; padding: 14px 18px 6px 18px;
             margin: 6px 0 26px 0; font-size: 0.92rem; color: #3d4a5a;}
.cara-baca b {color: #1d2a3a;}
.cara-baca ul {margin: 6px 0 0 0; padding-left: 18px;}
.cara-baca li {margin-bottom: 6px;}

@media (max-width: 640px) {
  .br {grid-template-columns: 1fr; gap: 2px; padding: 4px 0 8px 0;}
  .bl {text-align: left;}
  .bh > div:first-child {display: none;}
  .faktor {display: block;}
  .faktor > div {margin-bottom: 12px;}
  .skor .angka {font-size: 2.8rem;}
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
@st.cache_resource
def muat_model():
    model = XGBClassifier()
    model.load_model(FOLDER / "model_xgb_churn.json")
    with open(FOLDER / "kolom_fitur.json") as f:
        kolom = json.load(f)
    return model, kolom, shap.TreeExplainer(model)


model, KOLOM_FITUR, explainer = muat_model()

KONTRAK = {"Bulanan": "Month-to-month", "1 tahun": "One year", "2 tahun": "Two year"}
INTERNET = {"Fiber optic": "Fiber optic", "DSL": "DSL", "Tidak berlangganan": "No"}
BAYAR = {
    "Transfer bank otomatis": "Bank transfer (automatic)",
    "Kartu kredit otomatis": "Credit card (automatic)",
    "Cek elektronik": "Electronic check",
    "Cek via pos": "Mailed check",
}
# Kelompok tagihan per bulan (USD, sesuai data asli) -> nilai tengah yang dikirim ke model
TAGIHAN = {
    "Di bawah $20": 19,
    "$20 – $40": 30,
    "$40 – $60": 50,
    "$60 – $80": 70,
    "$80 – $100": 90,
    "Di atas $100": 110,
}
LAYANAN = {
    "OnlineSecurity": "Online Security",
    "OnlineBackup": "Online Backup",
    "DeviceProtection": "Device Protection",
    "TechSupport": "Tech Support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming Movies",
}


def skala_html(skor=None):
    penanda = "" if skor is None else f'<div class="penanda" style="left:{skor*100:.1f}%"></div>'
    return f"""
    <div class="skala">
      <div class="jalur">
        <div style="width:30%;background:{SEGMEN['Rendah']['warna']}"></div>
        <div style="width:40%;background:{SEGMEN['Sedang']['warna']}"></div>
        <div style="width:30%;background:{SEGMEN['Tinggi']['warna']}"></div>
      </div>
      {penanda}
      <div class="angka-skala"><span style="left:0">0%</span><span style="left:30%">30%</span><span style="left:70%">70%</span><span style="left:100%">100%</span></div>
    </div>"""


def grafik_html(df, judul=True):
    """Grafik batang dua arah: merah ke kanan menaikkan risiko, biru ke kiri menurunkan."""
    maks = df["Kontribusi"].abs().max() or 1
    baris = ""
    for _, r in df.iterrows():
        lebar = abs(r["Kontribusi"]) / maks * 48
        if r["Kontribusi"] >= 0:
            posisi = f"left:50%;width:{lebar:.1f}%;background:{MERAH};border-radius:0 4px 4px 0"
        else:
            posisi = f"right:50%;width:{lebar:.1f}%;background:{BIRU};border-radius:4px 0 0 4px"
        baris += (f'<div class="br"><div class="bl">{r["Faktor"]}</div>'
                  f'<div class="bt"><div class="mid"></div><div class="bar" style="{posisi}"></div></div></div>')
    kepala = ('<div class="br bh"><div></div><div class="bt-label">'
              f'<span style="color:{BIRU}">menurunkan risiko</span>'
              f'<span style="color:{MERAH}">menaikkan risiko</span></div></div>') if judul else ""
    return f'<div class="grafik">{kepala}{baris}</div>'


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("Cek Risiko Churn Pelanggan")
st.markdown('<p class="sub">Masukkan data pelanggan untuk melihat seberapa besar kemungkinan ia '
            'berhenti berlangganan, beserta alasannya.</p>', unsafe_allow_html=True)

kiri, kanan = st.columns([5, 7], gap="large")

with kiri:
    with st.form("data_pelanggan", border=True):
        st.subheader("Data pelanggan")

        st.markdown("**Profil**")
        gender = st.radio("Jenis kelamin", ["Perempuan", "Laki-laki"], horizontal=True)
        a, b = st.columns(2)
        senior = a.toggle("Usia 65 tahun ke atas")
        partner = b.toggle("Sudah menikah")
        dependents = a.toggle("Punya tanggungan")

        st.markdown("**Langganan**")
        tenure = st.number_input("Lama berlangganan (bulan)", min_value=0, max_value=72,
                                 value=12, step=1)
        a, b = st.columns(2)
        kontrak = a.selectbox("Kontrak", list(KONTRAK))
        bayar = b.selectbox("Cara bayar", list(BAYAR))
        kelompok_tagihan = st.selectbox("Tagihan per bulan (USD)", list(TAGIHAN), index=3)
        paperless = st.toggle("Tagihan dikirim lewat email (tanpa kertas)", value=True)

        st.markdown("**Layanan**")
        a, b = st.columns(2)
        internet = a.selectbox("Internet", list(INTERNET))
        phone = b.selectbox("Telepon rumah", ["1 saluran", "Lebih dari 1 saluran",
                                              "Tidak berlangganan"])
        st.caption("Layanan tambahan internet")
        a, b = st.columns(2)
        layanan_input = {}
        for i, (kol, label) in enumerate(LAYANAN.items()):
            layanan_input[kol] = (a if i % 2 == 0 else b).checkbox(label)

        hitung = st.form_submit_button("Hitung risiko", type="primary",
                                       use_container_width=True)

if hitung:
    st.session_state["hasil"] = True

with kanan:
    # Model butuh satu angka, jadi dipakai nilai tengah dari kelompok tagihan yang dipilih
    tagihan_usd = TAGIHAN[kelompok_tagihan]

    if not st.session_state.get("hasil"):
        st.markdown('<div class="kosong"><b>Belum ada hasil</b>'
                    'Lengkapi data di sebelah kiri, lalu tekan <i>Hitung risiko</i>.</div>',
                    unsafe_allow_html=True)
    else:

        # -------------------------------------------------------------
        # Input -> fitur (harus sama dengan preprocessing di notebook)
        # -------------------------------------------------------------
        punya_internet = internet != "Tidak berlangganan"
        punya_telepon = phone != "Tidak berlangganan"
        layanan = {k: int(v and punya_internet) for k, v in layanan_input.items()}
        monthly_usd = tagihan_usd
        total_usd = tenure * monthly_usd

        baris = {
            "gender": int(gender == "Laki-laki"),
            "SeniorCitizen": int(senior),
            "Partner": int(partner),
            "Dependents": int(dependents),
            "tenure": tenure,
            "PhoneService": int(punya_telepon),
            "MultipleLines": int(phone == "Lebih dari 1 saluran"),
            **layanan,
            "PaperlessBilling": int(paperless),
            "MonthlyCharges": monthly_usd,
            "TotalCharges": total_usd,
            "num_addons": sum(layanan.values()),
            "InternetService_Fiber optic": int(INTERNET[internet] == "Fiber optic"),
            "InternetService_No": int(INTERNET[internet] == "No"),
            "Contract_One year": int(KONTRAK[kontrak] == "One year"),
            "Contract_Two year": int(KONTRAK[kontrak] == "Two year"),
            "PaymentMethod_Credit card (automatic)": int(BAYAR[bayar] == "Credit card (automatic)"),
            "PaymentMethod_Electronic check": int(BAYAR[bayar] == "Electronic check"),
            "PaymentMethod_Mailed check": int(BAYAR[bayar] == "Mailed check"),
        }
        X = pd.DataFrame([baris])[KOLOM_FITUR].astype(float)

        skor = float(model.predict_proba(X)[0, 1])
        segmen = "Tinggi" if skor > 0.7 else "Sedang" if skor >= 0.3 else "Rendah"
        gaya = SEGMEN[segmen]

        sv = explainer(X)
        nilai = pd.Series(sv.values[0], index=KOLOM_FITUR)
        gabung = lambda awalan: nilai[[c for c in KOLOM_FITUR if c.startswith(awalan)]].sum()
        yt = lambda v: "ya" if v else "tidak"

        # Kolom one-hot digabung jadi satu faktor (nilai SHAP bersifat aditif)
        faktor = pd.DataFrame([
            (f"Berlangganan {tenure} bulan", nilai["tenure"]),
            (f"Kontrak {kontrak.lower()}", gabung("Contract_")),
            (f"Internet: {internet}", gabung("InternetService_")),
            (f"Bayar dengan {bayar.lower()}", gabung("PaymentMethod_")),
            (f"Tagihan {kelompok_tagihan.replace('Di ', 'di ')}/bulan", nilai["MonthlyCharges"]),
            (f"Total tagihan sekitar {dolar(tenure * tagihan_usd)}", nilai["TotalCharges"]),
            (f"Tagihan tanpa kertas: {yt(paperless)}", nilai["PaperlessBilling"]),
            (f"Usia 65+: {yt(senior)}", nilai["SeniorCitizen"]),
            (f"Sudah menikah: {yt(partner)}", nilai["Partner"]),
            (f"Punya tanggungan: {yt(dependents)}", nilai["Dependents"]),
            (gender, nilai["gender"]),
            (f"Telepon rumah: {phone.lower()}", nilai["PhoneService"] + nilai["MultipleLines"]),
            (f"{baris['num_addons']} layanan tambahan", nilai["num_addons"]),
            *[(f"{label}: {yt(layanan[k])}", nilai[k]) for k, label in LAYANAN.items()],
        ], columns=["Faktor", "Kontribusi"])
        faktor = faktor.reindex(faktor["Kontribusi"].abs().sort_values(ascending=False).index)
        naik = faktor[faktor["Kontribusi"] > 0.02].head(3)["Faktor"]
        turun = faktor[faktor["Kontribusi"] < -0.02].head(3)["Faktor"]

        daftar = lambda s: ("<ul>" + "".join(f"<li>{x}</li>" for x in s) + "</ul>"
                            if len(s) else "<p style='color:#5b6675'>Tidak ada yang menonjol</p>")

        # Kartu skor
        st.markdown(f"""
        <div class="skor" style="background:{gaya['latar']}">
          <span class="angka" style="color:{gaya['warna']}">{skor:.0%}</span>
          <span class="label" style="background:{gaya['warna']}">Risiko {segmen.lower()}</span>
          <div class="arti">{gaya['arti']}</div>
          {skala_html(skor)}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="faktor" style="margin-top:18px">
          <div><h4 style="color:{MERAH}">Yang menaikkan risiko</h4>{daftar(naik)}</div>
          <div><h4 style="color:{BIRU}">Yang menahan pelanggan</h4>{daftar(turun)}</div>
        </div>""", unsafe_allow_html=True)

        # Grafik faktor
        st.subheader("Seberapa besar pengaruh tiap faktor")
        st.markdown(grafik_html(faktor.head(8)), unsafe_allow_html=True)
        st.markdown(f"""
        <div class="cara-baca">
          <b>Cara membaca</b>
          <ul>
            <li>Batang <span style="color:{MERAH};font-weight:700">merah</span> ke kanan membuat
                pelanggan lebih mungkin berhenti. Batang <span style="color:{BIRU};font-weight:700">biru</span>
                ke kiri membuatnya lebih mungkin bertahan.</li>
            <li>Semakin panjang batangnya, semakin besar pengaruh faktor tersebut.</li>
            <li>Skor di bawah 30% tergolong rendah, 30–70% sedang, dan di atas 70% tinggi. Pada data uji,
                yang benar-benar berhenti: 5% di kelompok rendah, 27% sedang, 65% tinggi.</li>
            <li>Skor paling tepat dipakai untuk menentukan pelanggan mana yang dihubungi lebih dulu,
                bukan sebagai peluang pasti.</li>
          </ul>
        </div>""", unsafe_allow_html=True)

        # Saran
        st.subheader("Langkah yang bisa diambil")
        saran = []
        for nama in faktor[faktor["Kontribusi"] > 0.02]["Faktor"]:
            if nama.startswith("Berlangganan"):
                saran.append(("Dampingi di bulan-bulan awal",
                              "Hubungi pelanggan di bulan ke-1, 3, dan 6 untuk memastikan layanannya lancar."))
            elif nama.startswith("Kontrak") and kontrak != "2 tahun":
                saran.append(("Tawarkan kontrak lebih panjang",
                              "Beri potongan harga atau bonus kuota bila beralih ke kontrak 1–2 tahun."))
            elif nama.startswith("Internet: Fiber"):
                saran.append(("Tanyakan kepuasan layanan Fiber",
                              "Cek apakah keluhannya soal harga atau kualitas jaringan."))
            elif nama.startswith("Bayar dengan cek elektronik"):
                saran.append(("Ajak beralih ke pembayaran otomatis",
                              "Beri insentif kecil untuk pembayaran lewat autodebet."))
            elif (nama.startswith(("Online Security", "Tech Support")) and punya_internet):
                saran.append(("Berikan layanan proteksi",
                              "Gratiskan Online Security atau Tech Support selama beberapa bulan."))
            elif nama.startswith("Tagihan") and "kertas" not in nama:
                saran.append(("Tinjau ulang paketnya",
                              "Tagihannya tinggi. Tawarkan paket yang lebih sesuai kebutuhan."))
        saran = list(dict.fromkeys(saran))[:3]

        if saran:
            st.markdown("".join(f'<div class="saran"><b>{j}</b><span>{t}</span></div>'
                                for j, t in saran), unsafe_allow_html=True)
        else:
            st.markdown("Tidak ada tindakan khusus. Pertahankan kualitas layanan.")

        with st.expander("Detail teknis"):
            st.markdown("SHAP waterfall plot dalam satuan log-odds, dari nilai dasar "
                        "`E[f(X)]` ke prediksi `f(x)`. Untuk kelompok tagihan, yang dikirim ke model "
                        "adalah nilai tengahnya.")
            plt.figure()
            shap.plots.waterfall(sv[0], max_display=12, show=False)
            st.pyplot(plt.gcf(), clear_figure=True)
            st.dataframe(X.T.rename(columns={0: "Nilai yang dikirim ke model"}),
                         use_container_width=True)
