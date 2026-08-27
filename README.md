# Heart Attack Risk Predictor ❤️

> Aplikasi prediksi risiko serangan jantung berbasis **Machine Learning (XGBoost)** dengan antarmuka interaktif **Streamlit**, dilatih pada dataset kardiovaskular Indonesia.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white) ![XGBoost](https://img.shields.io/badge/Model-XGBoost-017CEE) ![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.8182-2ecc71) ![License](https://img.shields.io/badge/Use-Educational%20Only-lightgrey)

---

## Informasi Proyek

| Item | Keterangan |
|------|------------|
| Mata Kuliah | Artificial Intelligence |
| Judul Proyek | Heart Attack Prediction |
| Kelompok | Webabiq |
| Model Utama | XGBoost (ROC-AUC 0.8182) |
| Platform | Streamlit Web App |
| Dosen Pengampu | Andik Yulianto, S.T., M.T |

---

## Tentang Proyek

**Heart Attack Risk Predictor** adalah aplikasi web yang memperkirakan probabilitas seseorang mengalami serangan jantung berdasarkan faktor risiko klinis dan gaya hidup (usia, tekanan darah, kadar kolesterol, diabetes, hipertensi, riwayat penyakit jantung, kebiasaan merokok, dll).

Proyek ini terdiri dari dua bagian:

1. **Pelatihan model** — eksplorasi data, feature engineering, dan perbandingan beberapa algoritma ML & Deep Learning di dalam notebook.
2. **Aplikasi prediksi** — antarmuka Streamlit yang memuat model terbaik (XGBoost, ROC-AUC 0.8182) untuk melakukan inferensi secara real-time, lengkap dengan visualisasi gauge probabilitas.

> **Disclaimer:** Aplikasi ini dibuat untuk tujuan edukasi/akademik. Hasil prediksi **bukan diagnosis medis** dan tidak menggantikan pemeriksaan oleh tenaga kesehatan profesional.

---

## Fitur Utama

- **Prediksi real-time** probabilitas risiko serangan jantung (%).
- **Kategori tingkat risiko** — *Lower risk* (<35%), *Moderate risk* (35–55%), *Higher risk* (>55%) dengan kode warna hijau/kuning/merah.
- **Visualisasi gauge** interaktif berbasis Plotly.
- **Model XGBoost** dengan *monotonic constraints* (arah pengaruh fitur sesuai logika medis, mis. LDL naik = risiko naik, HDL naik = risiko turun).
- **Klasifikasi @0.5** — keputusan kelas model (*Heart attack likely / No heart attack*).
- **Panel metrik** — Test ROC-AUC, jumlah fitur, dan jenis model.
- **UI custom** dengan tema warna & font yang rapi, serta halaman **About**.

---

## Struktur Proyek

```
AI/
  app.py                                 # Aplikasi Streamlit (UI + inferensi)
  Artificial Intelligence main.ipynb     # Notebook training & eksperimen model
  heart_attack_prediction_indonesia.csv  # Dataset (158.355 baris, 28 kolom)
  artifacts/
    roc_auc_results.pkl                  # Bundle model: XGBoost + scaler + fitur + ROC-AUC
  requirements.txt                       # Daftar dependensi Python
  .streamlit/
    config.toml                          # Konfigurasi tema & server Streamlit
  README.md
```

---

## Dataset

- **Sumber:** `heart_attack_prediction_indonesia.csv`
- **Ukuran:** 158.355 baris x 28 kolom
- **Target:** `heart_attack` (0 = tidak, 1 = ya)
- **Contoh fitur:** `age`, `hypertension`, `diabetes`, `cholesterol_level`, `cholesterol_hdl`, `cholesterol_ldl`, `obesity`, `waist_circumference`, `blood_pressure_systolic`, `fasting_blood_sugar`, `smoking_status`, `previous_heart_disease`, dll.

Kolom yang tidak relevan dibuang saat preprocessing, dan `smoking_status` dipetakan ke numerik (Never = 0, Past = 2, Current = 3).

---

## Pipeline Machine Learning

Seluruh proses ada di **`Artificial Intelligence main.ipynb`**:

1. **Preprocessing** — drop kolom tidak relevan, mapping `smoking_status`, hapus nilai LDL negatif.
2. **Feature Engineering** — membuat fitur turunan: `ldl_hdl_ratio`, `non_hdl_cholesterol`, `metabolic_syndrome`, `bp_hypertension`, `age_previous_hd`, `age_smoking`, `diabetes_hypertension`, `ldl_with_diabetes`.
3. **Split & Scaling** — `train_test_split` (80/20, stratified) + `MinMaxScaler`.
4. **Perbandingan Model:**

| Model | Catatan |
|-------|---------|
| Random Forest | GridSearch (max_depth, dll), `class_weight='balanced'` |
| Linear SVC | GridSearch (C, max_iter, tol) |
| XGBoost | `scale_pos_weight` untuk data imbalance + *monotonic constraints* — model final yang di-deploy (ROC-AUC 0.8182) |
| CatBoost | Threshold tuning berbasis F1 |
| Voting Ensemble | Soft voting (RF + XGB + CatBoost) |
| Deep Learning | Keras Sequential (Dense + Dropout) |

5. **Metrik utama:** ROC-AUC (cocok untuk data tidak seimbang). Nilai tersimpan di `roc_auc_results.pkl` dan ditampilkan otomatis di aplikasi.
6. **Interpretabilitas:** analisis SHAP untuk melihat kontribusi tiap fitur.

---

## Cara Instalasi & Menjalankan Aplikasi

Tersedia dua cara. **Cara A (online via Streamlit Cloud)** tidak memerlukan instalasi apa pun. **Cara B (lokal)** dijalankan di komputer sendiri.

### Cara A — Online (Tanpa Instalasi, untuk Pemula)

1. **Fork** repository proyek ke akun GitHub Anda (klik tombol Fork di pojok kanan atas halaman repo).
2. Pastikan repository hasil fork bersifat **Public** (Settings > Danger Zone > Change visibility > Public).
3. Buka **share.streamlit.io** lalu login menggunakan akun GitHub yang sama.
4. Klik **Create app** lalu pilih **Deploy a public app from GitHub**.
5. Isi **Repository** = repo hasil fork, **Branch** = `main`, **Main file path** = `app.py`.
6. Klik **Deploy!** dan tunggu 1-3 menit hingga aplikasi online.

Setelah berhasil, Streamlit memberi link publik (mis. `https://nama-app.streamlit.app`) yang bisa dibuka siapa saja tanpa perlu install Python atau punya akun GitHub.

### Cara B — Instalasi Lokal

Membutuhkan **Python 3.10+** terpasang di komputer.

**1) Clone repository:**

```bash
git clone <URL-REPO-ANDA>
cd ml-main
```

**2) (Opsional) Buat virtual environment:**

```bash
python -m venv .venv
source .venv/bin/activate
```

(Windows: `.venv\Scripts\activate`)

**3) Install dependensi:**

```bash
pip install -r requirements.txt
```

Dependensi utama: `streamlit`, `pandas`, `scikit-learn`, `xgboost`, `plotly`.

**4) Jalankan aplikasi:**

```bash
streamlit run app.py
```

Aplikasi akan otomatis terbuka di browser pada alamat `http://localhost:8501`. Jika tidak terbuka otomatis, salin alamat tersebut ke browser secara manual.

> **Catatan:** Aplikasi memerlukan file model `artifacts/roc_auc_results.pkl`. Pastikan file ini ikut ter-upload ke GitHub / tersedia di folder proyek, jika tidak akan muncul pesan "Model bundle not found".

---

## Cara Menggunakan Aplikasi

1. Pada halaman utama, gunakan menu **Navigation** di sidebar untuk memilih **Predict** atau **About**.
2. Isi karakteristik pasien pada halaman **Predict** (usia, kolesterol, tekanan darah, status merokok, dll).
3. Klik tombol **Predict heart attack risk** untuk melihat hasil prediksi: probabilitas risiko (%), kategori risiko, dan gauge visual.
4. Buka tab **About** untuk info model & cara deployment.

---

## Anggota Kelompok — Webabiq

| No | Nama | NIM |
|----|------|-----|
| 1 | Lam Jeen Sin Anthony | 2432070 |
| 2 | Devin Ng | 2432012 |
| 3 | Nicholas Syahputra | 2432050 |
| 4 | Xlhynz | 2432059 |
| 5 | Muhammad Anugrah Wahyu Saputra | 2432003 |

---

## Disclaimer

Aplikasi ini dikembangkan untuk tujuan akademik / edukasi. Prediksi dihasilkan oleh model machine learning dan tidak boleh dijadikan dasar keputusan medis. Selalu konsultasikan dengan dokter atau tenaga kesehatan profesional untuk diagnosis yang akurat.
