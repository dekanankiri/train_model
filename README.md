# StockWise AI — Model Training & Analytics Prototype

Prototipe pipeline AI untuk **StockWise AI** mencakup:
1. **Klasifikasi Pola Permintaan (SBC)** (*Syntetos–Boylan–Croston*: `Lancar` vs `Putus-putus`).
2. **Demand Forecasting & Backtesting**:
   - **LightGBM Kuantil** (`q10`, `q50`, `q90`) — model utama pola *Lancar*.
   - **Hurdle Dua Tahap** (`LGBMClassifier` + `LGBMRegressor` *log-magnitude*) — model utama pola *Putus-putus*.
   - Pembanding: **ARIMA** dan **Rata-rata Bergerak (MA-3)**.
3. **Explainable AI (SHAP)** menggunakan `TreeExplainer` untuk faktor penggerak ramalan per SKU (**Nov 2026 – Jan 2027**).
4. **Segmentasi RFM & Rekomendasi Stok Berlebih** untuk memeringkat distributor berdasarkan *Recency*, *Frequency*, *Monetary*, serta afinitas produk.

## Struktur Direktori

```text
├── data/               # Dataset sintetis 5 tahun bulanan (Nov 2021 - Okt 2026)
├── scripts/            # Generator dataset sintetis (generate_data.py)
├── src/                # Modul features, models, dan rfm
├── artifacts/          # Model terlatih (.joblib) & metadata.json
├── outputs/            # Hasil backtest, ramalan, SHAP, dan rekomendasi RFM (.csv)
├── train.py            # Script utama training, backtest, & evaluasi
└── requirements.txt
```

## Cara Menjalankan (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 1. Generate ulang data sintetis 5 tahun (opsional, data sudah tersedia di data/)
python scripts\generate_data.py

# 2. Jalankan backtest, training penuh, SHAP, & RFM
python train.py
```
