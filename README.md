# StockWise AI — Modul Ramalan Sales, Production, & Projected Stock (V4)

Repositori ini adalah implementasi modul **Demand, Production, & Stock Forecasting** untuk **StockWise AI** (studi kasus **PT Panasonic Gobel Life Solutions Manufacturing Indonesia — Pabrik Cileungsi**). Modul ini menghasilkan proyeksi 6 bulan (`FCT_Sales`, `FCT_Prod`, `FCT_Stock`) beserta keluaran berstandar **Kontrak Data Terpadu** untuk dikonsumsi modul **Early Warning System (EWS)**.

## Fitur & Arsitektur V4

1. **Ingesti & Pembersihan Data ([src/loader.py](src/loader.py))**
   - Mendukung input workbook Excel privat (`--excel`) maupun folder CSV (`--data-dir`).
   - Mengonversi error Excel (`#DIV/0!`, sel kosong) menjadi `NaN`, memvalidasi *primary key* `(SKU_ID, periode)`, dan memisahkan kolom rekap akhir bulan `BO+OI` (`bo_oi_recap`) dari `fct_existing` (forecast PIC).
2. **Klasifikasi SBC & Rekayasa Fitur ([src/features.py](src/features.py))**
   - Mengklasifikasikan pola permintaan per SKU menjadi `Lancar` ($\text{ADI} < 1{,}32$) dan `Putus-putus` ($\text{ADI} \ge 1{,}32$) hanya dari data latih agar bebas *data leakage*.
   - Membangun 20 fitur inti (`SKU_ID`, `kategori`, `lag_*`, `rolling_*`, `nonzero_rate_*`, `months_since_last_sale`, serta kalender musiman Indonesia 2020–2027).
3. **Model Ramalan 1 Titik Prediksi & Neraca Stok ([src/models.py](src/models.py))**
   - **Ramalan Penjualan (`FCT_Sales`, horizon default 6 bulan):** Menggunakan **`Hurdle Dua Tahap`** (`TwoStageHurdleForecaster`: `LGBMClassifier` + `LGBMRegressor` *log-magnitude* dengan *winsorized Duan smearing*) untuk **kedua pola (`Lancar` & `Putus-putus`)** agar ekspektasi volume bebas dari bias median-nol pada data *intermittent* (menurunkan `Cum_Bias_pct` segmen `Putus-putus` pada `lighting_dummy` dari `-84,80%` pada `q50` menjadi `-9,96%`).
   - **Pembanding & Evaluasi Backtest (12 bulan rekursif & *rolling 1-step*):** `LightGBM Kuantil (q50)`, `LightGBM Tweedie`, `Hybrid (AI + Forecast PIC)`, `ARIMA`, `Rata-rata Bergerak (MA-3)`, `Forecast Eksisting (PIC)`, `Rekap Order Akhir Bulan (BO+OI)`, dan `Business Plan (BP)`.
   - **Ramalan Produksi (`FCT_Prod`) & Neraca Stok (`FCT_Stock`):** Memprioritaskan rencana produksi PPIC (`planned_prod_map`, `prod_source="rencana_ppic"`) bila tersedia dan otomatis *fallback* ke `ProductionForecaster` (`prod_source="ai_fallback"`), lalu menghitung `FCT_Stock` dan `proyeksi_defisit` secara rekursif 1–6 bulan.
4. **Explainable AI (SHAP) ([src/models.py](src/models.py))**
   - `TreeExplainer` menghasilkan top-5 fitur penggerak per SKU (`shap_drivers_<slug>.csv`) sesuai model terpilih (`Hurdle Dua Tahap`).
5. **Adapter Kontrak EWS ([src/models.py](src/models.py))**
   - `build_ews_contract_frame` dan `map_sku_to_ews` mengekspor `ews_input_<slug>.csv` berkolom baku `SKU` (`SKU-01`..`SKU-20`), `Bulan` (`YYYY-MM`), `FCT_Prod` & `FCT_Sales` (integer pembulatan setengah ke atas), dan `Pola_SBC`.
6. **Catatan Modul Legacy ([src/rfm.py](src/rfm.py))**
   - Output RFM/*overstock* lama dinonaktifkan secara *default* (hanya aktif bila flag `--include-legacy-rfm` diberikan), karena fungsi tersebut telah dipisah ke modul **EWS**, **Customer Matching**, dan **Wilayah Potensial (B2G)**.

## Struktur Direktori

```text
├── data/               # Data input CSV & workbook dummy Lighting (lighting_dummy.xlsx)
├── scripts/            # Generator dummy (generate_lighting_dummy.py, generate_data.py, create_template.py)
├── src/                # Modul inti: loader.py, features.py, models.py (+ rfm.py legacy)
├── tests/              # Unit test kontrak EWS & integritas pipeline (test_phase1_contract.py)
├── artifacts/          # Model terlatih (.joblib) & metadata.json
├── outputs/            # Hasil backtest, forecast_<slug>.csv, ews_input_<slug>.csv, SHAP (.csv)
├── progress/           # Master plan & laporan progress V1–V4
├── train.py            # CLI utama training, backtest, forecasting, SHAP, & ekspor kontrak EWS
└── requirements.txt
```

## Cara Menjalankan (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 1. Generate ulang 20 SKU dummy Lighting (LGT-01..LGT-20, 60 bulan + snapshot)
python scripts\generate_lighting_dummy.py

# 2. Jalankan backtest 12 bulan, training penuh, ramalan 6 bulan, SHAP, & ekspor kontrak EWS
python train.py --excel data\lighting_dummy.xlsx --out-dir outputs\lighting_dummy --artifacts-dir artifacts\lighting_dummy --backtest-months 12 --forecast-months 6

# Opsi tambahan:
#   --skip-arima          Lewati pembanding ARIMA agar eksekusi lebih cepat
#   --include-legacy-rfm  Sertakan output overstock/RFM legacy

# 3. Jalankan unit test kontrak EWS & integritas modul
python -m unittest discover -s tests -v
```

## Ringkasan Hasil Uji V4 (`data/lighting_dummy.xlsx`)

Pengujian pada `20` SKU $\times$ `60` bulan (`1.200` baris, `backtest 12 bulan`, `forecast 6 bulan`):
- **Backtest Sales Rekursif 12 Bulan (`Hurdle Dua Tahap`):** `WAPE = 33,61%`, `Cum_Bias = +0,47%` secara keseluruhan (`Lancar`: `WAPE 32,53%`, `Cum_Bias +0,62%`; `Putus-putus`: `WAPE 106,03%`, `Cum_Bias -9,96%` vs `-84,80%` pada `q50`).
- **Neraca Stok AI (`Stock_Akhir`):** `WAPE = 23,56%`, `Cum_Bias = -0,24%`.
- **Produksi (`ACT_Prod`):** `FCT Prod Eksisting` (rencana PPIC) `WAPE = 13,18%` vs `AI Production (Hurdle)` `53,72%` (karena itu rencana PPIC diprioritaskan melalui `planned_prod_map`).
- **Keluaran Kontrak EWS (`outputs/lighting_dummy/ews_input_jan2026_jun2026.csv`):** `120` baris (`SKU-01`..`SKU-20`, `2026-01`..`2026-06`), **5/5 unit test** di `tests/test_phase1_contract.py` lulus (`OK`).

## Dokumentasi Progress

- [Master Plan Ramalan](progress/master_plan_ramalan_deka.md)
- [Laporan Progress V1](progress/laporan_progress_v1.md) · [V2](progress/laporan_progress_v2.md) · [V3](progress/laporan_progress_v3.md) · [V4](progress/laporan_progress_v4.md)
