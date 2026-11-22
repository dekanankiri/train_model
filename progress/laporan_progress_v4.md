# Laporan Progress V4 — Integrasi Kontrak EWS, Model Produksi & Neraca Stok, serta Perbaikan Bias Pola Putus-putus

**Referensi Versi:** Iterasi V4 (setelah [laporan_progress_v3.md](file:///c:/hackathon/train_model/progress/laporan_progress_v3.md))  
**Cakupan Modul:** [src/loader.py](file:///c:/hackathon/train_model/src/loader.py), [src/features.py](file:///c:/hackathon/train_model/src/features.py), [src/models.py](file:///c:/hackathon/train_model/src/models.py), [train.py](file:///c:/hackathon/train_model/train.py), [scripts/generate_lighting_dummy.py](file:///c:/hackathon/train_model/scripts/generate_lighting_dummy.py), [tests/test_phase1_contract.py](file:///c:/hackathon/train_model/tests/test_phase1_contract.py)

---

## 1. Ringkasan Progress yang Dilakukan

Iterasi V4 memindahkan pipeline Ramalan dari prototipe eksperimen sales menjadi **hulu resmi rantai StockWise AI** yang memasok proyeksi 6 bulan (`FCT_Sales`, `FCT_Prod`) ke modul **Early Warning System (EWS)** sesuai **Kontrak Data Terpadu**.

Perubahan utama yang diimplementasikan pada V4:

1. **Penyelarasan Horizon, Target 1 Titik, & Generator Dummy Lighting ([train.py](file:///c:/hackathon/train_model/train.py), [scripts/generate_lighting_dummy.py](file:///c:/hackathon/train_model/scripts/generate_lighting_dummy.py))**
   - Mengubah default `--forecast-months 6` (selaras definisi *deadstock* `> 6 bulan` di EWS) dan `--backtest-months 12` (evaluasi siklus tahunan penuh).
   - Menghilangkan interval `q10`–`q90` dari keluaran utama agar fokus pada **1 titik prediksi** (`QuantileLGBMForecaster` default `quantiles=(0.50,)`).
   - Membangun [generate_lighting_dummy.py](file:///c:/hackathon/train_model/scripts/generate_lighting_dummy.py) berisi **20 SKU dummy Lighting** (`LGT-01` s.d. `LGT-20`: 10 pola `Lancar` + 10 pola `Putus-putus`) dengan `1.200` baris historis (`2021-01` s.d. `2025-12`) dan `20` baris snapshot per `2025-12-31`.
2. **Model Produksi (`ACT_Prod`), Override Rencana PPIC, & Neraca Stok (`FCT_Stock`) ([src/features.py](file:///c:/hackathon/train_model/src/features.py), [src/models.py](file:///c:/hackathon/train_model/src/models.py))**
   - Menambahkan fitur produksi (`PROD_FEATURE_COLS`: lag produksi, `stok_akhir_lag_1`, `sales_curr`/`FCT_Sales`) dan `ProductionForecaster` berbasis `TwoStageHurdleForecaster` (`production_hurdle.joblib`).
   - Menambahkan `recursive_production_and_stock_forecast` dan `compute_production_stock_metrics` untuk menghitung proyeksi neraca stok `FCT_Stock` dan `proyeksi_defisit` secara rekursif 1–6 bulan.
   - Menambahkan dukungan `planned_prod_map` sehingga bila rencana produksi PPIC tersedia untuk horizon depan, nilainya dipakai langsung (`prod_source = "rencana_ppic"`); model AI hanya menjadi cadangan (`prod_source = "ai_fallback"`) saat rencana PPIC belum diisi.
3. **Perbaikan Routing `Putus-putus` untuk Menghilangkan Bias Ekstrem (`Hurdle Dua Tahap`) ([src/models.py](file:///c:/hackathon/train_model/src/models.py), [train.py](file:///c:/hackathon/train_model/train.py))**
   - Mengarahkan `model_terpilih` dan `FCT_Sales` untuk **kedua segmen (`Lancar` & `Putus-putus`)** ke **`Hurdle Dua Tahap`**, menggantikan routing sementara `LightGBM Kuantil (q50)` pada pola `Putus-putus` (sementara `lgbm_q50` tetap dipertahankan di kolom pembanding & backtest).
   - Menyelaraskan `compute_shap_drivers(..., intermittent_model_key="hurdle")` agar penjelasan SHAP pada SKU `Putus-putus` juga menjelaskan model `Hurdle Dua Tahap` yang dipakai memprediksi `FCT_Sales`.
4. **Adapter Kontrak Data Terpadu & Pembersihan Batas Modul ([src/loader.py](file:///c:/hackathon/train_model/src/loader.py), [src/models.py](file:///c:/hackathon/train_model/src/models.py), [train.py](file:///c:/hackathon/train_model/train.py))**
   - Menambahkan `map_sku_to_ews`, `_round_half_up_nonneg`, dan `build_ews_contract_frame` untuk mengekspor `ews_input_<slug>.csv` berkolom baku `SKU` (`SKU-01`..`SKU-20`), `Bulan` (`YYYY-MM`), `FCT_Prod` (integer), `FCT_Sales` (integer), dan `Pola_SBC`.
   - Memisahkan alias `"bo+oi"` di [src/loader.py](file:///c:/hackathon/train_model/src/loader.py) menjadi `"bo_oi_recap"` agar tidak bercampur dengan `"fct_existing"` (forecast PIC bulan sebelumnya).
   - Menonaktifkan eksekusi otomatis `src/rfm.py` (`overstock_summary.csv` & `overstock_recommendations.csv`) kecuali bila flag `--include-legacy-rfm` diberikan.
   - Menambahkan suite pengujian kontrak di [tests/test_phase1_contract.py](file:///c:/hackathon/train_model/tests/test_phase1_contract.py).

---

## 2. Alasan & *Thought Process* di Balik Keputusan Teknis

1. **Mengapa `LightGBM Kuantil (q50)` Tidak Dipakai untuk `FCT_Sales` Pola `Putus-putus` ke EWS?**
   - Pada seri *intermittent* ($\text{ADI} \ge 1{,}32$), mayoritas observasi bernilai `0`. Secara matematis, **median bersyarat (`q50`)** dari distribusi dengan $P(Y=0) > 0{,}5$ akan tertarik ke `0`.
   - Pada uji `lighting_dummy`, `q50` memang mendapat `WAPE` sedikit lebih rendah (`100,90%` vs `106,03%`) namun dengan **`Cum_Bias_pct = -84,80%`** (prediksi `LGT-11`..`LGT-14` kolaps ke `0,66`–`6,55` unit). Jika dikirim ke mesin FIFO EWS, `FCT_Sales` yang nyaris nol akan membuat EWS salah membaca hampir semua SKU *Putus-putus* sebagai *overstock/deadstock* ekstrem.
   - Sebaliknya, **`Hurdle Dua Tahap`** menaksir *expected value* $P(Y > 0 \mid X) \times \mathbb{E}[Y \mid Y > 0, X]$ dengan `Cum_Bias_pct` jauh lebih terkendali (**`-9,96%`** pada `Putus-putus` dan **`+0,47%`** keseluruhan), sehingga menjaga konservasi volume pada neraca stok EWS.
2. **Mengapa Rencana Produksi PPIC (`planned_prod_map`) Diprioritaskan di Atas `AI Production`?**
   - Hasil backtest `ACT_Prod` pada `lighting_dummy` membuktikan bahwa `FCT Prod Eksisting` (rencana PPIC) memiliki **`WAPE 13,18%`**, sedangkan `AI Production (Hurdle)` memiliki **`WAPE 53,72%`** (`MA-3 Production` `52,12%`). Karena produksi adalah variabel keputusan internal pabrik (bukan acak seperti permintaan pasar), rencana PPIC harus menjadi sumber utama `FCT_Prod` saat tersedia, dan ML hanya menjadi *fallback* jika jadwal PPIC belum diinput.
3. **Mengapa Pembulatan *Half-Up* & Pemetaan `LGT-nn` $\rightarrow$ `SKU-nn` Dilakukan di Adapter Ekspor?**
   - Mempertahankan nilai kontinu (*float*) pada `forecast_<slug>.csv` untuk audit akurasi/dasbor Ramalan, sambil memastikan `ews_input_<slug>.csv` memenuhi kontrak integer & pola ID baku (`SKU-01`..`SKU-20`) tanpa perlu modifikasi manual oleh pemilik modul EWS.

---

## 3. Hasil Uji & Backtest V4 (`data/lighting_dummy.xlsx`, 20 SKU × 60 Bulan)

Pengujian penuh dijalankan pada [data/lighting_dummy.xlsx](file:///c:/hackathon/train_model/data/lighting_dummy.xlsx) (`1.200` baris, `20` fitur aktif, selesai dalam **`26,89 detik`**):

### A. Ringkasan Backtest Sales Rekursif 12 Bulan (`outputs/lighting_dummy/backtest_summary.csv`)

| Segmen | Model | `n_obs` | `MAE` | `RMSE` | `WAPE_pct` | `Cum_Bias_pct` | `Pinball_q50` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Keseluruhan** | LightGBM Tweedie | 240 | 1.289,41 | 3.801,84 | 35,57% | -14,67% | 644,70 |
| **Keseluruhan** | LightGBM Kuantil (q50) | 240 | 1.318,69 | 4.071,96 | 36,38% | -7,70% | 659,34 |
| **Keseluruhan** | **Hurdle Dua Tahap** | 240 | **1.218,21** | **3.769,29** | **33,61%** | **+0,47%** | **609,11** |
| **Keseluruhan** | ARIMA | 240 | 1.132,53 | 3.616,57 | 31,24% | -3,02% | 566,26 |
| **Keseluruhan** | Rata-rata Bergerak (MA-3) | 240 | 1.067,93 | 3.496,83 | 29,46% | -6,49% | 533,96 |
| **Keseluruhan** | Forecast Eksisting (PIC) | 232 | 1.129,69 | 3.542,56 | 30,61% | -5,27% | 564,84 |
| **Keseluruhan** | Business Plan (BP) | 218 | 1.013,87 | 3.193,43 | 29,16% | +1,06% | 506,94 |
| **Pola Lancar** | **Hurdle Dua Tahap** | 120 | **2.324,03** | **5.327,72** | **32,53%** | **+0,62%** | **1.162,01** |
| **Pola Lancar** | LightGBM Tweedie | 120 | 2.464,89 | 5.372,91 | 34,50% | -13,96% | 1.232,44 |
| **Pola Lancar** | LightGBM Kuantil (q50) | 120 | 2.530,41 | 5.755,10 | 35,42% | -6,56% | 1.265,21 |
| **Pola Putus-putus** | **Hurdle Dua Tahap** | 120 | 112,40 | **174,42** | 106,03% | **-9,96%** | 56,20 |
| **Pola Putus-putus** | LightGBM Kuantil (q50) | 120 | 106,96 | 201,53 | 100,90% | -84,80% | 53,48 |
| **Pola Putus-putus** | LightGBM Tweedie | 120 | 113,93 | 199,62 | 107,47% | -62,50% | 56,96 |

### B. Ringkasan Backtest Rolling 1-Bulan ke Depan (`outputs/lighting_dummy/backtest_rolling_1step_summary.csv`)

| Segmen | Model | `n_obs` | `MAE` | `RMSE` | `WAPE_pct` | `Cum_Bias_pct` | `Pinball_q50` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Keseluruhan** | **Hurdle Dua Tahap** | 240 | **1.172,33** | **3.746,51** | **32,34%** | **-4,78%** | **586,17** |
| **Keseluruhan** | Hybrid (AI + Forecast PIC) | 232 | 1.039,50 | 3.297,40 | 28,17% | -8,56% | 519,75 |
| **Keseluruhan** | Forecast Eksisting (PIC) | 232 | 1.129,69 | 3.542,56 | 30,61% | -5,27% | 564,84 |
| **Pola Putus-putus** | Hurdle Dua Tahap | 120 | 114,08 | 176,28 | 107,61% | **-12,30%** | 57,04 |
| **Pola Putus-putus** | LightGBM Kuantil (q50) | 120 | 109,25 | 211,87 | 103,06% | -69,99% | 54,62 |
| **Pola Putus-putus** | Hybrid (AI + Forecast PIC) | 115 | 44,43 | 73,86 | 42,22% | +0,68% | 22,22 |

### C. Ringkasan Backtest Production & Neraca Stok (Rekursif 12 Bulan)

| Target | Model | `n_obs` | `MAE` | `RMSE` | `WAPE_pct` | `Cum_Bias_pct` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **ACT_Prod** | AI Production (Hurdle) | 231 | 2.022,31 | 5.680,20 | 53,72% | +2,85% |
| **ACT_Prod** | **FCT Prod Eksisting (Rencana PPIC)** | 231 | **496,29** | **1.669,67** | **13,18%** | **+2,79%** |
| **ACT_Prod** | MA-3 Production | 231 | 1.962,37 | 5.721,02 | 52,12% | -19,11% |
| **Stock_Akhir** | **Neraca Stok AI (`FCT_Prod - FCT_Sales`)** | 235 | **2.002,88** | **5.425,63** | **23,56%** | **-0,24%** |

### D. Uji Kontrak & Integritas (`tests/test_phase1_contract.py`)

- Seluruh **5 dari 5** unit test di [tests/test_phase1_contract.py](file:///c:/hackathon/train_model/tests/test_phase1_contract.py) lulus (`OK` dalam `0,186 detik`):
  1. `test_lighting_dummy_uses_two_digit_sku_ids` (`LGT-01`..`LGT-20`).
  2. `test_sku_mapping_and_ews_contract_rounding` (`LGT-nn` $\rightarrow$ `SKU-nn`, pembulatan *half-up*).
  3. `test_loader_separates_bo_oi_from_fct_existing` (`BO+OI` $\rightarrow$ `bo_oi_recap`).
  4. `test_ppic_plan_overrides_ai_production_fallback` (prioritas rencana PPIC & fallback AI).
  5. `test_generated_lighting_outputs_conform_to_ews_contract` (`120` baris [ews_input_jan2026_jun2026.csv](file:///c:/hackathon/train_model/outputs/lighting_dummy/ews_input_jan2026_jun2026.csv), skema regex `SKU-nn` & `YYYY-MM`, tidak kolaps ke nol pada `Putus-putus`).

---

## 4. Kelebihan, Kekurangan, & Status Kesiapan Integrasi (M6)

### Kelebihan
- **Siap Kontrak EWS:** [ews_input_jan2026_jun2026.csv](file:///c:/hackathon/train_model/outputs/lighting_dummy/ews_input_jan2026_jun2026.csv) langsung memenuhi format **Kontrak Data Terpadu** (integer *half-up*, format `SKU-nn`, `YYYY-MM`, `Pola_SBC`).
- **Neraca Stok Seimbang:** Mengganti `q50` dengan `Hurdle Dua Tahap` pada pola `Putus-putus` menekan bias `Stock_Akhir` ke **`-0,24%`**, mencegah alarm *deadstock* palsu di EWS dan penggelembungan `kuota_b2g` di modul Wilayah.
- **Batas Modul Bersih:** RFM/overstock lama tidak lagi dijalankan secara default; `BO+OI` tidak lagi menyamar sebagai forecast PIC.

### Kekurangan & Catatan Terbuka
- Hasil evaluasi di atas masih memakai data dummy Lighting (menunggu data riil *Lighting* dari Panasonic).
- Pada `Pola Putus-putus`, `WAPE` prediksi ML murni tanpa sinyal order berjalan masih tinggi (`106,03%`) karena sifat acak *outlier* proyek; perlu dipantau bersama uji backtest EWS (Tahap 5) saat integrasi M6.
- Jika data riil PPIC mengisi `fct_prod` masa depan, mekanisme `planned_prod_map` perlu disambungkan ke query `produksi.rencana_bulanan` di `pipeline/ramalan/` pada milestone M6.
