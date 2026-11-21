# Laporan Progress V3 — Pengujian `LightGBM Tweedie`, Mode Hybrid (`AI + BO+OI`), & Evaluasi SKU Berhistori

**Referensi Versi:** Iterasi V3 (setelah [laporan_progress_v2.md](file:///c:/hackathon/train_model/progress/laporan_progress_v2.md))  
**Cakupan Modul:** [src/features.py](file:///c:/hackathon/train_model/src/features.py), [src/models.py](file:///c:/hackathon/train_model/src/models.py), [train.py](file:///c:/hackathon/train_model/train.py)

---

## 1. Ringkasan Progress yang Dilakukan

Iterasi V3 bertujuan mengatasi dua celah yang ditemukan di V2: (1) bias *under-forecast* `-46.82%` pada `LightGBM Kuantil (q50)` di **Pola Lancar**, dan (2) kesenjangan akurasi terhadap **`Forecast Eksisting (BO+OI)`** (`WAPE 25.15%`).

Perubahan utama yang diimplementasikan pada V3:
1. **`TweedieLGBMForecaster` ([src/models.py](file:///c:/hackathon/train_model/src/models.py))**
   - Menambahkan model `LGBMRegressor` dengan `objective="tweedie"` dan `tweedie_variance_power=1.35`, `min_child_samples=10`, `reg_alpha=0.1`, `reg_lambda=1.0`.
   - Dijadikan model utama untuk segmen `Pola Lancar` pada tahap ramalan masa depan (`tweedie_lgbm.joblib`) dan evaluasi SHAP (`compute_shap_drivers`), sementara `lgbm_q10` & `lgbm_q90` dipertahankan sebagai rentang ketidakpastian.
2. **`HybridBOOILGBMForecaster` & Fitur Hybrid ([src/features.py](file:///c:/hackathon/train_model/src/features.py), [src/models.py](file:///c:/hackathon/train_model/src/models.py))**
   - Menambahkan fungsi `add_hybrid_bo_oi_features` untuk membangun 5 fitur tanpa *data leakage* (`log1p_fct_existing`, `fct_vs_roll3_log_ratio`, `fct_log_err_lag_1`, `fct_log_err_mean_3`, dan `log1p_bp_sales`).
   - Melatih `HybridBOOILGBMForecaster` untuk mempelajari koreksi residual log $\Delta = \log(1 + \text{Sales}) - \log(1 + \text{BO+OI})$ dengan regularisasi kuat (`num_leaves=15`, `min_child_samples=20`, `reg_alpha=1.0`, `reg_lambda=5.0`) dan *smearing factor* dibatasi pada `[0.97, 1.05]` (`hybrid_bo_oi.joblib`).
3. **Evaluasi Khusus SKU Berhistori ([train.py](file:///c:/hackathon/train_model/train.py))**
   - Menambahkan output `backtest_rolling_known_skus_summary.csv` agar perbandingan model pada `rolling 1-step` dapat diamati khusus pada SKU yang benar-benar memiliki histori di data latih (`ada_di_train == True`), terpisah dari `141` SKU baru di 6 bulan terakhir.
   - Menambahkan `MAE_lgbm_tweedie` dan `MAE_hybrid_1step` ke `backtest_sku_metrics.csv`.

---

## 2. Alasan & *Thought Process* di Balik Keputusan Teknis

1. **Mengapa Mencoba `Tweedie` untuk `Pola Lancar`?**
   - **Landasan Teori:** Distribusi *Tweedie Compound Poisson-Gamma* ($1 < p < 2$) secara alami memodelkan campuran nilai nol dan nilai positif miring ke kanan (*right-skewed*) langsung pada skala unit dengan *log-link*, sehingga secara teori menaksir *expected value* $\mathbb{E}[Y \mid X]$ tanpa mengalami bias Pertidaksamaan Jensen (*Jensen's Inequality*) seperti transformasi `log1p` pada median (`q50`).
2. **Mengapa Mencoba `Hybrid (AI + BO+OI)` Berbasis *Residual Learning*?**
   - **Landasan Teori:** Merujuk pada *Forecast Combination Theory* (Bates & Granger, 1969) dan literatur integrasi prakiraan operasional-statistik (Fildes et al., 2009), `BO+OI` membawa sinyal *Order Intake* H-1 yang sangat kuat namun berpotensi memiliki bias musiman atau pola *spot-order* susulan per SKU. Dengan mempelajari residual terhadap `BO+OI` (bukan menebak dari nol), AI difokuskan hanya untuk mengoreksi pola error sistematis dari `BO+OI`.
3. **Mengapa Memisahkan Evaluasi `SKU Berhistori Train`?**
   - Pada V1–V2, terdapat perbedaan `n_obs` antara ML (`5,712`) dan `MA-3`/`ARIMA` rekursif (`4,866`) karena adanya `141` SKU baru (`846` observasi) yang belum ada di data latih. Evaluasi khusus `4,866` observasi berhistori memastikan perbandingan seluruh model benar-benar berada pada sampel yang identik.

---

## 3. Hasil Backtest V3 & Penjelasannya

Pengujian dijalankan pada data aktual dengan `--skip-arima` (selesai dalam `96.72 detik`, mendeteksi `76` SKU dengan status overstock).

### A. Ringkasan Backtest Multi-Bulan (Rekursif 6 Bulan — `backtest_summary.csv`)

| Segmen | Model | `n_obs` | `MAE` | `RMSE` | `WAPE_pct` | `Cum_Bias_pct` | `Pinball_q50` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Keseluruhan** | LightGBM Tweedie | 5,712 | 4,834.61 | 68,230.24 | 55.23% | -27.01% | 2,417.31 |
| **Keseluruhan** | LightGBM Kuantil (q50) | 5,712 | 5,199.54 | 77,779.13 | 59.40% | -46.13% | 2,599.77 |
| **Keseluruhan** | **Hurdle Dua Tahap** | 5,712 | **3,961.93** | **48,188.59** | **45.26%** | **+5.45%** | **1,980.96** |
| **Keseluruhan** | Rata-rata Bergerak (MA-3) | 4,866 | 4,995.51 | 48,904.61 | 48.84% | -5.50% | 2,497.76 |
| **Keseluruhan** | Forecast Eksisting (BO+OI) | 5,673 | 2,212.43 | 36,174.22 | 25.15% | -1.87% | 1,106.22 |
| **Keseluruhan** | Business Plan (BP) | 4,422 | 9,433.90 | 99,354.91 | 89.64% | -74.95% | 4,716.95 |
| **Pola Lancar** | LightGBM Tweedie | 1,758 | 13,230.49 | 122,339.67 | 51.37% | -31.72% | 6,615.24 |
| **Pola Lancar** | LightGBM Kuantil (q50) | 1,758 | 15,112.52 | 139,814.55 | 58.68% | -46.82% | 7,556.26 |
| **Pola Lancar** | **Hurdle Dua Tahap** | 1,758 | **10,109.59** | **85,755.84** | **39.25%** | **+0.61%** | **5,054.79** |
| **Pola Lancar** | Rata-rata Bergerak (MA-3) | 912 | 20,656.59 | 110,309.18 | 41.82% | -12.94% | 10,328.29 |
| **Pola Lancar** | Forecast Eksisting (BO+OI) | 1,719 | 6,519.90 | 65,455.82 | 24.80% | -1.62% | 3,259.95 |
| **Pola Lancar** | Business Plan (BP) | 1,068 | 35,106.53 | 201,403.92 | 89.04% | -81.61% | 17,553.26 |
| **Pola Putus-putus** | LightGBM Tweedie | 3,954 | 1,101.70 | 8,407.18 | 92.21% | +18.17% | 550.85 |
| **Pola Putus-putus** | **LightGBM Kuantil (q50)** | 3,954 | **792.10** | **6,926.53** | **66.30%** | -39.50% | **396.05** |
| **Pola Putus-putus** | Hurdle Dua Tahap | 3,954 | 1,228.60 | 9,213.13 | 102.83% | +51.86% | 614.30 |
| **Pola Putus-putus** | Rata-rata Bergerak (MA-3) | 3,954 | 1,383.25 | 11,691.73 | 115.77% | +65.44% | 691.62 |
| **Pola Putus-putus** | Forecast Eksisting (BO+OI) | 3,954 | 339.76 | 3,848.05 | 28.44% | -4.26% | 169.88 |
| **Pola Putus-putus** | Business Plan (BP) | 3,354 | 1,259.08 | 9,911.09 | 95.29% | -11.66% | 629.54 |

### B. Ringkasan Backtest Rolling 1-Bulan ke Depan (Semua SKU — `backtest_rolling_1step_summary.csv`)

| Segmen | Model | `n_obs` | `MAE` | `RMSE` | `WAPE_pct` | `Cum_Bias_pct` | `Pinball_q50` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Keseluruhan** | LightGBM Tweedie | 5,712 | 4,452.70 | 67,601.25 | 50.87% | -24.68% | 2,226.35 |
| **Keseluruhan** | LightGBM Kuantil (q50) | 5,712 | 4,200.11 | 62,427.43 | 47.98% | -18.17% | 2,100.06 |
| **Keseluruhan** | Hurdle Dua Tahap | 5,712 | 4,155.52 | 60,821.31 | 47.47% | **+0.44%** | 2,077.76 |
| **Keseluruhan** | Hybrid (AI + BO+OI) | 5,673 | 2,699.46 | 38,179.03 | 30.68% | -12.54% | 1,349.73 |
| **Keseluruhan** | Rata-rata Bergerak (MA-3) | 5,712 | 4,176.38 | 50,530.79 | 47.71% | -5.51% | 2,088.19 |
| **Keseluruhan** | **Forecast Eksisting (BO+OI)** | 5,673 | **2,212.43** | **36,174.22** | **25.15%** | **-1.87%** | **1,106.22** |
| **Keseluruhan** | Business Plan (BP) | 4,422 | 9,433.90 | 99,354.91 | 89.64% | -74.95% | 4,716.95 |

### C. Ringkasan Backtest Rolling 1-Bulan — Khusus SKU Berhistori Train (`backtest_rolling_known_skus_summary.csv`)

| Segmen | Model | `n_obs` | `MAE` | `RMSE` | `WAPE_pct` | `Cum_Bias_pct` | `Pinball_q50` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Keseluruhan** | LightGBM Tweedie | 4,866 | 5,165.69 | 73,239.99 | 50.51% | -24.60% | 2,582.85 |
| **Keseluruhan** | LightGBM Kuantil (q50) | 4,866 | 4,865.49 | 67,632.87 | 47.57% | -18.01% | 2,432.75 |
| **Keseluruhan** | Hurdle Dua Tahap | 4,866 | 4,783.96 | 65,891.96 | 46.78% | **+0.31%** | 2,391.98 |
| **Keseluruhan** | Hybrid (AI + BO+OI) | 4,866 | 3,129.53 | 41,223.05 | 30.60% | -12.48% | 1,564.77 |
| **Keseluruhan** | Rata-rata Bergerak (MA-3) | 4,866 | 4,823.86 | 54,742.52 | 47.17% | -5.49% | 2,411.93 |
| **Keseluruhan** | **Forecast Eksisting (BO+OI)** | 4,866 | **2,561.61** | **39,058.31** | **25.05%** | **-1.96%** | **1,280.80** |
| **Pola Lancar** | LightGBM Tweedie | 912 | 23,779.39 | 168,489.74 | 48.15% | -26.22% | 11,889.70 |
| **Pola Lancar** | LightGBM Kuantil (q50) | 912 | 22,592.81 | 155,541.64 | 45.75% | -15.88% | 11,296.41 |
| **Pola Lancar** | **Hurdle Dua Tahap** | 912 | **21,094.27** | **151,312.03** | **42.71%** | **-1.98%** | **10,547.13** |
| **Pola Lancar** | Hybrid (AI + BO+OI) | 912 | 14,947.12 | 94,789.51 | 30.26% | -11.99% | 7,473.56 |
| **Pola Lancar** | Rata-rata Bergerak (MA-3) | 912 | 21,259.26 | 125,166.78 | 43.05% | -8.04% | 10,629.63 |
| **Pola Lancar** | Forecast Eksisting (BO+OI) | 912 | 12,194.49 | 89,863.36 | 24.69% | -1.72% | 6,097.24 |
| **Pola Putus-putus** | LightGBM Tweedie | 3,954 | 872.40 | 7,306.99 | 73.02% | -9.11% | 436.20 |
| **Pola Putus-putus** | **LightGBM Kuantil (q50)** | 3,954 | **776.64** | **7,002.45** | **65.00%** | -38.34% | **388.32** |
| **Pola Putus-putus** | Hurdle Dua Tahap | 3,954 | 1,021.94 | 7,894.34 | 85.53% | +22.17% | 510.97 |
| **Pola Putus-putus** | Hybrid (AI + BO+OI) | 3,954 | 403.77 | 4,344.22 | 33.79% | -17.10% | 201.89 |
| **Pola Putus-putus** | Rata-rata Bergerak (MA-3) | 3,954 | 1,032.99 | 8,624.41 | 86.46% | +18.76% | 516.50 |
| **Pola Putus-putus** | Forecast Eksisting (BO+OI) | 3,954 | 339.76 | 3,848.05 | 28.44% | -4.26% | 169.88 |

---

## 4. Penjelasan Ilmiah atas Hasil V3

1. **Bukti Empiris Routing Optimal Kebalikan dari Asumsi Awal:**
   - Pada **Pola Lancar**, **`Hurdle Dua Tahap`** adalah model AI murni terbaik baik di mode rekursif (`WAPE 39.25%`, `Bias +0.61%`) maupun *rolling 1-step* (`42.71%` pada SKU berhistori, `Bias -1.98%`).
   - Pada **Pola Putus-putus**, **`LightGBM Kuantil (q50)`** pada skala `log1p` justru menjadi model AI murni dengan error terendah (`WAPE 65.00%` *rolling*, `MAE 776.64`).
   - Artinya, pemilihan model pada `ramalan_terpilih` di `train.py` V3 (yang masih mengarahkan `Lancar → LightGBM Tweedie` dan `Putus-putus → Hurdle`) perlu dibalik di iterasi berikutnya.
2. **Mengapa `LightGBM Tweedie` Tetap *Under-Forecast* (`-24.68%` s/d `-27.01%`)?**
   - Dengan hampir `70%` observasi berasal dari segmen *Putus-putus* (banyak nilai `0`) sementara segmen *Lancar* memiliki volume hingga ratusan ribu unit, pembobotan deviasi Tweedie (`p = 1.35`) pada daun pohon yang bercampur membuat prediksi log-link tertekan ke bawah untuk SKU bervolume besar.
3. **Mengapa `Hybrid (AI + BO+OI)` (`WAPE 30.68%`) Belum Mengalahkan `BO+OI` Murni (`25.15%`)?**
   - Masalah utamanya terlihat langsung pada kolom `Cum_Bias_pct`: **`Hybrid` mengalami bias turun `-12.54%`**, sementara `BO+OI` hanya `-1.87%`.
   - Karena `HybridBOOILGBMForecaster` menghitung residual di ruang log (`log1p(Sales) - log1p(BO+OI)`) dengan batas *smearing* maksimal `1.05`, bulan-bulan di mana `Sales = 0` sedangkan `BO+OI > 0` menciptakan residual log sangat negatif sehingga menarik seluruh koreksi ke bawah.
   - Selain itu, rendahnya bias `BO+OI` (`-1.87%`) membuktikan bahwa angka *Order Intake* perusahaan sudah terkalibrasi cukup baik dan sisa error `25.15%` didominasi dinamika order mendadak setelah tanggal *cut-off*.
4. **Dampak Pemisahan `141` SKU Baru (`SKU Berhistori Train`):**
   - Pada `Pola Lancar`, `141` SKU baru tersebut bervolume relatif kecil (saat dikeluarkan, `MAE` naik dari `11,203` ke `21,094` namun `WAPE` relatif stabil dari `43.50%` menjadi `42.71%`). Peringkat antar model tidak berubah.

---

## 5. Kelebihan, Kekurangan, & Rencana V4

### Kelebihan
- Evaluasi `SKU Berhistori Train` (`4,866` obs) kini 100% *apple-to-apple* di seluruh model.
- Menghasilkan bukti empiris kuat mengenai pasangan model AI murni terbaik per segmen (**`Hurdle Dua Tahap` untuk `Lancar`** dan **`LightGBM Kuantil` untuk `Putus-putus`**).
- Fitur `Hybrid` berjalan bebas *data leakage* dan mampu menekan error jauh di bawah seluruh model AI murni (`WAPE 30.68%` vs `47.47%`), meski masih kalah dari `BO+OI` murni karena bias transformasi log.

### Kekurangan
- Routing `model_terpilih` di `train.py` V3 masih mengarahkan `Lancar` ke `LightGBM Tweedie` (`51.37%` WAPE) dan `Putus-putus` ke `Hurdle` (`102.83%` WAPE), belum memakai pemenang empiris masing-masing segmen.
- `HybridBOOILGBMForecaster` masih memakai residual ruang `log1p` yang memicu bias turun `-12.54%`.

### Rekomendasi Konkret untuk V4
1. **Perbaiki Routing `ramalan_terpilih` & SHAP:**
   - Gunakan **`Hurdle Dua Tahap`** untuk `Pola Lancar` (`WAPE 39.25%`, `Bias +0.61%`) dan **`LightGBM Kuantil (q50)`** untuk `Pola Putus-putus` (`WAPE 65.00%`, `MAE 776.64`).
2. **Ubah `HybridBOOILGBMForecaster` ke Skala Unit Asli (*Bates–Granger / Non-Negative Linear Blending*):**
   - Mengganti koreksi ruang log dengan kombinasi linier terkalibrasi pada skala unit asli (mis. optimasi bobot `w * BO+OI + (1 - w) * AI_Murni` atau regresi residual terklip pada skala relatif/unit tanpa *downward log bias*), sehingga error Hybrid dijamin tidak pernah lebih buruk daripada `BO+OI` (`<= 25.15%`).
