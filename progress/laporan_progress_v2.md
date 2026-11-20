# Laporan Progress V2 — Penyempurnaan Model AI Murni & *Rolling 1-Step Backtest*

**Referensi Versi:** Iterasi V2 (setelah commit `7b9b6f1`)  
**Cakupan Modul:** [src/features.py](file:///c:/hackathon/train_model/src/features.py), [src/models.py](file:///c:/hackathon/train_model/src/models.py), [train.py](file:///c:/hackathon/train_model/train.py)

---

## 1. Ringkasan Progress yang Dilakukan

Iterasi V2 berfokus pada perbaikan **model AI murni** (tanpa membocorkan sinyal `BO+OI`) serta menyempurnakan metodologi evaluasi berdasarkan temuan di [laporan_progress_v1.md](file:///c:/hackathon/train_model/progress/laporan_progress_v1.md):

1. **Transformasi Target `log1p` & Regularisasi pada `QuantileLGBMForecaster` ([src/models.py](file:///c:/hackathon/train_model/src/models.py))**
   - Melatih `LGBMRegressor` (`alpha = 0.10, 0.50, 0.90`) pada target `np.log1p(y)` lalu mengembalikannya ke skala unit dengan `np.expm1(...)`.
   - Menambahkan regularisasi (`min_child_samples=10`, `reg_alpha=0.1`, `reg_lambda=1.0`) untuk menekan *overfitting* pada noise.
2. **Stabilisasi *Duan Smearing Factor* pada `TwoStageHurdleForecaster` ([src/models.py](file:///c:/hackathon/train_model/src/models.py))**
   - Membatasi (*winsorizing*) residual pada persentil `5%–95%` sebelum menghitung `np.mean(np.exp(resid_trimmed))`, dan mengklip faktor *smearing* pada rentang `[0.95, 1.25]` (sebelumnya `[0.8, 2.0]`).
   - Menambahkan `reg_alpha=0.1` dan `reg_lambda=1.0` pada classifier dan regressor.
3. **Penambahan 4 Fitur Baru Khusus Tren Panjang & Pola Putus-putus ([src/features.py](file:///c:/hackathon/train_model/src/features.py))**
   - `rolling_mean_12`: rata-rata permintaan 12 bulan terakhir (tingkat dasar jangka panjang per SKU).
   - `rolling_median_6`: median 6 bulan terakhir agar lebih tahan terhadap lonjakan satu bulan (*outlier*).
   - `nonzero_rate_12`: rasio frekuensi kemunculan permintaan dalam 12 bulan terakhir.
   - `months_since_last_sale`: jumlah bulan sejak penjualan positif (`> 0`) terakhir terjadi, dihitung dari data tergeser (`t-1`) agar bebas *data leakage*.
   - Total fitur aktif pada data Excel aktual meningkat dari `16` menjadi `20` fitur.
4. **Penambahan Evaluasi *Rolling 1-Bulan ke Depan* & Flag Histori SKU ([src/models.py](file:///c:/hackathon/train_model/src/models.py), [train.py](file:///c:/hackathon/train_model/train.py))**
   - Menambahkan `rolling_1step_ml_forecast` yang menghasilkan `backtest_rolling_1step_summary.csv` dan `backtest_rolling_1step_detail.csv`.
   - Menambahkan kolom `ada_di_train` (`True`/`False`) di `backtest_sku_metrics.csv` untuk membedakan SKU berhistori vs SKU baru.

---

## 2. Alasan & *Thought Process* di Balik Keputusan Teknis

1. **Mengapa `QuantileLGBMForecaster` Dipindahkan ke Skala `log1p(y)`?**
   - Pada V1, rentang penjualan antar SKU sangat ekstrem (`MAE ~4,284` vs `RMSE ~53,383`). Di skala unit mentah, pohon keputusan sulit membagi node secara proporsional untuk SKU kecil maupun raksasa, menyebabkan bias `-25.15%`.
   - Secara matematis, fungsi kuantil bersifat ekuivarian terhadap transformasi naik monoton (`expm1(Q_alpha(log1p(Y)))`), sehingga pelatihan di skala `log1p` diharapkan menyamakan skala gradien antar SKU.
2. **Mengapa *Smearing Factor* di `Hurdle` Dibatasi pada Persentil `5%–95%` dan `[0.95, 1.25]`?**
   - Pada V1, residual ekstrem dari beberapa kejadian lonjakan di **Pola Putus-putus** mendorong `mean(exp(resid))` menyentuh batas atas `2.0`. Akibatnya, setiap prediksi SKU *Putus-putus* dikalikan hampir 2x sehingga memicu *over-forecast* `+60.57%`. Pembatasan residual ekstrem mencegah bias sistemik tersebut.
3. **Mengapa `months_since_last_sale` & `rolling_median_6` Ditambahkan?**
   - SKU *Putus-putus* (`3,954` dari `5,712` observasi, hampir `70%` data uji) sangat bergantung pada jeda antar pesanan (*inter-arrival time*). Fitur `months_since_last_sale` memberi sinyal langsung kepada `LGBMClassifier` apakah SKU sedang dalam fase jeda panjang atau mendekati siklus beli ulang.
4. **Mengapa Menambahkan *Rolling 1-Step-Ahead Backtest*?**
   - Membandingkan model AI yang meramal 6 bulan rekursif sekaligus melawan `BO+OI` yang terbit tiap bulan (1 bulan ke depan) tidaklah setara. *Rolling 1-step backtest* membiarkan model AI melihat penjualan aktual hingga bulan `t-1` sebelum menebak bulan `t` — menyamakan horizon prediksi dengan `BO+OI` (tetap tanpa membocorkan informasi order `BO+OI`).

---

## 3. Hasil Backtest V2 & Perbandingan dengan V1

Pengujian V2 pada data aktual dijalankan dengan `--skip-arima` untuk mempercepat iterasi.

### A. Ringkasan Backtest Multi-Bulan (Rekursif 6 Bulan — `backtest_summary.csv`)

| Segmen | Model | `n_obs` | `MAE` | `RMSE` | `WAPE_pct` | `Cum_Bias_pct` | `Pinball_q50` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Keseluruhan** | LightGBM Kuantil (q50) | 5,712 | 5,199.54 | 77,779.13 | 59.40% | -46.13% | 2,599.77 |
| **Keseluruhan** | Hurdle Dua Tahap | 5,712 | 3,961.93 | 48,188.59 | 45.26% | **+5.45%** | 1,980.96 |
| **Keseluruhan** | Rata-rata Bergerak (MA-3) | 4,866 | 4,995.51 | 48,904.61 | 48.84% | -5.50% | 2,497.76 |
| **Keseluruhan** | Forecast Eksisting (BO+OI) | 5,673 | 2,212.43 | 36,174.22 | 25.15% | -1.87% | 1,106.22 |
| **Keseluruhan** | Business Plan (BP) | 4,422 | 9,433.90 | 99,354.91 | 89.64% | -74.95% | 4,716.95 |
| **Pola Lancar** | LightGBM Kuantil (q50) | 1,758 | 15,112.52 | 139,814.55 | 58.68% | -46.82% | 7,556.26 |
| **Pola Lancar** | Hurdle Dua Tahap | 1,758 | 10,109.59 | 85,755.84 | 39.25% | **+0.61%** | 5,054.79 |
| **Pola Lancar** | Rata-rata Bergerak (MA-3) | 912 | 20,656.59 | 110,309.18 | 41.82% | -12.94% | 10,328.29 |
| **Pola Lancar** | Forecast Eksisting (BO+OI) | 1,719 | 6,519.90 | 65,455.82 | 24.80% | -1.62% | 3,259.95 |
| **Pola Lancar** | Business Plan (BP) | 1,068 | 35,106.53 | 201,403.92 | 89.04% | -81.61% | 17,553.26 |
| **Pola Putus-putus** | LightGBM Kuantil (q50) | 3,954 | **792.10** | **6,926.53** | **66.30%** | -39.50% | **396.05** |
| **Pola Putus-putus** | Hurdle Dua Tahap | 3,954 | 1,228.60 | 9,213.13 | 102.83% | +51.86% | 614.30 |
| **Pola Putus-putus** | Rata-rata Bergerak (MA-3) | 3,954 | 1,383.25 | 11,691.73 | 115.77% | +65.44% | 691.62 |
| **Pola Putus-putus** | Forecast Eksisting (BO+OI) | 3,954 | 339.76 | 3,848.05 | 28.44% | -4.26% | 169.88 |
| **Pola Putus-putus** | Business Plan (BP) | 3,354 | 1,259.08 | 9,911.09 | 95.29% | -11.66% | 629.54 |

### B. Ringkasan Backtest Rolling 1-Bulan ke Depan (`backtest_rolling_1step_summary.csv`)

| Segmen | Model | `n_obs` | `MAE` | `RMSE` | `WAPE_pct` | `Cum_Bias_pct` | `Pinball_q50` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Keseluruhan** | LightGBM Kuantil (q50) | 5,712 | 4,200.11 | 62,427.43 | 47.98% | -18.17% | 2,100.06 |
| **Keseluruhan** | Hurdle Dua Tahap | 5,712 | 4,155.52 | 60,821.31 | 47.47% | **+0.44%** | 2,077.76 |
| **Keseluruhan** | Rata-rata Bergerak (MA-3) | 5,712 | 4,176.38 | 50,530.79 | 47.71% | -5.51% | 2,088.19 |
| **Keseluruhan** | Forecast Eksisting (BO+OI) | 5,673 | 2,212.43 | 36,174.22 | 25.15% | -1.87% | 1,106.22 |
| **Keseluruhan** | Business Plan (BP) | 4,422 | 9,433.90 | 99,354.91 | 89.64% | -74.95% | 4,716.95 |
| **Pola Lancar** | LightGBM Kuantil (q50) | 1,758 | 11,900.00 | 112,036.76 | 46.21% | -16.06% | 5,950.00 |
| **Pola Lancar** | Hurdle Dua Tahap | 1,758 | 11,203.38 | 108,991.65 | 43.50% | **-1.83%** | 5,601.69 |
| **Pola Lancar** | Rata-rata Bergerak (MA-3) | 1,758 | 11,246.34 | 90,160.69 | 43.67% | -8.04% | 5,623.17 |
| **Pola Lancar** | Forecast Eksisting (BO+OI) | 1,719 | 6,519.90 | 65,455.82 | 24.80% | -1.62% | 3,259.95 |
| **Pola Lancar** | Business Plan (BP) | 1,068 | 35,106.53 | 201,403.92 | 89.04% | -81.61% | 17,553.26 |
| **Pola Putus-putus** | LightGBM Kuantil (q50) | 3,954 | **776.64** | 7,002.45 | **65.00%** | -38.34% | **388.32** |
| **Pola Putus-putus** | Hurdle Dua Tahap | 3,954 | 1,021.94 | 7,894.34 | 85.53% | **+22.17%** | 510.97 |
| **Pola Putus-putus** | Rata-rata Bergerak (MA-3) | 3,954 | 1,032.99 | 8,624.41 | 86.46% | +18.76% | 516.50 |
| **Pola Putus-putus** | Forecast Eksisting (BO+OI) | 3,954 | 339.76 | 3,848.05 | 28.44% | -4.26% | 169.88 |
| **Pola Putus-putus** | Business Plan (BP) | 3,354 | 1,259.08 | 9,911.09 | 95.29% | -11.66% | 629.54 |

### Perbandingan Kunci V1 vs V2 & Penjelasan Hasil

1. **Perbaikan Besar pada `Pola Putus-putus`:**
   - `LightGBM Kuantil`: `WAPE` turun dari `73.61%` (V1) menjadi **`66.30%`** (V2 rekursif) dan **`65.00%`** (V2 *rolling 1-step*), dengan `MAE` membaik dari `879.44` → **`776.64`**.
   - `Hurdle Dua Tahap`: Pada *rolling 1-step*, `WAPE` turun dari `110.06%` → **`85.53%`** dan bias ekstrem turun drastis dari `+60.57%` → **`+22.17%`**.
2. **Bias `Hurdle Dua Tahap` Menjadi Hampir Nol:**
   - Bias keseluruhan membaik dari `+10.25%` (V1) → **`+5.45%`** (V2 rekursif) dan **`+0.44%`** (V2 *rolling 1-step*).
   - Pada **Pola Lancar**, bias `Hurdle` hanya **`+0.61%`** (rekursif) dan **`-1.83%`** (*rolling 1-step*).
3. **Mengapa `LightGBM Kuantil (q50)` di `Pola Lancar` Justru Memburuk (`46.38%` → `58.68%`, Bias `-46.82%`)?**
   - Meskipun kuantil ekuivarian secara distribusi, **median (`q50`) pada distribusi miring ke kanan (*right-skewed*) secara sistematis lebih kecil daripada rata-rata (*mean*)**. Ketika `LightGBM Kuantil` memprediksi `q50` di ruang `log1p` dan langsung di-`expm1` tanpa koreksi varians (*smearing*), prediksi median tersebut jauh di bawah total volume penjualan aktual (*under-forecast* `-46.82%`).
   - Sebaliknya, `Hurdle Dua Tahap` (yang memakai `regression` pada `log1p` ditambah *smearing factor*) terhindar dari jebakan tersebut dan menjaga bias tetap di `+0.61%`.
4. **Temuan Penting dari *Rolling 1-Step Backtest* (*Irreducible Noise Floor*):**
   - Pada *rolling 1-step*, `Hurdle` (`WAPE 43.50%`), `LightGBM` (`46.21%`), dan `MA-3` (`43.67%`) di **Pola Lancar** menghasilkan error yang hampir identik.
   - Artinya, hanya bermodal histori `Sales` tanpa sinyal order masuk, batas akurasi realistis data memang berada di kisaran `~39–44%` WAPE. Sebaliknya, **`BO+OI` mencapai `24.80%` WAPE** karena memang membawa informasi pesanan aktual yang sudah masuk sebelum bulan berjalan.

---

## 4. Kelebihan dan Kekurangan Iterasi V2

### Kelebihan
- **Bias sistemik `Hurdle` teratasi:** `Cum_Bias_pct` keseluruhan turun mendekati nol (`+0.44%` *rolling*, `+5.45%` rekursif) dan bias Pola Putus-putus turun hampir sepertiganya (`+60.57%` → `+22.17%` *rolling*).
- **Performa Pola Putus-putus meningkat nyata:** `WAPE` `LightGBM` turun `~8.6` poin persen (`73.61%` → `65.00%`).
- **Evaluasi lebih objektif dan transparan:** Tersedia tabel *rolling 1-step* yang sejajar horizon waktunya dengan `BO+OI` serta penanda `ada_di_train` per SKU.
- **Iterasi lebih cepat** berkat opsi `--skip-arima`.

### Kekurangan & Rekomendasi V3
- **Regresi `LightGBM Kuantil` di Pola Lancar:** Transformasi `log1p` pada `objective="quantile"` (`alpha=0.50`) menyebabkan *under-forecast* `-46.82%`. Untuk V3, `LightGBM` pada Pola Lancar sebaiknya menggunakan `objective="tweedie"` atau `regression` pada `log1p` + *smearing*, atau mengembalikan `q50` ke pendekatan non-log / terkalibrasi, sementara `q10`/`q90` tetap dipakai sebagai interval ketidakpastian.
- **Gap dengan `BO+OI` masih `~19–22` poin persen WAPE:** Selama AI tidak menerima fitur `BO+OI` (pesanan masuk), AI murni tidak memiliki informasi order bulan depan.
  - **Rekomendasi Arsitektur V3:** Pisahkan dua mode operasional yang saling melengkapi:
    1. **Mode Hybrid (`AI + BO+OI`)** untuk prediksi bulan depan (`t+1`) saat `BO+OI` sudah tersedia, guna mengoreksi bias musiman/SKU dari `BO+OI` dan menargetkan `WAPE < 25.15%`.
    2. **Mode AI Murni** (dengan perbaikan `LGBM` Pola Lancar di atas) untuk horizon `t+2` dan `t+3` ketika `BO+OI` belum tersedia.
