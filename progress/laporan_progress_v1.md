# Laporan Progress V1 — Adaptasi Pipeline ke Data Aktual & Benchmark `BO+OI`

**Referensi Versi:** Commit `7b9b6f1` (`feat: add Excel template, actual data loader, and BO+OI backtest benchmark`)  
**Cakupan Modul:** [scripts/create_template.py](file:///c:/hackathon/train_model/scripts/create_template.py), [src/loader.py](file:///c:/hackathon/train_model/src/loader.py), [src/features.py](file:///c:/hackathon/train_model/src/features.py), [src/models.py](file:///c:/hackathon/train_model/src/models.py), [src/rfm.py](file:///c:/hackathon/train_model/src/rfm.py), [train.py](file:///c:/hackathon/train_model/train.py)

---

## 1. Ringkasan Progress yang Dilakukan

Pada Iterasi V1, fokus utama adalah **memigrasikan prototipe `StockWise AI`** (yang sebelumnya hanya berjalan pada dataset sintetis di `data/*.csv`) agar mampu memproses **data aktual perusahaan periode 2020–2026** secara aman, bersih, dan terukur terhadap *benchmark* bisnis eksisting.

Langkah-langkah yang telah diselesaikan:
1. **Perancangan Template Data Aktual ([scripts/create_template.py](file:///c:/hackathon/train_model/scripts/create_template.py))**
   - Menghasilkan [StockWise_Template_Data_Aktual.xlsx](file:///c:/hackathon/train_model/templates/StockWise_Template_Data_Aktual.xlsx) dengan 4 sheet (`0_Panduan_Skema`, `1_Demand_Bulanan`, `2_Inventory_Snapshot`, `3_Transaksi_Distributor`).
   - Memetakan struktur *wide multi-header* laporan aktual (`WD SMI/SMII`: `Item Code`, `Line`, `Sales`, `Stock`, `BP`, `BO+OI`, `FCT Prod`, `ACT Prod`, `STD`, `60%XSTD Min`, `Turn Over`) ke format *long* berbasis primary key `(SKU_ID, periode)`.
2. **Pembangunan Loader & Validator Data ([src/loader.py](file:///c:/hackathon/train_model/src/loader.py))**
   - Menambahkan `load_dataset`, `clean_demand_frame`, dan `derive_inventory_snapshot_from_demand`.
   - Menormalkan alias nama kolom (mis. `DATE` → `periode`, `tur_over_hari` → `turn_over_hari`), membuang baris metadata template/contoh, memvalidasi keunikan `(SKU_ID, periode)`, dan mengurutkan data secara kronologis.
3. **Pemilihan Fitur Adaptif & Kalender Otomatis ([src/features.py](file:///c:/hackathon/train_model/src/features.py))**
   - Menambahkan `select_feature_cols` dan `enrich_calendar_features` agar fitur kalender Indonesia 2020–2027 (`hari_libur_bulan`, `is_musim_puncak`, `LEBARAN_PEAK_MONTHS`) terisi otomatis dan kolom eksternal kosong (`harga`, `promo`, `diskon_pct`, `lead_time_hari`) dilewati tanpa error.
4. **Dukungan Benchmark Eksisting & Mode Tanpa Transaksi ([src/models.py](file:///c:/hackathon/train_model/src/models.py), [src/rfm.py](file:///c:/hackathon/train_model/src/rfm.py), [train.py](file:///c:/hackathon/train_model/train.py))**
   - Menambahkan evaluasi `Forecast Eksisting (BO+OI)` (`fct_existing`) dan `Business Plan (BP)` (`bp_sales`) pada `compute_backtest_metrics`.
   - Menambahkan `build_overstock_summary` untuk menghasilkan `overstock_summary.csv` ketika `3_Transaksi_Distributor` belum tersedia.
   - Menambahkan argumen CLI (`--excel`, `--out-dir`, `--artifacts-dir`, `--backtest-months`, `--forecast-months`, `--skip-arima`) agar data dan output sensitif dapat disimpan **di luar workspace**.

---

## 2. Alasan & *Thought Process* di Balik Keputusan Teknis

1. **Mengapa Data Aktual Disimpan di Luar Workspace & Memakai Argumen CLI?**
   - **Privasi Data:** File di dalam workspace dapat dibaca oleh tool AI. Dengan menyimpan Excel asli, output CSV, dan artefak `.joblib` di path lokal terpisah melalui `--excel`, `--out-dir`, dan `--artifacts-dir`, seluruh data sensitif perusahaan tetap terisolasi di komputer pengguna.
2. **Mengapa Error `#DIV/0!` dan Sel Kosong Diubah Menjadi `NaN` (Bukan `0`)?**
   - Pada kolom `turn_over_hari` (ketahanan stok dalam hari), `#DIV/0!` terjadi saat `Sales = 0` sementara `Stock > 0`. Mengubahnya menjadi `0` akan keliru menafsirkan stok mengendap sebagai stok bertahan `0` hari.
   - Pada `fct_existing` (`BO+OI`) dan `bp_sales`, sel kosong berarti tidak ada forecast pada bulan tersebut. Dengan mempertahankannya sebagai `NaN`, `compute_backtest_metrics` hanya mengevaluasi baris yang terisi (`notna()`), sehingga metrik `WAPE` dan `MAE` tetap adil.
   - Seluruh kolom angka dikonversi dengan `pd.to_numeric(..., errors="coerce")` agar tidak terbaca sebagai string akibat teks error Excel.
3. **Mengapa `stok_akhir`, `act_prod`, dan `turn_over_hari` Bulan Berjalan Tidak Dijadikan Fitur Prediksi `Sales`?**
   - Nilai `stok_akhir(t)`, `act_prod(t)`, dan `turn_over_hari(t)` baru diketahui pada akhir bulan `t` (atau dihitung langsung dari `Sales(t)`). Memasukkannya sebagai fitur input untuk meramal `Sales(t)` akan menyebabkan ***data leakage*** yang membuat skor backtest bagus secara semu namun gagal saat dipakai meramal masa depan.
4. **Mengapa `BO+OI` Dijadikan *Benchmark* Utama (Bukan `FCT Prod`)?**
   - Dari inspeksi struktur kolom dan karakteristik *hardcoded*, `BO+OI` (*Back Order + Order Intake* pada akhir bulan sebelumnya) adalah proyeksi **permintaan/penjualan (`Sales`)**, sedangkan `FCT Prod` adalah rencana **produksi pabrik** yang juga mempertimbangkan target penumpukan/pengurangan stok. Pembanding yang *apple-to-apple* terhadap target `jumlah_terjual` (`Sales`) adalah `BO+OI` dan `BP`.
5. **Mengapa Menggunakan `16` Fitur Inti yang Diturunkan dari Histori + Kalender?**
   - Karena data aktual belum memiliki kolom `harga`, `promo`, `diskon_pct`, dan `lead_time_hari`, `select_feature_cols` mengaktifkan `16` fitur yang selalu tersedia untuk setiap SKU (`SKU_ID_cat`, `kategori_cat`, `bulan`, `sin_bulan`, `cos_bulan`, `hari_libur_bulan`, `is_musim_puncak`, `lag_1`, `lag_2`, `lag_3`, `lag_6`, `lag_12`, `rolling_mean_3`, `rolling_mean_6`, `rolling_std_3`, `nonzero_rate_6`), sehingga pipeline tetap berjalan mulus untuk ramalan rekursif 3 bulan ke depan.

---

## 3. Hasil Backtest V1 & Penjelasannya

Pengujian dijalankan pada data aktual (evaluasi 6 bulan terakhir; total `5,712` observasi pada model ML ≈ `952` SKU × 6 bulan).

### Tabel Ringkasan Backtest V1 (`backtest_summary.csv`)

| Segmen | Model | `n_obs` | `MAE` | `RMSE` | `WAPE_pct` | `Cum_Bias_pct` | `Pinball_q50` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Keseluruhan** | LightGBM Kuantil (q50) | 5,712 | 4,284.70 | 53,383.09 | 48.95% | -25.15% | 2,142.35 |
| **Keseluruhan** | Hurdle Dua Tahap | 5,712 | 3,936.05 | 45,466.48 | 44.97% | +10.25% | 1,968.02 |
| **Keseluruhan** | ARIMA | 4,866 | 4,766.93 | 49,346.65 | 46.61% | -15.86% | 2,383.47 |
| **Keseluruhan** | Rata-rata Bergerak (MA-3) | 4,866 | 4,995.51 | 48,904.61 | 48.84% | -5.50% | 2,497.76 |
| **Keseluruhan** | **Forecast Eksisting (BO+OI)** | 5,673 | **2,212.43** | **36,174.22** | **25.15%** | **-1.87%** | **1,106.22** |
| **Keseluruhan** | Business Plan (BP) | 4,422 | 9,433.90 | 99,354.91 | 89.64% | -74.95% | 4,716.95 |
| **Pola Lancar** | LightGBM Kuantil (q50) | 1,758 | 11,943.63 | 95,637.72 | 46.38% | -24.83% | 5,971.82 |
| **Pola Lancar** | Hurdle Dua Tahap | 1,758 | 9,831.19 | 80,541.67 | 38.17% | +5.00% | 4,915.59 |
| **Pola Lancar** | ARIMA | 912 | 20,645.04 | 112,527.38 | 41.80% | -20.83% | 10,322.52 |
| **Pola Lancar** | Rata-rata Bergerak (MA-3) | 912 | 20,656.59 | 110,309.18 | 41.82% | -12.94% | 10,328.29 |
| **Pola Lancar** | **Forecast Eksisting (BO+OI)** | 1,719 | **6,519.90** | **65,455.82** | **24.80%** | **-1.62%** | **3,259.95** |
| **Pola Lancar** | Business Plan (BP) | 1,068 | 35,106.53 | 201,403.92 | 89.04% | -81.61% | 17,553.26 |
| **Pola Putus-putus** | LightGBM Kuantil (q50) | 3,954 | 879.44 | 7,078.58 | 73.61% | -28.24% | 439.72 |
| **Pola Putus-putus** | Hurdle Dua Tahap | 3,954 | 1,314.99 | 10,105.37 | 110.06% | +60.57% | 657.50 |
| **Pola Putus-putus** | ARIMA | 3,954 | 1,104.61 | 8,725.52 | 92.45% | +31.58% | 552.30 |
| **Pola Putus-putus** | Rata-rata Bergerak (MA-3) | 3,954 | 1,383.25 | 11,691.73 | 115.77% | +65.44% | 691.62 |
| **Pola Putus-putus** | **Forecast Eksisting (BO+OI)** | 3,954 | **339.76** | **3,848.05** | **28.44%** | **-4.26%** | **169.88** |
| **Pola Putus-putus** | Business Plan (BP) | 3,354 | 1,259.08 | 9,911.09 | 95.29% | -11.66% | 629.54 |

### Penjelasan Hasil Utama
1. **Model AI mengalahkan `ARIMA`, `MA-3`, dan `BP` pada segmen target masing-masing:**
   - Di **Pola Lancar**, `Hurdle Dua Tahap` mencatat `WAPE = 38.17%` (mengungguli `ARIMA` `41.80%` dan `MA-3` `41.82%`).
   - Di **Pola Putus-putus**, `LightGBM Kuantil` mencatat `WAPE = 73.61%` (mengungguli `ARIMA` `92.45%` dan `MA-3` `115.77%`).
   - `Business Plan (BP)` memiliki error sangat tinggi (`WAPE = 89.64%`, `Cum_Bias = -74.95%`), menandakan banyak nilai kosong/nol atau target awal tahun yang tidak diperbarui.
2. **Mengapa `BO+OI` (`WAPE 25.15%`) jauh lebih rendah daripada model AI (`44.97%`–`48.95%`)?**
   - `BO+OI` dibuat pada **H-1 sebelum bulan berjalan** dan sudah memuat informasi pesanan riil masuk (*Order Intake* + *Back Order*), sedangkan model AI pada V1 belum diberi fitur `BO+OI`.
   - Selain itu, model AI diuji meramal **6 bulan berturut-turut secara rekursif** tanpa melihat penjualan aktual bulan di tengah periode uji, sedangkan `BO+OI` diperbarui setiap bulan untuk 1 bulan ke depan.
3. **Tiga Akar Masalah Teknis yang Teridentifikasi di V1:**
   - **Disparitas `n_obs` (`5,712` vs `4,866`):** Ada `141` SKU di **Pola Lancar** (`1,758 - 912 = 846` observasi / 6 bulan) yang baru muncul di 6 bulan uji dan tidak memiliki histori di data latih (`train_raw`), sehingga `ARIMA`/`MA-3` menghasilkan `NaN` sedangkan ML harus memprediksi dengan fitur lag kosong.
   - **Ketimpangan Skala Ekstrem pada `LightGBM Kuantil`:** Karena dilatih langsung pada unit mentah (SKU puluhan hingga ratusan ribu unit), `LightGBM Kuantil` mengalami bias *under-forecast* sebesar `-25.15%`.
   - **Ledakan *Smearing Factor* pada `Hurdle` di Pola Putus-putus:** Nilai `np.mean(np.exp(resid))` yang terklip hingga `2.0` melipatgandakan prediksi pola intermiten sehingga `Cum_Bias_pct` melonjak ke `+60.57%` (`WAPE = 110.06%`).

---

## 4. Kelebihan dan Kekurangan Iterasi V1

### Kelebihan
- Pipeline berjalan *end-to-end* pada data aktual nyata dengan menjaga isolasi privasi data di luar workspace.
- Toleran terhadap kondisi data dunia nyata (`DATE` acak, `#DIV/0!`, campuran `0` dan kosong, tanpa `tanggal_kedaluwarsa`, tanpa `3_Transaksi_Distributor`).
- Langsung menyediakan *benchmark* bisnis (`BO+OI` dan `BP`) di tabel evaluasi sehingga posisi model AI terukur secara objektif.
- Model AI V1 sudah lebih baik daripada `ARIMA`, `MA-3`, dan `BP` secara keseluruhan (`Hurdle` `WAPE 44.97%` vs `ARIMA` `46.61%`).

### Kekurangan
- `LightGBM Kuantil` menderita bias negatif besar (`-25.15%`) akibat variasi skala volume antar SKU yang ekstrem.
- `TwoStageHurdleForecaster` menderita bias positif ekstrem (`+60.57%`) pada **Pola Putus-putus** akibat *smearing factor* tidak tahan *outlier*.
- Belum tersedia fitur khusus intermitensi jangka panjang (`months_since_last_sale`, *rolling 12 bulan*).
- Evaluasi backtest hanya memakai mode rekursif 6 bulan, sehingga perbandingan terhadap `BO+OI` (1 bulan ke depan) belum *apple-to-apple*.
- Fitting `ARIMA` pada hampir seribu SKU memakan waktu cukup lama (`~312.56 detik` total run).
