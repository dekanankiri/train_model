
# Konteks Proyek & Arsitektur AI: StockWise AI

Dokumen ini merangkum hasil inspeksi desain UI/UX di **FigCraft** serta pembahasan teknis mengenai arsitektur model AI, kebutuhan data, referensi jurnal ilmiah, dan kelayakan perangkat keras untuk membangun prototipe **StockWise AI**.

---

## 1. Status Desain FigCraft yang Diinspeksi

- **Koneksi**: `figcraft.exe --control 7979`
- **Window Aktif**: `StockWise AI • – FigCraft` (`id: 0`)
- **Tab Terbuka**: `StockWise AI` (1 tab; status `dirty: true`, `path: null`, halaman aktif Page `2`, tool aktif `move`, zoom ~40%)

---

## 2. Komponen Model AI & Analitik di Desain `StockWise AI`

### A. Demand Forecasting & Evaluasi Backtest

1. **LightGBM Kuantil** *(Model Utama — Pola Lancar)*
   - Dipilih sebagai model terbaik karena memiliki galat *backtest* terendah untuk SKU dengan pola permintaan **Lancar**.
2. **Hurdle / Hurdle Dua Tahap** *(Model Utama — Pola Putus-putus)*
   - Tercantum pada layar `BR-2b · Ramalan per SKU (Putus-putus)` (`Baris Hurdle` dan `Baris Hurdle dua tahap`).
   - Memisahkan prediksi menjadi dua tahap: probabilitas munculnya permintaan (*occurrence*) dan besaran volumenya (*magnitude*).
3. **ARIMA** *(Model Pembanding)*
   - Ditampilkan di tabel perbandingan model (`Baris ARIMA`) sebagai pembanding *time-series* klasik.
4. **Rata-rata Bergerak (Moving Average)** *(Model Pembanding)*
   - Ditampilkan pada baris perbandingan berikutnya sebagai *baseline* statistik sederhana.

### B. Explainable AI (XAI)

- **SHAP (SHapley Additive exPlanations)**
  - Tercantum pada `Tab Explainable AI` dan layar `BR-2 Ramalan per SKU (Explainable AI)`.
  - Keterangan di UI: *"Faktor penggerak ramalan Nov 2026 – Jan 2027 · dihitung dengan SHAP"*.
  - Berperan sebagai metode *post-hoc explainability* di atas model forecasting (bukan model yang di-training terpisah) untuk menjelaskan kontribusi fitur terhadap hasil ramalan per SKU.

### C. Segmentasi & Rekomendasi Penyaluran Stok Berlebih

- **RFM (Recency, Frequency, Monetary)**
  - Menjawab masalah: *"sisa stok dijual ke pelanggan yang salah"*.
  - Peran di UI: *"Distributor dikelompokkan dengan RFM, diperingkat berdasarkan potensi pembelian, lalu ditawarkan stok berlebih."*
  - Menampilkan komponen **`Pengaruh RFM pada peringkat`**: `Recency`, `Frequency`, dan `Monetary`.

---

## 3. Ringkasan Kebutuhan Training Model

| Komponen                     | Kategori                       | Perlu Training ML? | Keterangan                                                         |
| :--------------------------- | :----------------------------- | :-----------------: | :----------------------------------------------------------------- |
| **LightGBM Kuantil**   | Forecasting (Pola Lancar)      |    **Ya**    | Model utama *supervised ML* berbasis GBDT                        |
| **Hurdle (dua tahap)** | Forecasting (Pola Putus-putus) |    **Ya**    | Klasifikasi kejadian + regresi volume permintaan                   |
| **ARIMA**              | Pembanding *Time-Series*     | *Fitting* per SKU | Di-*fit* pada histori seri waktu untuk *backtest*              |
| **Rata-rata bergerak** | *Baseline* Statistik         |   **Tidak**   | Perhitungan rata-rata bergerak langsung                            |
| **SHAP**               | Explainable AI                 |   **Tidak**   | Dihitung *post-hoc* (mis. `TreeExplainer`) dari model terlatih |
| **RFM**                | Segmentasi & Peringkat         |   **Tidak**   | Agregasi metrik R, F, M + skoring/*clustering*                   |

**Kesimpulan:** Secara inti terdapat **2 model ML utama** yang harus di-training (**LightGBM Kuantil** dan **Hurdle dua tahap**), ditambah *fitting* **ARIMA** bila disertakan sebagai pembanding di *backtest*.

---

## 4. Kebutuhan Data untuk Prototipe

1. **Data Historis Permintaan / Penjualan per SKU** *(untuk Forecasting, Backtest, & SHAP)*
   - `tanggal/periode`, `SKU_ID`, `nama_produk`, `kategori`
   - `jumlah_terjual` / `demand` (wajib mencakup periode bernilai `0` agar pola *Putus-putus* terbaca)
   - Fitur konteks tambahan *(opsional, memperkaya SHAP & LightGBM)*: `harga`, `promo/diskon`, `hari_libur/musim`, `lead_time`
2. **Data Stok / Inventori** *(untuk deteksi stok berlebih & rekomendasi)*
   - `SKU_ID`, `stok_tersedia` (*on-hand*), `safety_stock` / `reorder_point`, `umur_stok` atau `tanggal_kedaluwarsa`, `lokasi/gudang`
3. **Data Transaksi Distributor / Pelanggan** *(untuk RFM)*
   - `distributor_ID` / `customer_ID`, `nama_distributor`
   - `tanggal_transaksi` (untuk *Recency* & *Frequency*)
   - `nilai_transaksi` / `total_pembelian` (untuk *Monetary*)
   - `SKU_ID` & `qty` yang biasa dibeli (untuk relevansi penawaran stok berlebih)

---

## 5. Referensi Jurnal Ilmiah

### A. LightGBM Kuantil & *Probabilistic Forecasting*

1. **Makridakis, Spiliotis, Assimakopoulos, Petropoulos, et al. (2022)** — *The M5 uncertainty competition: Results, findings and conclusions*, **International Journal of Forecasting**, Vol. 38, Issue 4.
2. **Lainder & Wolfinger (2022)** — *Forecasting with gradient boosted trees: augmentation, tuning, and cross-validation strategies: Winning solution to the M5 Uncertainty competition*, **International Journal of Forecasting**, Vol. 38, Issue 4.
3. **Chiew & Choong (2022)** — *A solution for M5 Forecasting - Uncertainty: Hybrid gradient boosting and autoregressive recurrent neural network for quantile estimation*, **International Journal of Forecasting**, Vol. 38, Issue 4.

### B. Pola Permintaan (*Lancar vs. Putus-putus*) & Model *Hurdle*

4. **Syntetos, Boylan, & Croston (2005)** — *On the Categorization of Demand Patterns*, **Journal of the Operational Research Society**, Vol. 56, No. 5, pp. 495–503 (`DOI: 10.1057/palgrave.jors.2601841`). *(Dasar klasifikasi SBC berbasis ADI & $CV^2$).*
5. **Snyder, Ord, & Beaumont (2012)** — *Forecasting the intermittent demand for slow-moving inventories: A modelling approach*, **International Journal of Forecasting**. *(Model hurdle-shifted Poisson).*
6. **Sarlo, Fernandes, & Borenstein (2023)** — *Lumpy and intermittent retail demand forecasts with score-driven models*, **European Journal of Operational Research**. *(Model hurdle Poisson & hurdle negative binomial).*

### C. Explainable AI (SHAP) pada *Demand Forecasting* & Inventori

7. **Arboleda-Florez & Castro-Zuluaga (2023)** — *Interpreting direct sales' demand forecasts using SHAP values*, **Production** (`DOI: 10.1590/0103-6513.20220035`).
8. **Ntakolia, Kokkotis, Karlsson, & Moustakidis (2021)** — *An Explainable Machine Learning Model for Material Backorder Prediction in Inventory Management*, **Sensors** (`DOI: 10.3390/s21237926`).

### D. Segmentasi RFM & B2B

9. **Cheng & Chen (2009)** — *Classifying the segmentation of customer value via RFM model and RS theory*, **Expert Systems with Applications**, Vol. 36, Issue 3, pp. 4176–4184.
10. **Abbasimehr & Bahrini (2022)** — *An analytical framework based on the recency, frequency, and monetary model and time series clustering techniques for dynamic segmentation*, **Expert Systems with Applications**, Vol. 192, 116373.
11. **Rungruang, Riyapan, Intarasit, Chuarkham, & Muangprathub (2024)** — *RFM model customer segmentation based on hierarchical approach using FCA*, **Expert Systems with Applications**, Vol. 237, 121449.
12. **Kandeil, Saad, & Youssef (2014)** — *A Two-Phase Clustering Analysis for B2B Customer Segmentation*, **IEEE International Conference on Intelligent Networking and Collaborative Systems (INCoS)**, pp. 221–228.

---

## 6. Kelayakan Hardware (Laptop Intel Core i3-1215U)

- **Status**: **Sangat layak** untuk training dan inferensi prototipe (tidak memerlukan GPU).
- **Kinerja per komponen**:
  - **LightGBM Kuantil & Hurdle**: Sangat cepat (detik hingga ~1 menit untuk puluhan ribu baris).
  - **ARIMA**: Ringan, namun jika jumlah SKU >1.000 dengan `auto_arima`, batasi ke sampel SKU atau jalankan paralel (`n_jobs`).
  - **SHAP**: Gunakan `TreeExplainer` agar cepat (hindari `KernelExplainer` pada data besar).
  - **RFM**: Instan (<1 detik).
- **Catatan Memori**: Jika menggunakan RAM 8 GB, jaga efisiensi tipe data (`float32`/`category`) dan hindari pembuatan puluhan fitur *lag/rolling* pada jutaan baris sekaligus.
