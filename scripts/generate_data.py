"""Generator dataset sintetis 5 tahun (bulanan) untuk prototipe StockWise AI.

Menghasilkan 3 dataset utama sesuai kebutuhan di konteks.md:
1. data/demand_monthly.csv        : Historis permintaan bulanan per SKU (Nov 2021 - Okt 2026, 60 bulan)
2. data/inventory_snapshot.csv    : Snapshot stok gudang per 2026-10-31 (termasuk kasus stok berlebih)
3. data/distributor_transactions.csv : Transaksi B2B distributor yang konsisten dengan demand bulanan
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


SEED = 42
START_PERIOD = "2021-11-01"
N_MONTHS = 60  # Nov 2021 - Okt 2026 (horizon ramalan berikutnya: Nov 2026 - Jan 2027)
SNAPSHOT_DATE = pd.Timestamp("2026-10-31")

# Bulan dengan lonjakan musiman di Indonesia (Ramadan/Lebaran & Akhir Tahun)
LEBARAN_PEAK_MONTHS = {
    (2022, 4), (2022, 5),
    (2023, 3), (2023, 4),
    (2024, 3), (2024, 4),
    (2025, 3),
    (2026, 2), (2026, 3),
}

WAREHOUSES = ["Gudang Jakarta", "Gudang Surabaya", "Gudang Bandung", "Gudang Medan"]


def build_sku_catalog() -> pd.DataFrame:
    """Mendefinisikan 40 SKU (24 pola Lancar, 16 pola Putus-putus)."""
    smooth_specs = [
        ("SKU-001", "Beras Premium 5kg", "Bahan Pokok", 72000, 420, 0.004, 0.25, 3),
        ("SKU-002", "Minyak Goreng 2L", "Bahan Pokok", 36000, 510, 0.005, 0.22, 3),
        ("SKU-003", "Gula Pasir 1kg", "Bahan Pokok", 17500, 460, 0.003, 0.20, 2),
        ("SKU-004", "Tepung Terigu 1kg", "Bahan Pokok", 14000, 380, 0.003, 0.28, 3),
        ("SKU-005", "Mie Instan Kardus (40pcs)", "Makanan & Minuman", 118000, 620, 0.006, 0.18, 2),
        ("SKU-006", "Air Mineral Galon 19L", "Makanan & Minuman", 21000, 740, 0.007, 0.12, 1),
        ("SKU-007", "Kopi Bubuk Sachet Box", "Makanan & Minuman", 48000, 310, 0.004, 0.15, 4),
        ("SKU-008", "Susu UHT Full Cream 1L", "Makanan & Minuman", 19500, 440, 0.005, 0.20, 3),
        ("SKU-009", "Teh Celup Box 50s", "Makanan & Minuman", 13500, 290, 0.002, 0.14, 3),
        ("SKU-010", "Biskuit Kaleng Assorted", "Makanan & Minuman", 54000, 220, 0.003, 0.55, 5),
        ("SKU-011", "Sirup Cocopandan 460ml", "Makanan & Minuman", 23000, 210, 0.002, 0.70, 4),
        ("SKU-012", "Kecap Manis Botol 520ml", "Makanan & Minuman", 24500, 350, 0.003, 0.18, 3),
        ("SKU-013", "Detergen Cair 800ml", "Kebutuhan Rumah Tangga", 22000, 390, 0.004, 0.12, 4),
        ("SKU-014", "Sabun Cuci Piring 755ml", "Kebutuhan Rumah Tangga", 16500, 410, 0.004, 0.10, 3),
        ("SKU-015", "Pembersih Lantai 780ml", "Kebutuhan Rumah Tangga", 15000, 300, 0.003, 0.10, 4),
        ("SKU-016", "Tisu Wajah 250 Sheets", "Kebutuhan Rumah Tangga", 14500, 470, 0.005, 0.12, 3),
        ("SKU-017", "Popok Bayi M-34", "Kebutuhan Rumah Tangga", 68000, 280, 0.005, 0.10, 5),
        ("SKU-018", "Sabun Mandi Cair 450ml", "Kebutuhan Rumah Tangga", 29000, 340, 0.004, 0.12, 4),
        ("SKU-019", "Vitamin C 500mg Strip 10s", "Kesehatan & Farmasi", 18000, 260, 0.002, 0.14, 5),
        ("SKU-020", "Minyak Kayu Putih 120ml", "Kesehatan & Farmasi", 46000, 230, 0.003, 0.15, 5),
        ("SKU-021", "Masker Medis Box 50s", "Kesehatan & Farmasi", 27000, 190, -0.005, 0.08, 4),
        ("SKU-022", "Kertas HVS A4 80gsm Rim", "Perlengkapan B2B", 52000, 330, 0.003, 0.12, 4),
        ("SKU-023", "Kardus Packing 40x30x30", "Perlengkapan B2B", 11000, 520, 0.006, 0.18, 3),
        ("SKU-024", "Lakban Bening 48mm Roll", "Perlengkapan B2B", 12500, 490, 0.005, 0.15, 3),
    ]

    intermittent_specs = [
        ("SKU-025", "Minyak Zaitun Extra Virgin 1L", "Bahan Pokok", 165000, 45, 0.52, 0.25, 10),
        ("SKU-026", "Tepung Almond Impor 1kg", "Bahan Pokok", 195000, 35, 0.45, 0.30, 12),
        ("SKU-027", "Parcel Lebaran Eksklusif", "Makanan & Minuman", 320000, 110, 0.28, 0.65, 14),
        ("SKU-028", "Kopi Single Origin Arabika 1kg", "Makanan & Minuman", 210000, 55, 0.58, 0.20, 7),
        ("SKU-029", "Cokelat Couverture Block 2.5kg", "Makanan & Minuman", 285000, 48, 0.48, 0.35, 9),
        ("SKU-030", "Disinfektan Industri 5L", "Kebutuhan Rumah Tangga", 145000, 60, 0.42, 0.15, 7),
        ("SKU-031", "Karpet Anti Slip Gudang", "Perlengkapan B2B", 420000, 22, 0.32, 0.10, 14),
        ("SKU-032", "Toner Printer Laserjet Pro", "Perlengkapan B2B", 680000, 18, 0.50, 0.10, 8),
        ("SKU-033", "Termometer Inframerah Medis", "Kesehatan & Farmasi", 240000, 25, 0.35, 0.10, 10),
        ("SKU-034", "Kit P3K Lengkap Perusahaan", "Kesehatan & Farmasi", 175000, 30, 0.40, 0.10, 9),
        ("SKU-035", "Suplemen Herbal Imunitas Botol", "Kesehatan & Farmasi", 135000, 42, 0.55, 0.20, 7),
        ("SKU-036", "Sarung Tangan Nitrile Box 100s", "Kesehatan & Farmasi", 89000, 75, 0.46, 0.12, 6),
        ("SKU-037", "Pallet Plastik Heavy Duty", "Perlengkapan B2B", 390000, 28, 0.30, 0.15, 15),
        ("SKU-038", "Timbangan Digital Gudang 150kg", "Perlengkapan B2B", 850000, 12, 0.27, 0.10, 14),
        ("SKU-039", "Stretch Film Roll 50cm x 300m", "Perlengkapan B2B", 98000, 65, 0.60, 0.18, 6),
        ("SKU-040", "Saus Truffle Premium 500g", "Bahan Pokok", 225000, 32, 0.38, 0.30, 12),
    ]

    rows = []
    for sku_id, nama, kat, harga, base_qty, trend, seas_amp, lead_time in smooth_specs:
        rows.append({
            "SKU_ID": sku_id,
            "nama_produk": nama,
            "kategori": kat,
            "pola_target": "Lancar",
            "base_harga": float(harga),
            "base_qty": float(base_qty),
            "trend_bulanan": float(trend),
            "occ_prob": 0.98,
            "seas_amp": float(seas_amp),
            "lead_time_hari": int(lead_time),
        })
    for sku_id, nama, kat, harga, base_qty, occ_prob, seas_amp, lead_time in intermittent_specs:
        rows.append({
            "SKU_ID": sku_id,
            "nama_produk": nama,
            "kategori": kat,
            "pola_target": "Putus-putus",
            "base_harga": float(harga),
            "base_qty": float(base_qty),
            "trend_bulanan": 0.001,
            "occ_prob": float(occ_prob),
            "seas_amp": float(seas_amp),
            "lead_time_hari": int(lead_time),
        })
    return pd.DataFrame(rows)


def build_distributors(rng: np.random.Generator) -> pd.DataFrame:
    """Membuat 48 distributor B2B dengan segmen RFM & preferensi kategori."""
    regions = ["Jabodetabek", "Jawa Barat", "Jawa Timur", "Jawa Tengah", "Sumatera", "Kalimantan"]
    categories = [
        "Bahan Pokok",
        "Makanan & Minuman",
        "Kebutuhan Rumah Tangga",
        "Kesehatan & Farmasi",
        "Perlengkapan B2B",
    ]
    segments = [
        ("Champions", 10, 1.8, 0.92, (0, 60)),
        ("Loyal", 12, 1.3, 0.78, (0, 60)),
        ("Potential Loyalist", 10, 0.9, 0.65, (30, 60)),
        ("At Risk", 8, 1.1, 0.72, (0, 48)),
        ("Hibernating", 8, 0.5, 0.35, (0, 40)),
    ]

    rows = []
    idx = 1
    for seg_name, count, weight, act_prob, (active_start, active_end) in segments:
        for _ in range(count):
            pref_cats = rng.choice(categories, size=2, replace=False).tolist()
            rows.append({
                "distributor_ID": f"DIST-{idx:03d}",
                "nama_distributor": f"PT Mitra Niaga {idx:02d}",
                "wilayah": str(rng.choice(regions)),
                "segmen_target": seg_name,
                "volume_weight": weight * rng.uniform(0.8, 1.25),
                "activity_prob": act_prob,
                "active_start_idx": active_start,
                "active_end_idx": active_end,
                "pref_cat_1": pref_cats[0],
                "pref_cat_2": pref_cats[1],
            })
            idx += 1
    return pd.DataFrame(rows)


def generate_monthly_demand(skus: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Simulasi 60 bulan demand per SKU dengan sinyal promo, harga, dan musiman."""
    periods = pd.date_range(START_PERIOD, periods=N_MONTHS, freq="MS")
    records = []

    for _, sku in skus.iterrows():
        is_smooth = sku["pola_target"] == "Lancar"
        for t_idx, dt in enumerate(periods):
            tahun = dt.year
            bulan = dt.month
            is_lebaran = int((tahun, bulan) in LEBARAN_PEAK_MONTHS)
            is_nataru = int(bulan in (11, 12))
            hari_libur_bulan = int(1 + is_nataru * 2 + is_lebaran * 4 + (bulan == 8))

            # Inflasi harga bertahap ~3% per tahun + fluktuasi kecil
            inflasi = (1.03 ** (t_idx / 12.0)) * rng.uniform(0.97, 1.03)
            promo = int(rng.random() < (0.22 if is_smooth else 0.14))
            diskon_pct = int(rng.choice([10, 15, 20])) if promo else 0
            harga = round(sku["base_harga"] * inflasi * (1 - diskon_pct / 100.0), -2)

            # Efek musiman tahunan + lonjakan Lebaran/Nataru
            seas_wave = 1.0 + 0.12 * np.sin(2 * np.pi * (bulan - 1) / 12.0)
            event_boost = 1.0 + sku["seas_amp"] * (0.75 * is_lebaran + 0.45 * is_nataru)
            promo_boost = 1.0 + (0.28 if is_smooth else 0.35) * promo + (diskon_pct / 100.0)
            price_ratio = (sku["base_harga"] * (1.03 ** (t_idx / 12.0))) / max(harga, 1.0)
            trend_mult = 1.0 + sku["trend_bulanan"] * t_idx

            if is_smooth:
                occurred = rng.random() < sku["occ_prob"]
                if occurred:
                    mean_qty = sku["base_qty"] * trend_mult * seas_wave * event_boost * promo_boost * (price_ratio ** 0.4)
                    qty = int(max(1, round(rng.normal(mean_qty, mean_qty * 0.12))))
                else:
                    qty = 0
            else:
                # Probabilitas kejadian naik saat promo / musim puncak
                p_occ = min(0.92, sku["occ_prob"] * (1.0 + 0.30 * promo + 0.40 * is_lebaran))
                occurred = rng.random() < p_occ
                if occurred:
                    mean_qty = sku["base_qty"] * event_boost * promo_boost
                    # Gamma distribution menghasilkan ekor kanan realistis (lumpy/intermittent)
                    shape = 2.2 if sku["seas_amp"] >= 0.25 else 4.0
                    scale = mean_qty / shape
                    qty = int(max(1, round(rng.gamma(shape, scale))))
                else:
                    qty = 0

            lead_time = int(max(1, sku["lead_time_hari"] + rng.integers(-1, 3)))
            records.append({
                "periode": dt.strftime("%Y-%m-%d"),
                "tahun": tahun,
                "bulan": bulan,
                "SKU_ID": sku["SKU_ID"],
                "nama_produk": sku["nama_produk"],
                "kategori": sku["kategori"],
                "jumlah_terjual": qty,
                "harga": int(harga),
                "promo": promo,
                "diskon_pct": diskon_pct,
                "hari_libur_bulan": hari_libur_bulan,
                "is_musim_puncak": int(is_lebaran or is_nataru),
                "lead_time_hari": lead_time,
            })

    return pd.DataFrame(records)


def generate_distributor_transactions(
    demand_df: pd.DataFrame,
    distributors: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Memecah demand bulanan positif menjadi transaksi distributor yang koheren untuk RFM."""
    periods_sorted = sorted(demand_df["periode"].unique())
    period_to_idx = {p: i for i, p in enumerate(periods_sorted)}

    tx_rows = []
    tx_counter = 1

    nonzero_df = demand_df[demand_df["jumlah_terjual"] > 0]
    for _, row in nonzero_df.iterrows():
        t_idx = period_to_idx[row["periode"]]
        dt_start = pd.Timestamp(row["periode"])
        days_in_month = dt_start.days_in_month
        total_qty = int(row["jumlah_terjual"])
        kat = row["kategori"]

        # Filter distributor yang aktif di bulan t_idx
        active_mask = (distributors["active_start_idx"] <= t_idx) & (t_idx < distributors["active_end_idx"])
        eligible = distributors[active_mask].copy()
        if eligible.empty:
            eligible = distributors.copy()

        # Bobot pilihan berdasarkan preferensi kategori & segmen
        cat_match = np.where(
            (eligible["pref_cat_1"] == kat) | (eligible["pref_cat_2"] == kat),
            2.4,
            0.7,
        )
        probs = eligible["volume_weight"].to_numpy() * eligible["activity_prob"].to_numpy() * cat_match
        probs = probs / probs.sum()

        n_orders = int(min(total_qty, rng.integers(1, 7) if total_qty >= 20 else rng.integers(1, 3)))
        chosen_idx = rng.choice(eligible.index.to_numpy(), size=n_orders, replace=True, p=probs)

        # Bagi total_qty ke n_orders secara proporsional
        if n_orders == 1:
            qtys = [total_qty]
        else:
            splits = rng.dirichlet(np.ones(n_orders) * 2.0)
            qtys = np.maximum(1, np.floor(splits * total_qty).astype(int))
            diff = total_qty - int(qtys.sum())
            qtys[0] += diff
            if qtys[0] <= 0:
                qtys = [total_qty]
                chosen_idx = chosen_idx[:1]

        for dist_row_idx, q in zip(chosen_idx, qtys):
            if q <= 0:
                continue
            dist = distributors.loc[dist_row_idx]
            day = int(rng.integers(1, days_in_month + 1))
            tx_date = dt_start + pd.Timedelta(days=day - 1)
            harga_satuan = int(row["harga"])
            tx_rows.append({
                "transaction_ID": f"TRX-{tx_counter:06d}",
                "tanggal_transaksi": tx_date.strftime("%Y-%m-%d"),
                "distributor_ID": dist["distributor_ID"],
                "nama_distributor": dist["nama_distributor"],
                "wilayah": dist["wilayah"],
                "SKU_ID": row["SKU_ID"],
                "kategori": kat,
                "qty": int(q),
                "harga_satuan": harga_satuan,
                "nilai_transaksi": int(q) * harga_satuan,
            })
            tx_counter += 1

    tx_df = pd.DataFrame(tx_rows).sort_values(["tanggal_transaksi", "transaction_ID"]).reset_index(drop=True)
    return tx_df


def generate_inventory_snapshot(
    demand_df: pd.DataFrame,
    skus: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Membuat snapshot stok per 2026-10-31 dengan variasi normal dan stok berlebih."""
    recent = demand_df[demand_df["periode"] >= "2025-11-01"]
    stats = (
        recent.groupby("SKU_ID")["jumlah_terjual"]
        .agg(avg_monthly="mean", std_monthly="std")
        .fillna(0.0)
    )

    # Pilih 12 SKU sebagai kasus stok berlebih agar modul overstock & RFM bisa didemokan
    overstock_skus = set(rng.choice(skus["SKU_ID"].to_numpy(), size=12, replace=False))
    perishable_cats = {"Bahan Pokok", "Makanan & Minuman", "Kesehatan & Farmasi"}

    rows = []
    for _, sku in skus.iterrows():
        sku_id = sku["SKU_ID"]
        avg_m = max(1.0, float(stats.loc[sku_id, "avg_monthly"]))
        std_m = float(stats.loc[sku_id, "std_monthly"])
        lt_months = sku["lead_time_hari"] / 30.0

        safety_stock = int(max(5, round(1.65 * std_m * np.sqrt(max(lt_months, 0.15)))))
        reorder_point = int(max(safety_stock + 5, round(avg_m * lt_months + safety_stock)))

        is_overstock = sku_id in overstock_skus
        if is_overstock:
            stok_tersedia = int(round(avg_m * rng.uniform(4.0, 7.5) + safety_stock))
            umur_stok_hari = int(rng.integers(110, 260))
        else:
            stok_tersedia = int(max(0, round(avg_m * rng.uniform(0.6, 1.8) + safety_stock * 0.5)))
            umur_stok_hari = int(rng.integers(10, 75))

        if sku["kategori"] in perishable_cats:
            sisa_hari = int(rng.integers(35, 110)) if is_overstock else int(rng.integers(150, 420))
            tgl_kadaluwarsa = (SNAPSHOT_DATE + pd.Timedelta(days=sisa_hari)).strftime("%Y-%m-%d")
        else:
            tgl_kadaluwarsa = ""

        rows.append({
            "tanggal_snapshot": SNAPSHOT_DATE.strftime("%Y-%m-%d"),
            "SKU_ID": sku_id,
            "nama_produk": sku["nama_produk"],
            "kategori": sku["kategori"],
            "lokasi_gudang": str(rng.choice(WAREHOUSES)),
            "stok_tersedia": stok_tersedia,
            "safety_stock": safety_stock,
            "reorder_point": reorder_point,
            "umur_stok_hari": umur_stok_hari,
            "tanggal_kedaluwarsa": tgl_kadaluwarsa,
        })

    return pd.DataFrame(rows)


def summarize_sbc(demand_df: pd.DataFrame) -> pd.DataFrame:
    """Menghitung metrik SBC (Syntetos-Boylan-Croston): ADI dan CV^2 per SKU."""
    rows = []
    for sku_id, grp in demand_df.groupby("SKU_ID"):
        vals = grp["jumlah_terjual"].to_numpy()
        nz = vals[vals > 0]
        n_total = len(vals)
        n_nz = len(nz)
        adi = n_total / n_nz if n_nz > 0 else np.inf
        cv2 = (np.std(nz, ddof=0) / np.mean(nz)) ** 2 if n_nz > 1 and np.mean(nz) > 0 else 0.0

        if adi < 1.32 and cv2 < 0.49:
            sbc = "Smooth (Lancar)"
        elif adi < 1.32 and cv2 >= 0.49:
            sbc = "Erratic (Lancar-Fluktuatif)"
        elif adi >= 1.32 and cv2 < 0.49:
            sbc = "Intermittent (Putus-putus)"
        else:
            sbc = "Lumpy (Putus-putus Ekstrem)"

        rows.append({
            "SKU_ID": sku_id,
            "zero_months": int(n_total - n_nz),
            "ADI": round(float(adi), 3),
            "CV2": round(float(cv2), 3),
            "sbc_kategori": sbc,
        })
    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(SEED)
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    skus = build_sku_catalog()
    distributors = build_distributors(rng)
    demand_df = generate_monthly_demand(skus, rng)
    tx_df = generate_distributor_transactions(demand_df, distributors, rng)
    inv_df = generate_inventory_snapshot(demand_df, skus, rng)

    demand_path = out_dir / "demand_monthly.csv"
    inv_path = out_dir / "inventory_snapshot.csv"
    tx_path = out_dir / "distributor_transactions.csv"

    demand_df.to_csv(demand_path, index=False)
    inv_df.to_csv(inv_path, index=False)
    tx_df.to_csv(tx_path, index=False)

    sbc_df = summarize_sbc(demand_df)
    print(f"Tersimpan: {demand_path} ({len(demand_df)} baris, {demand_df['periode'].min()} s.d. {demand_df['periode'].max()})")
    print(f"Tersimpan: {inv_path} ({len(inv_df)} baris)")
    print(f"Tersimpan: {tx_path} ({len(tx_df)} baris)")
    print("\nDistribusi Pola SBC (Syntetos-Boylan-Croston):")
    print(sbc_df["sbc_kategori"].value_counts().to_string())


if __name__ == "__main__":
    main()
