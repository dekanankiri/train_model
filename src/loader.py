"""Loader & validator data Excel/CSV untuk StockWise AI (mendukung data aktual & sintetis)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


COLUMN_ALIASES = {
    "date": "periode",
    "tanggal": "periode",
    "periode": "periode",
    "sku_id": "SKU_ID",
    "item code": "SKU_ID",
    "item_code": "SKU_ID",
    "nama_produk": "nama_produk",
    "kategori": "kategori",
    "line": "kategori",
    "jumlah_terjual": "jumlah_terjual",
    "sales": "jumlah_terjual",
    "stok_akhir": "stok_akhir",
    "stock": "stok_akhir",
    "bp_sales": "bp_sales",
    "bp": "bp_sales",
    "fct_existing": "fct_existing",
    "bo+oi": "fct_existing",
    "fct_prod": "fct_prod",
    "act_prod": "act_prod",
    "std_stok": "std_stok",
    "std": "std_stok",
    "min_stok_60pct": "min_stok_60pct",
    "60%xstd min": "min_stok_60pct",
    "turn_over_hari": "turn_over_hari",
    "tur_over_hari": "turn_over_hari",
    "turn over": "turn_over_hari",
    "harga": "harga",
    "promo": "promo",
    "diskon_pct": "diskon_pct",
    "hari_libur_bulan": "hari_libur_bulan",
    "is_musim_puncak": "is_musim_puncak",
    "lead_time_hari": "lead_time_hari",
}

NUMERIC_DEMAND_COLS = [
    "jumlah_terjual",
    "stok_akhir",
    "bp_sales",
    "fct_existing",
    "fct_prod",
    "act_prod",
    "std_stok",
    "min_stok_60pct",
    "turn_over_hari",
    "harga",
    "promo",
    "diskon_pct",
    "hari_libur_bulan",
    "is_musim_puncak",
    "lead_time_hari",
]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Menstandarkan nama kolom menggunakan alias (case-insensitive)."""
    rename_map = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[key]
        else:
            rename_map[col] = str(col).strip()
    return df.rename(columns=rename_map)


def _drop_template_metadata_rows(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Menghapus baris metadata tipe data (baris 2 template) atau baris kosong total."""
    out = df.dropna(how="all").copy()
    if out.empty:
        return out

    # Baris 2 template berisi teks seperti "DATE | WAJIB" atau "STRING | WAJIB"
    first_val = str(out.iloc[0].get(date_col, "")).upper()
    second_val = str(out.iloc[0].get("SKU_ID", "")).upper()
    if "WAJIB" in first_val or "OPSIONAL" in first_val or "WAJIB" in second_val:
        out = out.iloc[1:].copy()

    return out.reset_index(drop=True)


def _is_untouched_example_sheet(df: pd.DataFrame, marker_col: str, marker_vals: set[str]) -> bool:
    """Mendeteksi apakah sheet opsional hanya berisi baris contoh bawaan template."""
    if df.empty:
        return True
    vals = set(df[marker_col].dropna().astype(str).str.strip())
    return vals.issubset(marker_vals)


def clean_demand_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan, memvalidasi primary key (SKU_ID, periode), dan mengurutkan data demand."""
    df = _normalize_columns(raw_df)
    df = _drop_template_metadata_rows(df, "periode")

    required = {"periode", "SKU_ID", "jumlah_terjual"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Kolom wajib tidak ditemukan di data demand: {sorted(missing_cols)}")

    df["periode"] = pd.to_datetime(df["periode"], errors="coerce")
    df = df.dropna(subset=["periode", "SKU_ID"]).copy()
    # Normalkan ke tanggal 1 tiap bulan agar konsisten sebagai primary key bulanan
    df["periode"] = df["periode"].dt.to_period("M").dt.to_timestamp()

    df["SKU_ID"] = df["SKU_ID"].astype(str).str.strip()
    df = df[df["SKU_ID"] != ""].copy()

    if "nama_produk" not in df.columns:
        df["nama_produk"] = df["SKU_ID"]
    else:
        df["nama_produk"] = df["nama_produk"].fillna(df["SKU_ID"]).astype(str).str.strip()
        df.loc[df["nama_produk"] == "", "nama_produk"] = df.loc[df["nama_produk"] == "", "SKU_ID"]
        df = df[~df["nama_produk"].isin({"Wearing Device A (Contoh Lancar)", "Wearing Device B (Contoh Putus-putus)"})].copy()

    if "kategori" not in df.columns:
        df["kategori"] = "UNKNOWN"
    else:
        df["kategori"] = df["kategori"].fillna("UNKNOWN").astype(str).str.strip()
        df.loc[df["kategori"] == "", "kategori"] = "UNKNOWN"

    for col in NUMERIC_DEMAND_COLS:
        if col in df.columns:
            # Mengubah teks error Excel (#DIV/0!, #N/A, dll.) dan string kosong menjadi NaN
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if df["jumlah_terjual"].isna().any():
        n_bad = int(df["jumlah_terjual"].isna().sum())
        raise ValueError(f"Terdapat {n_bad} baris dengan jumlah_terjual kosong/non-numerik.")

    df["jumlah_terjual"] = df["jumlah_terjual"].clip(lower=0.0)

    # Validasi primary key (SKU_ID, periode)
    dup_mask = df.duplicated(subset=["SKU_ID", "periode"], keep=False)
    if dup_mask.any():
        n_dups = int(dup_mask.sum())
        sample = df.loc[dup_mask, ["SKU_ID", "periode"]].head(5).to_dict(orient="records")
        raise ValueError(
            f"Ditemukan {n_dups} baris duplikat pada kombinasi (SKU_ID, periode). "
            f"Contoh: {sample}"
        )

    df = df.sort_values(["SKU_ID", "periode"]).reset_index(drop=True)
    df["tahun"] = df["periode"].dt.year.astype(int)
    df["bulan"] = df["periode"].dt.month.astype(int)
    return df


def derive_inventory_snapshot_from_demand(demand_df: pd.DataFrame) -> pd.DataFrame:
    """Membentuk snapshot inventori terbaru dari baris bulan terakhir di data demand."""
    last_period = demand_df["periode"].max()
    snapshot_date = (last_period + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")

    latest = (
        demand_df.sort_values("periode")
        .groupby("SKU_ID", as_index=False)
        .last()
    )

    stok = latest["stok_akhir"].fillna(0.0) if "stok_akhir" in latest.columns else pd.Series(0.0, index=latest.index)
    if "min_stok_60pct" in latest.columns:
        safety = latest["min_stok_60pct"].copy()
    else:
        safety = pd.Series(np.nan, index=latest.index)

    if "std_stok" in latest.columns:
        std_val = latest["std_stok"]
        safety = safety.fillna(0.6 * std_val)
        reorder = std_val.fillna(safety)
    else:
        reorder = safety.fillna(0.0)

    safety = safety.fillna(0.0)
    umur = latest["turn_over_hari"].fillna(0.0) if "turn_over_hari" in latest.columns else pd.Series(0.0, index=latest.index)

    return pd.DataFrame({
        "tanggal_snapshot": snapshot_date,
        "SKU_ID": latest["SKU_ID"],
        "nama_produk": latest["nama_produk"],
        "kategori": latest["kategori"],
        "lokasi_gudang": "Gudang Utama",
        "stok_tersedia": stok.clip(lower=0.0).round(0).astype(int),
        "safety_stock": safety.clip(lower=0.0).round(0).astype(int),
        "reorder_point": reorder.clip(lower=0.0).round(0).astype(int),
        "umur_stok_hari": umur.clip(lower=0.0).round(0).astype(int),
        "tanggal_kedaluwarsa": "",
    })


def load_dataset(
    excel_path: Path | None = None,
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Memuat demand, inventory_snapshot, dan transaksi distributor (opsional) dari Excel atau CSV."""
    if excel_path is not None:
        xls = pd.ExcelFile(excel_path)
        sheet_names = xls.sheet_names

        demand_sheet = "1_Demand_Bulanan" if "1_Demand_Bulanan" in sheet_names else sheet_names[0]
        raw_demand = pd.read_excel(xls, sheet_name=demand_sheet)
        demand_df = clean_demand_frame(raw_demand)

        # Sheet 2: Inventory Snapshot (opsional; fallback otomatis dari bulan terakhir demand)
        inventory_df = None
        if "2_Inventory_Snapshot" in sheet_names:
            raw_inv = pd.read_excel(xls, sheet_name="2_Inventory_Snapshot")
            raw_inv = _normalize_columns(raw_inv)
            raw_inv = _drop_template_metadata_rows(raw_inv, "tanggal_snapshot")
            if not _is_untouched_example_sheet(
                raw_inv,
                "nama_produk",
                {"Wearing Device A (Contoh)", "Wearing Device B (Contoh)"},
            ):
                raw_inv["tanggal_snapshot"] = pd.to_datetime(raw_inv["tanggal_snapshot"], errors="coerce")
                raw_inv = raw_inv.dropna(subset=["tanggal_snapshot", "SKU_ID"]).copy()
                raw_inv["SKU_ID"] = raw_inv["SKU_ID"].astype(str).str.strip()
                raw_inv = raw_inv[raw_inv["SKU_ID"] != ""].copy()

                if "nama_produk" in raw_inv.columns:
                    raw_inv["nama_produk"] = raw_inv["nama_produk"].fillna(raw_inv["SKU_ID"]).astype(str).str.strip()
                    raw_inv.loc[raw_inv["nama_produk"] == "", "nama_produk"] = raw_inv.loc[raw_inv["nama_produk"] == "", "SKU_ID"]
                    raw_inv = raw_inv[
                        ~raw_inv["nama_produk"].isin({"Wearing Device A (Contoh)", "Wearing Device B (Contoh)"})
                    ].copy()
                else:
                    raw_inv["nama_produk"] = raw_inv["SKU_ID"]

                if not raw_inv.empty:
                    for col in ("stok_tersedia", "safety_stock", "reorder_point", "umur_stok_hari"):
                        if col in raw_inv.columns:
                            raw_inv[col] = pd.to_numeric(raw_inv[col], errors="coerce").fillna(0.0)
                        else:
                            raw_inv[col] = 0.0
                    raw_inv["kategori"] = (
                        raw_inv["kategori"].fillna("UNKNOWN").astype(str).str.strip()
                        if "kategori" in raw_inv.columns
                        else "UNKNOWN"
                    )
                    raw_inv["lokasi_gudang"] = (
                        raw_inv["lokasi_gudang"].fillna("Gudang Utama").astype(str).str.strip()
                        if "lokasi_gudang" in raw_inv.columns
                        else "Gudang Utama"
                    )
                    raw_inv["tanggal_kedaluwarsa"] = (
                        raw_inv["tanggal_kedaluwarsa"].fillna("")
                        if "tanggal_kedaluwarsa" in raw_inv.columns
                        else ""
                    )
                    raw_inv["tanggal_snapshot"] = raw_inv["tanggal_snapshot"].dt.strftime("%Y-%m-%d")
                    inventory_df = raw_inv

        if inventory_df is None or inventory_df.empty:
            inventory_df = derive_inventory_snapshot_from_demand(demand_df)

        # Sheet 3: Transaksi Distributor (opsional untuk modul RFM)
        tx_df = None
        if "3_Transaksi_Distributor" in sheet_names:
            raw_tx = pd.read_excel(xls, sheet_name="3_Transaksi_Distributor")
            raw_tx = _normalize_columns(raw_tx)
            raw_tx = _drop_template_metadata_rows(raw_tx, "tanggal_transaksi")
            if not _is_untouched_example_sheet(raw_tx, "nama_distributor", {"Distributor Contoh 1"}):
                raw_tx["tanggal_transaksi"] = pd.to_datetime(raw_tx["tanggal_transaksi"], errors="coerce")
                raw_tx = raw_tx.dropna(subset=["tanggal_transaksi", "distributor_ID", "SKU_ID"]).copy()
                if not raw_tx.empty:
                    for col in ("qty", "nilai_transaksi"):
                        raw_tx[col] = pd.to_numeric(raw_tx[col], errors="coerce").fillna(0.0)
                    raw_tx["nama_distributor"] = (
                        raw_tx["nama_distributor"].fillna(raw_tx["distributor_ID"])
                        if "nama_distributor" in raw_tx.columns
                        else raw_tx["distributor_ID"]
                    )
                    raw_tx["wilayah"] = (
                        raw_tx["wilayah"].fillna("UNKNOWN")
                        if "wilayah" in raw_tx.columns
                        else "UNKNOWN"
                    )
                    raw_tx["kategori"] = (
                        raw_tx["kategori"].fillna("UNKNOWN")
                        if "kategori" in raw_tx.columns
                        else "UNKNOWN"
                    )
                    tx_df = raw_tx

        return demand_df, inventory_df, tx_df

    # Mode default: membaca CSV dari data_dir
    assert data_dir is not None, "Harus menyertakan excel_path atau data_dir."
    demand_df = clean_demand_frame(pd.read_csv(data_dir / "demand_monthly.csv"))
    inventory_df = pd.read_csv(data_dir / "inventory_snapshot.csv")
    tx_path = data_dir / "distributor_transactions.csv"
    tx_df = pd.read_csv(tx_path) if tx_path.exists() else None
    return demand_df, inventory_df, tx_df
