"""Klasifikasi pola permintaan SBC dan rekayasa fitur deret waktu untuk StockWise AI."""

from __future__ import annotations

import numpy as np
import pandas as pd


CORE_FEATURE_COLS = [
    "SKU_ID_cat",
    "kategori_cat",
    "bulan",
    "sin_bulan",
    "cos_bulan",
    "hari_libur_bulan",
    "is_musim_puncak",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_std_3",
    "nonzero_rate_6",
]

OPTIONAL_EXTERNAL_COLS = ["harga", "promo", "diskon_pct", "lead_time_hari"]

FEATURE_COLS = [
    "SKU_ID_cat",
    "kategori_cat",
    "bulan",
    "sin_bulan",
    "cos_bulan",
    "harga",
    "promo",
    "diskon_pct",
    "hari_libur_bulan",
    "is_musim_puncak",
    "lead_time_hari",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_std_3",
    "nonzero_rate_6",
]

CATEGORICAL_COLS = ["SKU_ID_cat", "kategori_cat"]

# Bulan lonjakan musiman Indonesia (Ramadan/Lebaran & Nataru) untuk 2020-2027
LEBARAN_PEAK_MONTHS: set[tuple[int, int]] = {
    (2020, 5),
    (2021, 5),
    (2022, 4), (2022, 5),
    (2023, 3), (2023, 4),
    (2024, 3), (2024, 4),
    (2025, 3),
    (2026, 2), (2026, 3),
    (2027, 2), (2027, 3),
}


def select_feature_cols(demand_df: pd.DataFrame) -> list[str]:
    """Memilih fitur aktif berdasarkan ketersediaan kolom eksternal non-kosong di dataset."""
    active_ext = {
        col for col in OPTIONAL_EXTERNAL_COLS
        if col in demand_df.columns and demand_df[col].notna().any()
    }
    return [c for c in FEATURE_COLS if c in CORE_FEATURE_COLS or c in active_ext]


def enrich_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mengisi fitur kalender (tahun, bulan, hari_libur_bulan, is_musim_puncak) bila kosong."""
    out = df.copy()
    out["periode"] = pd.to_datetime(out["periode"])
    out["tahun"] = out["periode"].dt.year.astype(int)
    out["bulan"] = out["periode"].dt.month.astype(int)

    is_nataru = out["bulan"].isin([11, 12])
    is_leb = pd.Series(
        [(int(y), int(m)) in LEBARAN_PEAK_MONTHS for y, m in zip(out["tahun"], out["bulan"])],
        index=out.index,
    )
    auto_libur = 1 + is_nataru.astype(int) * 2 + is_leb.astype(int) * 4 + (out["bulan"] == 8).astype(int)
    auto_puncak = (is_nataru | is_leb).astype(int)

    if "hari_libur_bulan" not in out.columns:
        out["hari_libur_bulan"] = auto_libur
    else:
        out["hari_libur_bulan"] = pd.to_numeric(out["hari_libur_bulan"], errors="coerce").fillna(auto_libur)

    if "is_musim_puncak" not in out.columns:
        out["is_musim_puncak"] = auto_puncak
    else:
        out["is_musim_puncak"] = pd.to_numeric(out["is_musim_puncak"], errors="coerce").fillna(auto_puncak)

    return out


def classify_sbc(demand_df: pd.DataFrame) -> pd.DataFrame:
    """Mengklasifikasikan pola permintaan tiap SKU memakai aturan Syntetos-Boylan-Croston."""
    rows = []
    for sku_id, grp in demand_df.groupby("SKU_ID", sort=True):
        vals = grp.sort_values("periode")["jumlah_terjual"].to_numpy(dtype=float)
        nz = vals[vals > 0]
        n_total = len(vals)
        n_nz = len(nz)

        adi = n_total / n_nz if n_nz > 0 else float("inf")
        cv2 = (np.std(nz, ddof=0) / np.mean(nz)) ** 2 if n_nz > 1 and np.mean(nz) > 0 else 0.0

        if adi < 1.32 and cv2 < 0.49:
            sbc_detail = "Smooth"
        elif adi < 1.32:
            sbc_detail = "Erratic"
        elif cv2 < 0.49:
            sbc_detail = "Intermittent"
        else:
            sbc_detail = "Lumpy"

        pola_utama = "Lancar" if adi < 1.32 else "Putus-putus"
        rows.append({
            "SKU_ID": sku_id,
            "ADI": round(float(adi), 4),
            "CV2": round(float(cv2), 4),
            "sbc_detail": sbc_detail,
            "pola_utama": pola_utama,
        })
    return pd.DataFrame(rows)


def _compute_lag_features_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Menghitung fitur lag dan rolling per SKU tanpa kebocoran target masa depan."""
    out = enrich_calendar_features(df).sort_values(["SKU_ID", "periode"]).copy()
    grp = out.groupby("SKU_ID", observed=True)["jumlah_terjual"]

    for lag in (1, 2, 3, 6, 12):
        out[f"lag_{lag}"] = grp.shift(lag)

    shifted = grp.shift(1)
    out["rolling_mean_3"] = shifted.groupby(out["SKU_ID"], observed=True).transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    out["rolling_mean_6"] = shifted.groupby(out["SKU_ID"], observed=True).transform(
        lambda s: s.rolling(6, min_periods=1).mean()
    )
    out["rolling_std_3"] = shifted.groupby(out["SKU_ID"], observed=True).transform(
        lambda s: s.rolling(3, min_periods=1).std().fillna(0.0)
    )
    out["nonzero_rate_6"] = shifted.groupby(out["SKU_ID"], observed=True).transform(
        lambda s: s.gt(0).rolling(6, min_periods=1).mean()
    )

    out["sin_bulan"] = np.sin(2.0 * np.pi * (out["bulan"] - 1) / 12.0)
    out["cos_bulan"] = np.cos(2.0 * np.pi * (out["bulan"] - 1) / 12.0)
    return out


def prepare_training_frame(
    demand_df: pd.DataFrame,
    sku_categories: list[str] | None = None,
    prod_categories: list[str] | None = None,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Menyiapkan dataframe berfitur lengkap dan mengunci tipe kategori."""
    active_cols = feature_cols or select_feature_cols(demand_df)
    df = demand_df.copy()
    df["periode"] = pd.to_datetime(df["periode"])

    if sku_categories is None:
        sku_categories = sorted(df["SKU_ID"].unique().tolist())
    if prod_categories is None:
        prod_categories = sorted(df["kategori"].unique().tolist())

    df = _compute_lag_features_frame(df)
    df["SKU_ID_cat"] = pd.Categorical(df["SKU_ID"], categories=sku_categories)
    df["kategori_cat"] = pd.Categorical(df["kategori"], categories=prod_categories)

    # Gunakan lag_12 penuh jika riwayat SKU cukup panjang; untuk SKU baru (<=18 bln) cukup minimal lag_1
    sku_counts = df.groupby("SKU_ID", observed=True)["periode"].transform("count")
    keep_mask = df["lag_12"].notna() | ((sku_counts <= 18) & df["lag_1"].notna())
    df = df[keep_mask].reset_index(drop=True)

    for col in active_cols:
        if col in CATEGORICAL_COLS:
            continue
        if col == "bulan":
            df[col] = df[col].astype("int16")
        elif col in ("promo", "is_musim_puncak"):
            df[col] = df[col].fillna(0).astype("int8")
        else:
            df[col] = df[col].astype("float32")
    return df, sku_categories, prod_categories


def build_future_context_rows(
    history_df: pd.DataFrame,
    forecast_periods: pd.DatetimeIndex,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Membuat skeleton konteks kalender (dan fitur eksternal bila aktif) untuk horizon ramalan ke depan."""
    active_cols = set(feature_cols or select_feature_cols(history_df))
    latest = (
        history_df.sort_values("periode")
        .groupby("SKU_ID", as_index=False)
        .last()
    )
    rows = []
    for _, sku in latest.iterrows():
        for dt in forecast_periods:
            bulan = int(dt.month)
            tahun = int(dt.year)
            is_nataru = int(bulan in (11, 12))
            is_leb = int((tahun, bulan) in LEBARAN_PEAK_MONTHS)
            row = {
                "periode": dt,
                "tahun": tahun,
                "bulan": bulan,
                "SKU_ID": sku["SKU_ID"],
                "nama_produk": sku["nama_produk"],
                "kategori": sku["kategori"],
                "jumlah_terjual": np.nan,
                "hari_libur_bulan": int(1 + is_nataru * 2 + is_leb * 4 + (bulan == 8)),
                "is_musim_puncak": int(is_nataru or is_leb),
            }
            if "harga" in active_cols and "harga" in latest.columns:
                row["harga"] = float(sku["harga"]) if pd.notna(sku["harga"]) else np.nan
            if "promo" in active_cols:
                row["promo"] = 0
            if "diskon_pct" in active_cols:
                row["diskon_pct"] = 0.0
            if "lead_time_hari" in active_cols and "lead_time_hari" in latest.columns:
                row["lead_time_hari"] = float(sku["lead_time_hari"]) if pd.notna(sku["lead_time_hari"]) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def make_step_features(
    combined_history: pd.DataFrame,
    target_period: pd.Timestamp,
    sku_categories: list[str],
    prod_categories: list[str],
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Menghitung fitur 1 langkah ke depan untuk semua SKU pada target_period."""
    active_cols = feature_cols or select_feature_cols(combined_history)
    sub = combined_history[combined_history["periode"] <= target_period].copy()
    sub = _compute_lag_features_frame(sub)
    step_df = sub[sub["periode"] == target_period].copy()
    step_df["SKU_ID_cat"] = pd.Categorical(step_df["SKU_ID"], categories=sku_categories)
    step_df["kategori_cat"] = pd.Categorical(step_df["kategori"], categories=prod_categories)

    for col in active_cols:
        if col not in CATEGORICAL_COLS:
            step_df[col] = step_df[col].astype("float32")
    return step_df.sort_values("SKU_ID").reset_index(drop=True)
