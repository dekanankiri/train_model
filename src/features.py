"""Klasifikasi pola permintaan SBC dan rekayasa fitur deret waktu untuk StockWise AI."""

from __future__ import annotations

import numpy as np
import pandas as pd


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

# Bulan puncak Lebaran untuk horizon ramalan Nov 2026 - Jan 2027 (Lebaran 2027 jatuh pada Maret)
FORECAST_LEBARAN_MONTHS: set[tuple[int, int]] = set()


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
    out = df.sort_values(["SKU_ID", "periode"]).copy()
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
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Menyiapkan dataframe berfitur lengkap dan mengunci tipe kategori."""
    df = demand_df.copy()
    df["periode"] = pd.to_datetime(df["periode"])

    if sku_categories is None:
        sku_categories = sorted(df["SKU_ID"].unique().tolist())
    if prod_categories is None:
        prod_categories = sorted(df["kategori"].unique().tolist())

    df = _compute_lag_features_frame(df)
    df["SKU_ID_cat"] = pd.Categorical(df["SKU_ID"], categories=sku_categories)
    df["kategori_cat"] = pd.Categorical(df["kategori"], categories=prod_categories)

    # Gunakan baris setelah minimal lag_12 tersedia agar semua fitur lag terisi penuh
    df = df.dropna(subset=["lag_12"]).reset_index(drop=True)

    float_cols = [
        "sin_bulan", "cos_bulan", "harga", "diskon_pct", "hari_libur_bulan",
        "lead_time_hari", "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
        "rolling_mean_3", "rolling_mean_6", "rolling_std_3", "nonzero_rate_6",
    ]
    for col in float_cols:
        df[col] = df[col].astype("float32")
    df["bulan"] = df["bulan"].astype("int16")
    df["promo"] = df["promo"].astype("int8")
    df["is_musim_puncak"] = df["is_musim_puncak"].astype("int8")
    return df, sku_categories, prod_categories


def build_future_context_rows(
    history_df: pd.DataFrame,
    forecast_periods: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Membuat skeleton konteks kalender/harga/promo untuk horizon ramalan ke depan."""
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
            is_leb = int((tahun, bulan) in FORECAST_LEBARAN_MONTHS)
            rows.append({
                "periode": dt,
                "tahun": tahun,
                "bulan": bulan,
                "SKU_ID": sku["SKU_ID"],
                "nama_produk": sku["nama_produk"],
                "kategori": sku["kategori"],
                "jumlah_terjual": np.nan,
                "harga": float(sku["harga"]),
                "promo": 0,
                "diskon_pct": 0,
                "hari_libur_bulan": int(1 + is_nataru * 2 + is_leb * 4 + (bulan == 8)),
                "is_musim_puncak": int(is_nataru or is_leb),
                "lead_time_hari": int(sku["lead_time_hari"]),
            })
    return pd.DataFrame(rows)


def make_step_features(
    combined_history: pd.DataFrame,
    target_period: pd.Timestamp,
    sku_categories: list[str],
    prod_categories: list[str],
) -> pd.DataFrame:
    """Menghitung fitur 1 langkah ke depan untuk semua SKU pada target_period."""
    sub = combined_history[combined_history["periode"] <= target_period].copy()
    sub = _compute_lag_features_frame(sub)
    step_df = sub[sub["periode"] == target_period].copy()
    step_df["SKU_ID_cat"] = pd.Categorical(step_df["SKU_ID"], categories=sku_categories)
    step_df["kategori_cat"] = pd.Categorical(step_df["kategori"], categories=prod_categories)

    for col in FEATURE_COLS:
        if col not in CATEGORICAL_COLS:
            step_df[col] = step_df[col].astype("float32")
    return step_df.sort_values("SKU_ID").reset_index(drop=True)
