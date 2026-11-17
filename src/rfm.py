"""Segmentasi RFM distributor dan rekomendasi penyaluran stok berlebih."""

from __future__ import annotations

import numpy as np
import pandas as pd


RFM_WEIGHTS = {"R": 0.30, "F": 0.35, "M": 0.35}


def compute_rfm_distributors(
    tx_df: pd.DataFrame,
    snapshot_date: str = "2026-10-31",
) -> pd.DataFrame:
    """Menghitung metrik, skor, segmen, dan pengaruh R/F/M pada peringkat distributor."""
    tx = tx_df.copy()
    tx["tanggal_transaksi"] = pd.to_datetime(tx["tanggal_transaksi"])
    ref_date = pd.Timestamp(snapshot_date)

    rfm = (
        tx.groupby(["distributor_ID", "nama_distributor", "wilayah"], as_index=False)
        .agg(
            last_purchase=("tanggal_transaksi", "max"),
            frequency_order=("transaction_ID", "nunique"),
            monetary_total=("nilai_transaksi", "sum"),
            total_qty=("qty", "sum"),
        )
    )
    rfm["recency_hari"] = (ref_date - rfm["last_purchase"]).dt.days.clip(lower=0)

    # Skor 1-5 berbasis rank agar aman dari tie pada qcut
    rfm["R_score"] = pd.qcut(
        (-rfm["recency_hari"]).rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)
    rfm["F_score"] = pd.qcut(
        rfm["frequency_order"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)
    rfm["M_score"] = pd.qcut(
        rfm["monetary_total"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

    w_r = rfm["R_score"] * RFM_WEIGHTS["R"]
    w_f = rfm["F_score"] * RFM_WEIGHTS["F"]
    w_m = rfm["M_score"] * RFM_WEIGHTS["M"]
    total_score = w_r + w_f + w_m

    rfm["rfm_score"] = total_score.round(3)
    rfm["pengaruh_R_pct"] = (100.0 * w_r / total_score).round(1)
    rfm["pengaruh_F_pct"] = (100.0 * w_f / total_score).round(1)
    rfm["pengaruh_M_pct"] = (100.0 * w_m / total_score).round(1)

    def _assign_segment(row: pd.Series) -> str:
        r, f, m = row["R_score"], row["F_score"], row["M_score"]
        fm_avg = 0.5 * (f + m)
        if r >= 4 and fm_avg >= 4.0:
            return "Champions"
        if r >= 3 and fm_avg >= 3.0:
            return "Loyal"
        if r >= 4 and fm_avg < 3.0:
            return "Potential Loyalist"
        if r <= 2 and fm_avg >= 3.0:
            return "At Risk"
        return "Hibernating"

    rfm["segmen_rfm"] = rfm.apply(_assign_segment, axis=1)
    rfm = rfm.sort_values(["rfm_score", "monetary_total"], ascending=[False, False]).reset_index(drop=True)
    rfm["peringkat_rfm"] = np.arange(1, len(rfm) + 1)
    rfm["last_purchase"] = rfm["last_purchase"].dt.strftime("%Y-%m-%d")
    return rfm


def build_overstock_recommendations(
    inventory_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    tx_df: pd.DataFrame,
    rfm_df: pd.DataFrame,
    top_n_distributors: int = 3,
) -> pd.DataFrame:
    """Mendeteksi SKU stok berlebih dan merekomendasikan distributor teratas berdasar RFM + afinitas."""
    fc_3m = (
        forecast_df.groupby("SKU_ID", as_index=False)["ramalan_terpilih"]
        .sum()
        .rename(columns={"ramalan_terpilih": "ramalan_3_bulan"})
    )

    inv = inventory_df.merge(fc_3m, on="SKU_ID", how="left")
    inv["ramalan_3_bulan"] = inv["ramalan_3_bulan"].fillna(0.0)
    inv["kebutuhan_aman_3m"] = inv["ramalan_3_bulan"] + inv["safety_stock"]
    inv["kelebihan_stok"] = np.maximum(0.0, inv["stok_tersedia"] - inv["kebutuhan_aman_3m"]).round(0).astype(int)

    overstock = inv[inv["kelebihan_stok"] > 0].copy()
    if overstock.empty:
        return pd.DataFrame()

    # Afinitas historis distributor terhadap SKU & kategori
    sku_aff = (
        tx_df.groupby(["distributor_ID", "SKU_ID"], as_index=False)["qty"]
        .sum()
        .rename(columns={"qty": "hist_qty_sku"})
    )
    cat_aff = (
        tx_df.groupby(["distributor_ID", "kategori"], as_index=False)["qty"]
        .sum()
        .rename(columns={"qty": "hist_qty_kategori"})
    )

    rec_rows = []
    # Prioritaskan SKU dengan kelebihan stok besar & umur stok tinggi
    overstock = overstock.sort_values(["umur_stok_hari", "kelebihan_stok"], ascending=[False, False])

    for _, sku in overstock.iterrows():
        sku_id = sku["SKU_ID"]
        kat = sku["kategori"]

        cand = rfm_df.merge(
            sku_aff[sku_aff["SKU_ID"] == sku_id][["distributor_ID", "hist_qty_sku"]],
            on="distributor_ID",
            how="left",
        ).merge(
            cat_aff[cat_aff["kategori"] == kat][["distributor_ID", "hist_qty_kategori"]],
            on="distributor_ID",
            how="left",
        )
        cand["hist_qty_sku"] = cand["hist_qty_sku"].fillna(0.0)
        cand["hist_qty_kategori"] = cand["hist_qty_kategori"].fillna(0.0)

        # Skor kecocokan: 65% potensi RFM + 35% relevansi pembelian SKU/kategori
        max_sku = max(float(cand["hist_qty_sku"].max()), 1.0)
        max_cat = max(float(cand["hist_qty_kategori"].max()), 1.0)
        aff_norm = 0.6 * (cand["hist_qty_sku"] / max_sku) + 0.4 * (cand["hist_qty_kategori"] / max_cat)
        rfm_norm = cand["rfm_score"] / 5.0
        cand["skor_kecocokan"] = (0.65 * rfm_norm + 0.35 * aff_norm) * 100.0

        top_cands = cand.sort_values("skor_kecocokan", ascending=False).head(top_n_distributors)
        # Alokasikan porsi penawaran proporsional terhadap skor kecocokan
        weights = top_cands["skor_kecocokan"].to_numpy()
        weights = weights / weights.sum()
        n_top = len(top_cands)
        if sku["kelebihan_stok"] < n_top:
            alokasi = np.zeros(n_top, dtype=int)
            alokasi[: int(sku["kelebihan_stok"])] = 1
        else:
            alokasi = np.maximum(1, np.floor(weights * sku["kelebihan_stok"]).astype(int))
            alokasi[0] += int(sku["kelebihan_stok"] - alokasi.sum())

        for rank_idx, ((_, d), alok) in enumerate(zip(top_cands.iterrows(), alokasi), start=1):
            if alok <= 0:
                continue
            rec_rows.append({
                "SKU_ID": sku_id,
                "nama_produk": sku["nama_produk"],
                "kategori": kat,
                "lokasi_gudang": sku["lokasi_gudang"],
                "stok_tersedia": int(sku["stok_tersedia"]),
                "ramalan_3_bulan": round(float(sku["ramalan_3_bulan"]), 1),
                "kelebihan_stok": int(sku["kelebihan_stok"]),
                "umur_stok_hari": int(sku["umur_stok_hari"]),
                "tanggal_kedaluwarsa": sku["tanggal_kedaluwarsa"] if pd.notna(sku["tanggal_kedaluwarsa"]) else "",
                "peringkat_tawaran": rank_idx,
                "distributor_ID": d["distributor_ID"],
                "nama_distributor": d["nama_distributor"],
                "wilayah": d["wilayah"],
                "segmen_rfm": d["segmen_rfm"],
                "rfm_score": d["rfm_score"],
                "pengaruh_R_pct": d["pengaruh_R_pct"],
                "pengaruh_F_pct": d["pengaruh_F_pct"],
                "pengaruh_M_pct": d["pengaruh_M_pct"],
                "hist_beli_sku_qty": int(d["hist_qty_sku"]),
                "skor_kecocokan": round(float(d["skor_kecocokan"]), 2),
                "saran_alokasi_qty": int(alok),
            })

    return pd.DataFrame(rec_rows)
