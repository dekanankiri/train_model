"""Pipeline utama training, backtest, SHAP, dan RFM untuk prototipe StockWise AI."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import (
    FEATURE_COLS,
    build_future_context_rows,
    classify_sbc,
    prepare_training_frame,
)
from src.models import (
    QuantileLGBMForecaster,
    TwoStageHurdleForecaster,
    compute_backtest_metrics,
    compute_shap_drivers,
    forecast_arima_all_skus,
    forecast_moving_average_all_skus,
    recursive_ml_forecast,
)
from src.rfm import build_overstock_recommendations, compute_rfm_distributors


BACKTEST_MONTHS = 6
FORECAST_MONTHS = 3
SNAPSHOT_DATE = "2026-10-31"


def run_backtest(
    demand_df: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    """Evaluasi out-of-sample 6 bulan terakhir (Mei 2026 - Okt 2026)."""
    periods = sorted(demand_df["periode"].unique())
    split_idx = len(periods) - BACKTEST_MONTHS
    train_periods = set(periods[:split_idx])
    test_periods = pd.DatetimeIndex(periods[split_idx:])

    train_raw = demand_df[demand_df["periode"].isin(train_periods)].copy()
    test_raw = demand_df[demand_df["periode"].isin(test_periods)].copy()

    # Klasifikasi SBC dihitung hanya dari data latih agar bebas data leakage
    sbc_train = classify_sbc(train_raw)
    train_feat, sku_cats, prod_cats = prepare_training_frame(train_raw)

    X_train = train_feat[FEATURE_COLS]
    y_train = train_feat["jumlah_terjual"]

    q_model = QuantileLGBMForecaster().fit(X_train, y_train)
    h_model = TwoStageHurdleForecaster().fit(X_train, y_train)

    # Kosongkan target pada konteks uji sebelum prediksi rekursif
    test_ctx = test_raw.copy()
    test_ctx["jumlah_terjual"] = np.nan

    ml_preds, _ = recursive_ml_forecast(
        history_df=train_raw,
        future_context_df=test_ctx,
        quantile_model=q_model,
        hurdle_model=h_model,
        sbc_df=sbc_train,
        sku_categories=sku_cats,
        prod_categories=prod_cats,
    )
    arima_preds = forecast_arima_all_skus(train_raw, test_periods)
    ma_preds = forecast_moving_average_all_skus(train_raw, test_periods, window=3)

    actuals = test_raw[["periode", "SKU_ID", "jumlah_terjual"]].copy()
    eval_df = (
        actuals
        .merge(ml_preds, on=["periode", "SKU_ID"], how="left")
        .merge(arima_preds, on=["periode", "SKU_ID"], how="left")
        .merge(ma_preds, on=["periode", "SKU_ID"], how="left")
    )

    summary_df = compute_backtest_metrics(eval_df)
    summary_df.to_csv(out_dir / "backtest_summary.csv", index=False)

    # Ringkasan error per SKU untuk inspeksi detail
    sku_rows = []
    for (sku_id, pola), grp in eval_df.groupby(["SKU_ID", "pola_sbc"]):
        y = grp["jumlah_terjual"].to_numpy(dtype=float)
        sku_rows.append({
            "SKU_ID": sku_id,
            "pola_sbc": pola,
            "aktual_total_6b": round(float(y.sum()), 1),
            "MAE_lgbm_q50": round(float(np.mean(np.abs(y - grp["lgbm_q50"]))), 2),
            "MAE_hurdle": round(float(np.mean(np.abs(y - grp["hurdle_pred"]))), 2),
            "MAE_arima": round(float(np.mean(np.abs(y - grp["arima_pred"]))), 2),
            "MAE_ma3": round(float(np.mean(np.abs(y - grp["ma_pred"]))), 2),
        })
    pd.DataFrame(sku_rows).to_csv(out_dir / "backtest_sku_metrics.csv", index=False)
    eval_df.to_csv(out_dir / "backtest_predictions_detail.csv", index=False)
    return summary_df


def run_full_training_and_forecast(
    demand_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    tx_df: pd.DataFrame,
    artifacts_dir: Path,
    out_dir: Path,
) -> dict[str, object]:
    """Refit pada seluruh 60 bulan, ramal Nov 2026 - Jan 2027, hitung SHAP & RFM."""
    sbc_full = classify_sbc(demand_df)
    sbc_full.to_csv(out_dir / "sbc_classification.csv", index=False)

    full_feat, sku_cats, prod_cats = prepare_training_frame(demand_df)
    X_full = full_feat[FEATURE_COLS]
    y_full = full_feat["jumlah_terjual"]

    q_model = QuantileLGBMForecaster().fit(X_full, y_full)
    h_model = TwoStageHurdleForecaster().fit(X_full, y_full)

    last_period = demand_df["periode"].max()
    forecast_periods = pd.date_range(last_period + pd.offsets.MonthBegin(1), periods=FORECAST_MONTHS, freq="MS")
    future_ctx = build_future_context_rows(demand_df, forecast_periods)

    ml_fc, step_features = recursive_ml_forecast(
        history_df=demand_df,
        future_context_df=future_ctx,
        quantile_model=q_model,
        hurdle_model=h_model,
        sbc_df=sbc_full,
        sku_categories=sku_cats,
        prod_categories=prod_cats,
    )
    arima_fc = forecast_arima_all_skus(demand_df, forecast_periods)
    ma_fc = forecast_moving_average_all_skus(demand_df, forecast_periods, window=3)

    fc_df = (
        ml_fc
        .merge(arima_fc, on=["periode", "SKU_ID"], how="left")
        .merge(ma_fc, on=["periode", "SKU_ID"], how="left")
        .merge(sbc_full[["SKU_ID", "ADI", "CV2", "sbc_detail"]], on="SKU_ID", how="left")
    )

    is_lancar = fc_df["pola_sbc"] == "Lancar"
    fc_df["model_terpilih"] = np.where(is_lancar, "LightGBM Kuantil (q50)", "Hurdle Dua Tahap")
    fc_df["ramalan_terpilih"] = np.where(is_lancar, fc_df["lgbm_q50"], fc_df["hurdle_pred"]).round(2)

    for col in ("lgbm_q10", "lgbm_q50", "lgbm_q90", "hurdle_cond_mag", "hurdle_pred", "arima_pred", "ma_pred"):
        fc_df[col] = fc_df[col].round(2)
    fc_df["hurdle_prob_occ"] = fc_df["hurdle_prob_occ"].round(4)
    fc_df["periode"] = pd.to_datetime(fc_df["periode"]).dt.strftime("%Y-%m-%d")
    fc_df.to_csv(out_dir / "forecast_nov2026_jan2027.csv", index=False)

    # Explainable AI (SHAP TreeExplainer)
    shap_df = compute_shap_drivers(q_model, h_model, step_features, sbc_full, top_k=5)
    shap_df.to_csv(out_dir / "shap_drivers_nov2026_jan2027.csv", index=False)

    # RFM & Rekomendasi Penyaluran Stok Berlebih
    rfm_df = compute_rfm_distributors(tx_df, snapshot_date=SNAPSHOT_DATE)
    rfm_df.to_csv(out_dir / "rfm_distributors.csv", index=False)

    rec_df = build_overstock_recommendations(inventory_df, fc_df, tx_df, rfm_df, top_n_distributors=3)
    rec_df.to_csv(out_dir / "overstock_recommendations.csv", index=False)

    # Simpan artefak model terlatih
    joblib.dump(q_model, artifacts_dir / "quantile_lgbm.joblib", compress=3)
    joblib.dump(h_model, artifacts_dir / "hurdle_two_stage.joblib", compress=3)

    meta = {
        "train_period_start": str(demand_df["periode"].min().strftime("%Y-%m-%d")),
        "train_period_end": str(demand_df["periode"].max().strftime("%Y-%m-%d")),
        "forecast_periods": [d.strftime("%Y-%m-%d") for d in forecast_periods],
        "n_skus": int(demand_df["SKU_ID"].nunique()),
        "feature_cols": FEATURE_COLS,
        "sku_categories": sku_cats,
        "prod_categories": prod_cats,
        "hurdle_smearing_factor": round(h_model.smearing_factor, 4),
    }
    (artifacts_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "fc_df": fc_df,
        "shap_df": shap_df,
        "rfm_df": rfm_df,
        "rec_df": rec_df,
    }


def main() -> None:
    t0 = time.perf_counter()
    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    artifacts_dir = root / "artifacts"
    out_dir = root / "outputs"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    demand_df = pd.read_csv(data_dir / "demand_monthly.csv")
    demand_df["periode"] = pd.to_datetime(demand_df["periode"])
    inventory_df = pd.read_csv(data_dir / "inventory_snapshot.csv")
    tx_df = pd.read_csv(data_dir / "distributor_transactions.csv")

    print("1/3 Menjalankan Backtest (Mei 2026 - Okt 2026)...")
    backtest_summary = run_backtest(demand_df, out_dir)

    print("2/3 Training Penuh (60 bulan), Ramalan Nov 2026 - Jan 2027, SHAP & RFM...")
    res = run_full_training_and_forecast(demand_df, inventory_df, tx_df, artifacts_dir, out_dir)

    elapsed = time.perf_counter() - t0
    print(f"3/3 Selesai dalam {elapsed:.2f} detik.\n")
    print("=== Ringkasan Backtest (6 Bulan Terakhir) ===")
    print(backtest_summary.to_string(index=False))

    print("\n=== Ringkasan Segmen RFM Distributor ===")
    print(res["rfm_df"]["segmen_rfm"].value_counts().to_string())

    n_overstock_skus = res["rec_df"]["SKU_ID"].nunique() if not res["rec_df"].empty else 0
    print(f"\nSKU dengan stok berlebih terdeteksi: {n_overstock_skus} SKU")
    print(f"Artefak model tersimpan di: {artifacts_dir}")
    print(f"Laporan output tersimpan di: {out_dir}")


if __name__ == "__main__":
    main()
