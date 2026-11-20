"""Pipeline utama training, backtest, SHAP, dan RFM untuk prototipe StockWise AI."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import (
    build_future_context_rows,
    classify_sbc,
    prepare_training_frame,
    select_feature_cols,
)
from src.loader import load_dataset
from src.models import (
    QuantileLGBMForecaster,
    TwoStageHurdleForecaster,
    compute_backtest_metrics,
    compute_shap_drivers,
    forecast_arima_all_skus,
    forecast_moving_average_all_skus,
    recursive_ml_forecast,
    rolling_1step_ml_forecast,
)
from src.rfm import (
    build_overstock_recommendations,
    build_overstock_summary,
    compute_rfm_distributors,
)


MONTH_ID = {
    1: "jan", 2: "feb", 3: "mar", 4: "apr", 5: "mei", 6: "jun",
    7: "jul", 8: "agu", 9: "sep", 10: "okt", 11: "nov", 12: "des",
}


def _horizon_slug(periods: pd.DatetimeIndex) -> str:
    """Membentuk akhiran nama file seperti 'nov2026_jan2027'."""
    p0, p1 = periods[0], periods[-1]
    return f"{MONTH_ID[p0.month]}{p0.year}_{MONTH_ID[p1.month]}{p1.year}"


def run_backtest(
    demand_df: pd.DataFrame,
    out_dir: Path,
    backtest_months: int = 6,
    skip_arima: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluasi out-of-sample multi-step rekursif dan rolling 1-bulan ke depan."""
    periods = sorted(demand_df["periode"].unique())
    if len(periods) <= backtest_months + 2:
        raise ValueError(
            f"Jumlah bulan ({len(periods)}) terlalu sedikit untuk backtest {backtest_months} bulan."
        )

    split_idx = len(periods) - backtest_months
    train_periods = set(periods[:split_idx])
    test_periods = pd.DatetimeIndex(periods[split_idx:])

    train_raw = demand_df[demand_df["periode"].isin(train_periods)].copy()
    test_raw = demand_df[demand_df["periode"].isin(test_periods)].copy()

    feature_cols = select_feature_cols(train_raw)

    # Klasifikasi SBC dihitung hanya dari data latih agar bebas data leakage
    sbc_train = classify_sbc(train_raw)
    train_feat, sku_cats, prod_cats = prepare_training_frame(train_raw, feature_cols=feature_cols)

    X_train = train_feat[feature_cols]
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
        feature_cols=feature_cols,
    )
    ma_preds = forecast_moving_average_all_skus(train_raw, test_periods, window=3)

    actual_cols = ["periode", "SKU_ID", "jumlah_terjual"]
    for bench_col in ("fct_existing", "bp_sales"):
        if bench_col in test_raw.columns and test_raw[bench_col].notna().any():
            actual_cols.append(bench_col)

    eval_df = (
        test_raw[actual_cols].copy()
        .merge(ml_preds, on=["periode", "SKU_ID"], how="left")
        .merge(ma_preds, on=["periode", "SKU_ID"], how="left")
    )
    if not skip_arima:
        arima_preds = forecast_arima_all_skus(train_raw, test_periods)
        eval_df = eval_df.merge(arima_preds, on=["periode", "SKU_ID"], how="left")

    summary_df = compute_backtest_metrics(eval_df)
    summary_df.to_csv(out_dir / "backtest_summary.csv", index=False)

    # Evaluasi Rolling 1-Step-Ahead (adil untuk benchmark vs BO+OI yang terbit tiap bulan)
    rolling_preds = rolling_1step_ml_forecast(
        full_df=demand_df,
        test_periods=test_periods,
        quantile_model=q_model,
        hurdle_model=h_model,
        sbc_df=sbc_train,
        sku_categories=sku_cats,
        prod_categories=prod_cats,
        feature_cols=feature_cols,
    )
    rolling_eval_df = test_raw[actual_cols].copy().merge(
        rolling_preds, on=["periode", "SKU_ID"], how="left"
    )
    rolling_summary_df = compute_backtest_metrics(rolling_eval_df)
    rolling_summary_df.to_csv(out_dir / "backtest_rolling_1step_summary.csv", index=False)
    rolling_eval_df.to_csv(out_dir / "backtest_rolling_1step_detail.csv", index=False)

    train_skus = set(train_raw["SKU_ID"].unique())

    # Ringkasan error per SKU untuk inspeksi detail
    sku_rows = []
    for (sku_id, pola), grp in eval_df.groupby(["SKU_ID", "pola_sbc"]):
        y = grp["jumlah_terjual"].to_numpy(dtype=float)
        row = {
            "SKU_ID": sku_id,
            "pola_sbc": pola,
            "ada_di_train": sku_id in train_skus,
            "aktual_total": round(float(y.sum()), 1),
            "MAE_lgbm_q50": round(float(np.mean(np.abs(y - grp["lgbm_q50"]))), 2),
            "MAE_hurdle": round(float(np.mean(np.abs(y - grp["hurdle_pred"]))), 2),
            "MAE_ma3": round(float(np.mean(np.abs(y - grp["ma_pred"]))), 2),
        }
        if "arima_pred" in grp.columns:
            row["MAE_arima"] = round(float(np.mean(np.abs(y - grp["arima_pred"]))), 2)
        if "fct_existing" in grp.columns and grp["fct_existing"].notna().any():
            valid = grp["fct_existing"].notna()
            row["MAE_fct_existing"] = round(
                float(np.mean(np.abs(y[valid] - grp.loc[valid, "fct_existing"]))), 2
            )
        sku_rows.append(row)

    pd.DataFrame(sku_rows).to_csv(out_dir / "backtest_sku_metrics.csv", index=False)
    eval_df.to_csv(out_dir / "backtest_predictions_detail.csv", index=False)
    return summary_df, rolling_summary_df


def run_full_training_and_forecast(
    demand_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    tx_df: pd.DataFrame | None,
    artifacts_dir: Path,
    out_dir: Path,
    forecast_months: int = 3,
    skip_arima: bool = False,
) -> dict[str, object]:
    """Refit pada seluruh histori, ramal N bulan ke depan, hitung SHAP, overstock & RFM."""
    feature_cols = select_feature_cols(demand_df)

    sbc_full = classify_sbc(demand_df)
    sbc_full.to_csv(out_dir / "sbc_classification.csv", index=False)

    full_feat, sku_cats, prod_cats = prepare_training_frame(demand_df, feature_cols=feature_cols)
    X_full = full_feat[feature_cols]
    y_full = full_feat["jumlah_terjual"]

    q_model = QuantileLGBMForecaster().fit(X_full, y_full)
    h_model = TwoStageHurdleForecaster().fit(X_full, y_full)

    last_period = demand_df["periode"].max()
    forecast_periods = pd.date_range(
        last_period + pd.offsets.MonthBegin(1),
        periods=forecast_months,
        freq="MS",
    )
    slug = _horizon_slug(forecast_periods)

    future_ctx = build_future_context_rows(demand_df, forecast_periods, feature_cols=feature_cols)

    ml_fc, step_features = recursive_ml_forecast(
        history_df=demand_df,
        future_context_df=future_ctx,
        quantile_model=q_model,
        hurdle_model=h_model,
        sbc_df=sbc_full,
        sku_categories=sku_cats,
        prod_categories=prod_cats,
        feature_cols=feature_cols,
    )
    ma_fc = forecast_moving_average_all_skus(demand_df, forecast_periods, window=3)

    fc_df = ml_fc.merge(ma_fc, on=["periode", "SKU_ID"], how="left")
    if not skip_arima:
        arima_fc = forecast_arima_all_skus(demand_df, forecast_periods)
        fc_df = fc_df.merge(arima_fc, on=["periode", "SKU_ID"], how="left")

    fc_df = fc_df.merge(sbc_full[["SKU_ID", "ADI", "CV2", "sbc_detail"]], on="SKU_ID", how="left")

    is_lancar = fc_df["pola_sbc"] == "Lancar"
    fc_df["model_terpilih"] = np.where(is_lancar, "LightGBM Kuantil (q50)", "Hurdle Dua Tahap")
    fc_df["ramalan_terpilih"] = np.where(is_lancar, fc_df["lgbm_q50"], fc_df["hurdle_pred"]).round(2)

    round_cols = ["lgbm_q10", "lgbm_q50", "lgbm_q90", "hurdle_cond_mag", "hurdle_pred", "ma_pred"]
    if "arima_pred" in fc_df.columns:
        round_cols.append("arima_pred")
    for col in round_cols:
        fc_df[col] = fc_df[col].round(2)

    fc_df["hurdle_prob_occ"] = fc_df["hurdle_prob_occ"].round(4)
    fc_df["periode"] = pd.to_datetime(fc_df["periode"]).dt.strftime("%Y-%m-%d")
    fc_df.to_csv(out_dir / f"forecast_{slug}.csv", index=False)

    # Explainable AI (SHAP TreeExplainer)
    shap_df = compute_shap_drivers(
        q_model, h_model, step_features, sbc_full, top_k=5, feature_cols=feature_cols
    )
    shap_df.to_csv(out_dir / f"shap_drivers_{slug}.csv", index=False)

    # Ringkasan status stok (berfungsi meski tanpa data transaksi RFM)
    overstock_summary_df = build_overstock_summary(inventory_df, fc_df)
    overstock_summary_df.to_csv(out_dir / "overstock_summary.csv", index=False)

    # RFM & Rekomendasi Penyaluran Stok Berlebih (hanya jika data transaksi tersedia)
    rfm_df = pd.DataFrame()
    rec_df = pd.DataFrame()
    if tx_df is not None and not tx_df.empty:
        snapshot_date = (
            str(pd.to_datetime(inventory_df["tanggal_snapshot"]).max().strftime("%Y-%m-%d"))
            if "tanggal_snapshot" in inventory_df.columns
            else str((last_period + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d"))
        )
        rfm_df = compute_rfm_distributors(tx_df, snapshot_date=snapshot_date)
        rfm_df.to_csv(out_dir / "rfm_distributors.csv", index=False)

        rec_df = build_overstock_recommendations(inventory_df, fc_df, tx_df, rfm_df, top_n_distributors=3)
        rec_df.to_csv(out_dir / "overstock_recommendations.csv", index=False)

    # Simpan artefak model terlatih
    joblib.dump(q_model, artifacts_dir / "quantile_lgbm.joblib", compress=3)
    joblib.dump(h_model, artifacts_dir / "hurdle_two_stage.joblib", compress=3)

    meta = {
        "train_period_start": str(demand_df["periode"].min().strftime("%Y-%m-%d")),
        "train_period_end": str(demand_df["periode"].max().strftime("%Y-%m-%d")),
        "n_months": int(demand_df["periode"].nunique()),
        "forecast_periods": [d.strftime("%Y-%m-%d") for d in forecast_periods],
        "n_skus": int(demand_df["SKU_ID"].nunique()),
        "feature_cols": feature_cols,
        "sku_categories": sku_cats,
        "prod_categories": prod_cats,
        "hurdle_smearing_factor": round(h_model.smearing_factor, 4),
    }
    (artifacts_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "fc_df": fc_df,
        "shap_df": shap_df,
        "overstock_summary_df": overstock_summary_df,
        "rfm_df": rfm_df,
        "rec_df": rec_df,
        "feature_cols": feature_cols,
        "forecast_periods": forecast_periods,
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Pipeline training & forecasting StockWise AI.")
    parser.add_argument(
        "--excel",
        type=Path,
        default=None,
        help="Path ke file Excel template (.xlsx) di lokal/luar workspace.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root / "data",
        help="Direktori CSV input jika tidak memakai --excel.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "outputs",
        help="Direktori penyimpanan laporan output (.csv).",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=root / "artifacts",
        help="Direktori penyimpanan model terlatih (.joblib).",
    )
    parser.add_argument("--backtest-months", type=int, default=6)
    parser.add_argument("--forecast-months", type=int, default=3)
    parser.add_argument("--skip-arima", action="store_true", help="Lewati ARIMA pembanding agar lebih cepat.")
    return parser.parse_args()


def main() -> None:
    t0 = time.perf_counter()
    args = parse_args()

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    demand_df, inventory_df, tx_df = load_dataset(
        excel_path=args.excel,
        data_dir=None if args.excel is not None else args.data_dir,
    )

    n_months = demand_df["periode"].nunique()
    n_skus = demand_df["SKU_ID"].nunique()
    active_feats = select_feature_cols(demand_df)
    print(f"Dataset dimuat: {len(demand_df):,} baris | {n_skus} SKU | {n_months} bulan")
    print(f"Fitur aktif ({len(active_feats)}): {', '.join(active_feats)}")

    print(f"1/3 Menjalankan Backtest ({args.backtest_months} bulan terakhir)...")
    backtest_summary, rolling_summary = run_backtest(
        demand_df,
        args.out_dir,
        backtest_months=args.backtest_months,
        skip_arima=args.skip_arima,
    )

    print(f"2/3 Training Penuh, Ramalan {args.forecast_months} Bulan ke Depan, SHAP & Overstock...")
    res = run_full_training_and_forecast(
        demand_df,
        inventory_df,
        tx_df,
        args.artifacts_dir,
        args.out_dir,
        forecast_months=args.forecast_months,
        skip_arima=args.skip_arima,
    )

    elapsed = time.perf_counter() - t0
    print(f"3/3 Selesai dalam {elapsed:.2f} detik.\n")
    print("=== Ringkasan Backtest Multi-Bulan (Rekursif 6 Bulan) ===")
    print(backtest_summary.to_string(index=False))

    print("\n=== Ringkasan Backtest Rolling 1-Bulan ke Depan ===")
    print(rolling_summary.to_string(index=False))

    if not res["rfm_df"].empty:
        print("\n=== Ringkasan Segmen RFM Distributor ===")
        print(res["rfm_df"]["segmen_rfm"].value_counts().to_string())
    else:
        print("\n[Info] Sheet transaksi distributor belum diisi -> modul RFM dilewati.")

    ov_sum = res["overstock_summary_df"]
    n_overstock_skus = int((ov_sum["status_stok"] == "Overstock").sum()) if not ov_sum.empty else 0
    print(f"\nSKU dengan stok berlebih terdeteksi: {n_overstock_skus} SKU")
    print(f"Artefak model tersimpan di: {args.artifacts_dir}")
    print(f"Laporan output tersimpan di: {args.out_dir}")


if __name__ == "__main__":
    main()
