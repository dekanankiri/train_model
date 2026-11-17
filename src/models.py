"""Model demand forecasting, backtesting, dan SHAP explainability untuk StockWise AI."""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from statsmodels.tsa.arima.model import ARIMA

from src.features import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    build_future_context_rows,
    make_step_features,
)


class QuantileLGBMForecaster:
    """Ensemble 3 model LightGBM Kuantil (q10, q50, q90) untuk pola Lancar."""

    def __init__(self, quantiles: tuple[float, ...] = (0.10, 0.50, 0.90), random_state: int = 42):
        self.quantiles = quantiles
        self.random_state = random_state
        self.models: dict[float, lgb.LGBMRegressor] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "QuantileLGBMForecaster":
        for q in self.quantiles:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=320,
                learning_rate=0.04,
                num_leaves=31,
                min_child_samples=8,
                subsample=0.85,
                subsample_freq=1,
                colsample_bytree=0.85,
                random_state=self.random_state,
                n_jobs=4,
                verbose=-1,
            )
            model.fit(X, y, categorical_feature=CATEGORICAL_COLS)
            self.models[q] = model
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        raw = np.column_stack([
            np.maximum(0.0, self.models[q].predict(X))
            for q in sorted(self.quantiles)
        ])
        # Pastikan urutan kuantil monoton (q10 <= q50 <= q90)
        sorted_q = np.sort(raw, axis=1)
        cols = [f"q{int(round(q * 100))}" for q in sorted(self.quantiles)]
        return pd.DataFrame(sorted_q, columns=cols, index=X.index)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_quantiles(X)["q50"].to_numpy()


class TwoStageHurdleForecaster:
    """Model Hurdle Dua Tahap (LGBMClassifier occurrence + LGBMRegressor log-magnitude)."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.clf = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=10,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            random_state=random_state,
            n_jobs=4,
            verbose=-1,
        )
        self.reg = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=220,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=8,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            random_state=random_state,
            n_jobs=4,
            verbose=-1,
        )
        self.smearing_factor: float = 1.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TwoStageHurdleForecaster":
        y_occ = (y > 0).astype(int)
        self.clf.fit(X, y_occ, categorical_feature=CATEGORICAL_COLS)

        pos_mask = y > 0
        X_pos = X.loc[pos_mask]
        y_pos_log = np.log1p(y.loc[pos_mask].to_numpy(dtype=float))
        self.reg.fit(X_pos, y_pos_log, categorical_feature=CATEGORICAL_COLS)

        resid = y_pos_log - self.reg.predict(X_pos)
        self.smearing_factor = float(np.clip(np.mean(np.exp(resid)), 0.8, 2.0))
        return self

    def predict_components(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        prob_occ = self.clf.predict_proba(X)[:, 1]
        log_mag = self.reg.predict(X)
        cond_mag = np.maximum(0.0, np.exp(log_mag) * self.smearing_factor - 1.0)
        expected = np.maximum(0.0, prob_occ * cond_mag)
        return prob_occ, cond_mag, expected

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        _, _, expected = self.predict_components(X)
        return expected


def recursive_ml_forecast(
    history_df: pd.DataFrame,
    future_context_df: pd.DataFrame,
    quantile_model: QuantileLGBMForecaster,
    hurdle_model: TwoStageHurdleForecaster,
    sbc_df: pd.DataFrame,
    sku_categories: list[str],
    prod_categories: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Menjalankan prediksi multi-bulan secara rekursif untuk LightGBM Kuantil & Hurdle."""
    hist = history_df.copy()
    hist["periode"] = pd.to_datetime(hist["periode"])

    fut = future_context_df.copy()
    fut["periode"] = pd.to_datetime(fut["periode"])
    future_periods = sorted(fut["periode"].unique())

    sbc_map = sbc_df.set_index("SKU_ID")["pola_utama"].to_dict()

    # Dua lintasan riwayat terpisah agar prediksi rekursif tiap model konsisten dengan lag-nya
    hist_lgbm = hist.copy()
    hist_hurdle = hist.copy()

    pred_rows = []
    step_feature_rows = []

    for dt in future_periods:
        step_ctx = fut[fut["periode"] == dt].copy()

        # 1) Step LightGBM Kuantil
        combined_lgbm = pd.concat([hist_lgbm, step_ctx], ignore_index=True)
        feat_lgbm = make_step_features(combined_lgbm, dt, sku_categories, prod_categories)
        q_preds = quantile_model.predict_quantiles(feat_lgbm[FEATURE_COLS])

        # 2) Step Hurdle Dua Tahap
        combined_hurdle = pd.concat([hist_hurdle, step_ctx], ignore_index=True)
        feat_hurdle = make_step_features(combined_hurdle, dt, sku_categories, prod_categories)
        prob_occ, cond_mag, hurdle_exp = hurdle_model.predict_components(feat_hurdle[FEATURE_COLS])

        for i in range(len(feat_lgbm)):
            sku_id = feat_lgbm.loc[i, "SKU_ID"]
            pola = sbc_map.get(sku_id, "Lancar")
            pred_rows.append({
                "periode": dt,
                "SKU_ID": sku_id,
                "nama_produk": feat_lgbm.loc[i, "nama_produk"],
                "kategori": feat_lgbm.loc[i, "kategori"],
                "pola_sbc": pola,
                "lgbm_q10": float(q_preds.loc[i, "q10"]),
                "lgbm_q50": float(q_preds.loc[i, "q50"]),
                "lgbm_q90": float(q_preds.loc[i, "q90"]),
                "hurdle_prob_occ": float(prob_occ[i]),
                "hurdle_cond_mag": float(cond_mag[i]),
                "hurdle_pred": float(hurdle_exp[i]),
            })

        # Simpan fitur langkah untuk analisis SHAP
        feat_lgbm_copy = feat_lgbm.copy()
        feat_lgbm_copy["model_source"] = "lgbm_q50"
        step_feature_rows.append(feat_lgbm_copy)

        feat_hurdle_copy = feat_hurdle.copy()
        feat_hurdle_copy["model_source"] = "hurdle"
        step_feature_rows.append(feat_hurdle_copy)

        # Update riwayat rekursif masing-masing model
        step_lgbm_update = step_ctx.sort_values("SKU_ID").reset_index(drop=True).copy()
        step_lgbm_update["jumlah_terjual"] = q_preds["q50"].to_numpy()
        hist_lgbm = pd.concat([hist_lgbm, step_lgbm_update], ignore_index=True)

        step_hurdle_update = step_ctx.sort_values("SKU_ID").reset_index(drop=True).copy()
        step_hurdle_update["jumlah_terjual"] = hurdle_exp
        hist_hurdle = pd.concat([hist_hurdle, step_hurdle_update], ignore_index=True)

    return pd.DataFrame(pred_rows), pd.concat(step_feature_rows, ignore_index=True)


def forecast_arima_all_skus(
    train_df: pd.DataFrame,
    forecast_periods: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Melakukan fitting ARIMA per SKU (seleksi orde ringan berbasis AIC)."""
    candidate_orders = [(1, 0, 1), (1, 1, 1), (0, 1, 1), (1, 0, 0)]
    horizon = len(forecast_periods)
    rows = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for sku_id, grp in train_df.groupby("SKU_ID", sort=True):
            series = grp.sort_values("periode")["jumlah_terjual"].to_numpy(dtype=float)
            best_aic = np.inf
            best_fc = None

            if np.std(series) > 1e-6:
                for order in candidate_orders:
                    try:
                        fit = ARIMA(series, order=order).fit()
                        if fit.aic < best_aic:
                            fc = fit.forecast(steps=horizon)
                            if np.all(np.isfinite(fc)):
                                best_aic = fit.aic
                                best_fc = fc
                    except Exception:
                        continue

            if best_fc is None:
                best_fc = np.full(horizon, float(np.mean(series)))

            best_fc = np.maximum(0.0, best_fc)
            for dt, val in zip(forecast_periods, best_fc):
                rows.append({"periode": dt, "SKU_ID": sku_id, "arima_pred": float(val)})

    return pd.DataFrame(rows)


def forecast_moving_average_all_skus(
    train_df: pd.DataFrame,
    forecast_periods: pd.DatetimeIndex,
    window: int = 3,
) -> pd.DataFrame:
    """Baseline statistik Rata-rata Bergerak (MA-3 rekursif per SKU)."""
    rows = []
    for sku_id, grp in train_df.groupby("SKU_ID", sort=True):
        vals = grp.sort_values("periode")["jumlah_terjual"].to_numpy(dtype=float).tolist()
        for dt in forecast_periods:
            pred = float(max(0.0, np.mean(vals[-window:])))
            vals.append(pred)
            rows.append({"periode": dt, "SKU_ID": sku_id, "ma_pred": pred})
    return pd.DataFrame(rows)


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    err = y_true - y_pred
    return float(np.mean(np.maximum(alpha * err, (alpha - 1.0) * err)))


def compute_backtest_metrics(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Menghitung metrik galat backtest per model dan per segmen pola (Lancar vs Putus-putus)."""
    model_cols = [
        ("LightGBM Kuantil (q50)", "lgbm_q50"),
        ("Hurdle Dua Tahap", "hurdle_pred"),
        ("ARIMA", "arima_pred"),
        ("Rata-rata Bergerak (MA-3)", "ma_pred"),
    ]

    subsets = [("Keseluruhan", eval_df)]
    for pola in ("Lancar", "Putus-putus"):
        sub = eval_df[eval_df["pola_sbc"] == pola]
        if not sub.empty:
            subsets.append((f"Pola {pola}", sub))

    rows = []
    for seg_name, sub in subsets:
        y = sub["jumlah_terjual"].to_numpy(dtype=float)
        y_sum = max(float(np.sum(y)), 1e-9)
        for model_name, col in model_cols:
            y_hat = sub[col].to_numpy(dtype=float)
            err = y - y_hat
            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err ** 2)))
            wape = float(100.0 * np.sum(np.abs(err)) / y_sum)
            cum_bias = float(100.0 * (np.sum(y_hat) - y_sum) / y_sum)
            pinball_q50 = _pinball_loss(y, y_hat, 0.50)
            rows.append({
                "segmen": seg_name,
                "model": model_name,
                "n_obs": int(len(sub)),
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2),
                "WAPE_pct": round(wape, 2),
                "Cum_Bias_pct": round(cum_bias, 2),
                "Pinball_q50": round(pinball_q50, 2),
            })

    return pd.DataFrame(rows)


def _extract_shap_matrix(explainer: shap.TreeExplainer, X: pd.DataFrame) -> np.ndarray:
    """Menormalisasi output shap_values ke matriks 2D (n_samples, n_features)."""
    vals = explainer.shap_values(X)
    if isinstance(vals, list):
        vals = vals[1] if len(vals) > 1 else vals[0]
    arr = np.asarray(vals, dtype=float)
    if arr.ndim == 3:
        arr = arr[:, :, 1]
    return arr


def compute_shap_drivers(
    quantile_model: QuantileLGBMForecaster,
    hurdle_model: TwoStageHurdleForecaster,
    step_features_df: pd.DataFrame,
    sbc_df: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """Menghitung faktor penggerak SHAP per SKU pada horizon Nov 2026 - Jan 2027."""
    sbc_map = sbc_df.set_index("SKU_ID")["pola_utama"].to_dict()

    lgbm_feat = step_features_df[step_features_df["model_source"] == "lgbm_q50"].reset_index(drop=True)
    hurdle_feat = step_features_df[step_features_df["model_source"] == "hurdle"].reset_index(drop=True)

    expl_lgbm = shap.TreeExplainer(quantile_model.models[0.50])
    shap_lgbm = _extract_shap_matrix(expl_lgbm, lgbm_feat[FEATURE_COLS])

    expl_occ = shap.TreeExplainer(hurdle_model.clf)
    shap_occ = _extract_shap_matrix(expl_occ, hurdle_feat[FEATURE_COLS])

    expl_mag = shap.TreeExplainer(hurdle_model.reg)
    shap_mag = _extract_shap_matrix(expl_mag, hurdle_feat[FEATURE_COLS])

    # Gabungkan kontribusi lokal tiap SKU rata-rata selama horizon 3 bulan
    rows = []
    for sku_id, pola in sorted(sbc_map.items()):
        if pola == "Lancar":
            mask = (lgbm_feat["SKU_ID"] == sku_id).to_numpy()
            sku_shap = shap_lgbm[mask].mean(axis=0)
            sku_vals = lgbm_feat.loc[mask, FEATURE_COLS].mean(numeric_only=True)
            model_used = "LightGBM Kuantil (q50)"
            satuan_efek = "unit_ramalan"
        else:
            mask = (hurdle_feat["SKU_ID"] == sku_id).to_numpy()
            # Rata-rata magnitudo absolut tertimbang antara tahap occurrence (log-odds) & magnitude (log1p)
            sku_shap = 0.5 * shap_occ[mask].mean(axis=0) + 0.5 * shap_mag[mask].mean(axis=0)
            sku_vals = hurdle_feat.loc[mask, FEATURE_COLS].mean(numeric_only=True)
            model_used = "Hurdle Dua Tahap"
            satuan_efek = "skor_gabungan_log"

        order = np.argsort(np.abs(sku_shap))[::-1][:top_k]
        for rank_idx, feat_idx in enumerate(order, start=1):
            fname = FEATURE_COLS[feat_idx]
            val = float(sku_vals[fname]) if fname in sku_vals.index else np.nan
            sval = float(sku_shap[feat_idx])
            rows.append({
                "SKU_ID": sku_id,
                "pola_sbc": pola,
                "model_penjelas": model_used,
                "peringkat_faktor": rank_idx,
                "fitur": fname.replace("_cat", ""),
                "rata_rata_nilai_fitur": round(val, 3) if np.isfinite(val) else None,
                "shap_kontribusi": round(sval, 4),
                "arah_pengaruh": "Menaikkan" if sval >= 0 else "Menurunkan",
                "satuan_efek": satuan_efek,
            })

    return pd.DataFrame(rows)
