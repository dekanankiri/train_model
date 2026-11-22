"""Model demand forecasting, backtesting, dan SHAP explainability untuk StockWise AI."""

from __future__ import annotations

import re
import warnings
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from statsmodels.tsa.arima.model import ARIMA

from src.features import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    HYBRID_EXTRA_COLS,
    PROD_FEATURE_COLS,
    _compute_lag_features_frame,
    add_hybrid_bo_oi_features,
    build_future_context_rows,
    make_production_step_features,
    make_step_features,
    prepare_production_training_frame,
)


class QuantileLGBMForecaster:
    """Model LightGBM Kuantil (default 1 titik median q50 pada skala log1p)."""

    def __init__(self, quantiles: tuple[float, ...] = (0.50,), random_state: int = 42):
        self.quantiles = quantiles
        self.random_state = random_state
        self.models: dict[float, lgb.LGBMRegressor] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "QuantileLGBMForecaster":
        y_log = np.log1p(np.maximum(0.0, y.to_numpy(dtype=float)))
        for q in self.quantiles:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=320,
                learning_rate=0.04,
                num_leaves=31,
                min_child_samples=10,
                reg_alpha=0.1,
                reg_lambda=1.0,
                subsample=0.85,
                subsample_freq=1,
                colsample_bytree=0.85,
                random_state=self.random_state,
                n_jobs=4,
                verbose=-1,
            )
            model.fit(X, y_log, categorical_feature=CATEGORICAL_COLS)
            self.models[q] = model
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        raw = np.column_stack([
            np.maximum(0.0, np.expm1(self.models[q].predict(X)))
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
            reg_alpha=0.1,
            reg_lambda=1.0,
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
            min_child_samples=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            random_state=random_state,
            n_jobs=4,
            verbose=-1,
        )
        self.smearing_factor: float = 1.0
        self._const_prob: float | None = None
        self._empty_pos: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TwoStageHurdleForecaster":
        y_occ = (y > 0).astype(int)
        if y_occ.nunique() < 2:
            self._const_prob = float(y_occ.iloc[0]) if len(y_occ) > 0 else 0.0
        else:
            self._const_prob = None
            self.clf.fit(X, y_occ, categorical_feature=CATEGORICAL_COLS)

        pos_mask = y > 0
        if not pos_mask.any():
            self._empty_pos = True
            self.smearing_factor = 1.0
            return self

        self._empty_pos = False
        X_pos = X.loc[pos_mask]
        y_pos_log = np.log1p(y.loc[pos_mask].to_numpy(dtype=float))
        self.reg.fit(X_pos, y_pos_log, categorical_feature=CATEGORICAL_COLS)

        resid = y_pos_log - self.reg.predict(X_pos)
        lo, hi = np.quantile(resid, [0.05, 0.95])
        resid_trimmed = np.clip(resid, lo, hi)
        self.smearing_factor = float(np.clip(np.mean(np.exp(resid_trimmed)), 0.95, 1.25))
        return self

    def predict_components(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._const_prob is not None:
            prob_occ = np.full(len(X), self._const_prob, dtype=float)
        else:
            prob_occ = self.clf.predict_proba(X)[:, 1]

        if self._empty_pos:
            cond_mag = np.zeros(len(X), dtype=float)
        else:
            log_mag = self.reg.predict(X)
            cond_mag = np.maximum(0.0, np.exp(log_mag) * self.smearing_factor - 1.0)
        expected = np.maximum(0.0, prob_occ * cond_mag)
        return prob_occ, cond_mag, expected

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        _, _, expected = self.predict_components(X)
        return expected


class TweedieLGBMForecaster:
    """LightGBM Tweedie (log-link Compound Poisson-Gamma) untuk taksiran mean bebas bias lintas skala."""

    def __init__(self, variance_power: float = 1.35, random_state: int = 42):
        self.variance_power = variance_power
        self.random_state = random_state
        self.model = lgb.LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=variance_power,
            n_estimators=320,
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            random_state=random_state,
            n_jobs=4,
            verbose=-1,
        )
        self._all_zero: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TweedieLGBMForecaster":
        y_clean = np.maximum(0.0, y.to_numpy(dtype=float))
        if not np.any(y_clean > 0):
            self._all_zero = True
            return self
        self._all_zero = False
        self.model.fit(X, y_clean, categorical_feature=CATEGORICAL_COLS)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._all_zero:
            return np.zeros(len(X), dtype=float)
        return np.maximum(0.0, self.model.predict(X))


class HybridBOOILGBMForecaster:
    """Model Hybrid (AI + BO+OI): mempelajari koreksi residual log terhadap Forecast Eksisting BO+OI."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.reg = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=160,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=20,
            reg_alpha=1.0,
            reg_lambda=5.0,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.80,
            random_state=random_state,
            n_jobs=4,
            verbose=-1,
        )
        self.smearing_factor: float = 1.0
        self.hybrid_cols: list[str] = []
        self._is_fitted: bool = False

    def fit(self, feat_df: pd.DataFrame, base_feature_cols: list[str]) -> "HybridBOOILGBMForecaster":
        self.hybrid_cols = list(base_feature_cols) + HYBRID_EXTRA_COLS
        if "fct_existing" not in feat_df.columns:
            self._is_fitted = False
            return self

        valid = feat_df[feat_df["fct_existing"].notna()].copy()
        if len(valid) < 30:
            self._is_fitted = False
            return self

        y_true = np.maximum(0.0, valid["jumlah_terjual"].to_numpy(dtype=float))
        log_fct = np.log1p(np.maximum(0.0, valid["fct_existing"].to_numpy(dtype=float)))
        y_res = np.clip(np.log1p(y_true) - log_fct, -1.5, 1.5)

        X_hyb = valid[self.hybrid_cols]
        self.reg.fit(X_hyb, y_res, categorical_feature=CATEGORICAL_COLS)

        resid = y_res - self.reg.predict(X_hyb)
        lo, hi = np.quantile(resid, [0.05, 0.95])
        self.smearing_factor = float(np.clip(np.mean(np.exp(np.clip(resid, lo, hi))), 0.97, 1.05))
        self._is_fitted = True
        return self

    def predict(self, feat_df: pd.DataFrame) -> np.ndarray:
        out = np.full(len(feat_df), np.nan, dtype=float)
        if not self._is_fitted or "fct_existing" not in feat_df.columns:
            return out

        mask = feat_df["fct_existing"].notna().to_numpy()
        if not mask.any():
            return out

        sub = feat_df.loc[mask]
        delta_hat = np.clip(self.reg.predict(sub[self.hybrid_cols]), -1.0, 1.0)
        log_fct = np.log1p(np.maximum(0.0, sub["fct_existing"].to_numpy(dtype=float)))
        out[mask] = np.maximum(0.0, np.exp(log_fct + delta_hat) * self.smearing_factor - 1.0)
        return out


class ProductionForecaster:
    """Model ramalan realisasi produksi (ACT_Prod) berbasis Two-Stage Hurdle + fitur stok & FCT_Sales."""

    def __init__(self, random_state: int = 42):
        self.core = TwoStageHurdleForecaster(random_state=random_state)
        self._is_fitted: bool = False

    def fit(
        self,
        demand_df: pd.DataFrame,
        sku_categories: list[str],
        prod_categories: list[str],
    ) -> "ProductionForecaster":
        if "act_prod" not in demand_df.columns or not demand_df["act_prod"].notna().any():
            self._is_fitted = False
            return self
        prod_feat = prepare_production_training_frame(demand_df, sku_categories, prod_categories)
        if prod_feat.empty:
            self._is_fitted = False
            return self
        self.core.fit(prod_feat[PROD_FEATURE_COLS], prod_feat["act_prod"])
        self._is_fitted = True
        return self

    def predict(self, step_feat_df: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            return np.zeros(len(step_feat_df), dtype=float)
        return self.core.predict(step_feat_df[PROD_FEATURE_COLS])


def recursive_production_and_stock_forecast(
    history_df: pd.DataFrame,
    sales_forecast_df: pd.DataFrame,
    initial_stock_map: dict[str, float],
    prod_model: ProductionForecaster,
    sku_categories: list[str],
    prod_categories: list[str],
    planned_prod_map: dict[tuple[pd.Timestamp, str], float] | None = None,
) -> pd.DataFrame:
    """Memproyeksikan FCT_Prod (rencana PPIC bila ada, fallback AI) dan FCT_Stock (t=1..H)."""
    hist = history_df.copy()
    hist["periode"] = pd.to_datetime(hist["periode"])

    fc_sales = sales_forecast_df.copy()
    fc_sales["periode"] = pd.to_datetime(fc_sales["periode"])
    future_periods = sorted(fc_sales["periode"].unique())

    stock_state = {str(k): max(0.0, float(v)) for k, v in initial_stock_map.items()}

    # Siapkan baseline MA-3 produksi per SKU
    ma_hist: dict[str, list[float]] = {}
    if "act_prod" in hist.columns:
        for sku_id, grp in hist.groupby("SKU_ID", sort=True):
            vals = pd.to_numeric(grp.sort_values("periode")["act_prod"], errors="coerce").dropna().tolist()
            ma_hist[str(sku_id)] = vals

    hist_prod = hist.copy()
    rows = []

    for dt in future_periods:
        step_sales = fc_sales[fc_sales["periode"] == dt].sort_values("SKU_ID").copy()
        sales_col = "FCT_Sales" if "FCT_Sales" in step_sales.columns else "ramalan_terpilih"

        # Sisipkan saldo stok akhir t-1 pada baris terakhir histori tiap SKU agar stok_akhir_lag_1 akurat
        if not hist_prod.empty:
            last_p = hist_prod["periode"].max()
            last_mask = hist_prod["periode"] == last_p
            if last_mask.any():
                hist_prod.loc[last_mask, "stok_akhir"] = hist_prod.loc[last_mask, "SKU_ID"].map(
                    lambda s: stock_state.get(str(s), 0.0)
                )

        step_ctx = step_sales[["periode", "SKU_ID", "nama_produk", "kategori"]].copy()
        step_ctx["tahun"] = pd.to_datetime(step_ctx["periode"]).dt.year.astype(int)
        step_ctx["bulan"] = pd.to_datetime(step_ctx["periode"]).dt.month.astype(int)
        step_ctx["jumlah_terjual"] = step_sales[sales_col].to_numpy(dtype=float)
        step_ctx["fct_sales_input"] = step_sales[sales_col].to_numpy(dtype=float)
        step_ctx["act_prod"] = np.nan
        step_ctx["stok_akhir"] = np.nan

        combined = pd.concat([hist_prod, step_ctx], ignore_index=True)
        feat_step = make_production_step_features(combined, dt, sku_categories, prod_categories)
        fct_prod_preds = prod_model.predict(feat_step)

        step_ctx_sorted = step_ctx.sort_values("SKU_ID").reset_index(drop=True)
        resolved_prod = np.zeros(len(feat_step), dtype=float)
        for i in range(len(feat_step)):
            sku_id = str(feat_step.loc[i, "SKU_ID"])
            f_sales = max(0.0, float(feat_step.loc[i, "sales_curr"]))
            plan_key = (pd.Timestamp(dt), sku_id)
            if planned_prod_map is not None and plan_key in planned_prod_map:
                f_prod = max(0.0, float(planned_prod_map[plan_key]))
                prod_src = "rencana_ppic"
            else:
                f_prod = max(0.0, float(fct_prod_preds[i]))
                prod_src = "ai_fallback"
            resolved_prod[i] = f_prod
            s_prev = stock_state.get(sku_id, 0.0)
            net_stock = s_prev + f_prod - f_sales
            f_stock = max(0.0, net_stock)
            deficit = max(0.0, -net_stock)
            stock_state[sku_id] = f_stock

            m_vals = ma_hist.setdefault(sku_id, [])
            ma_p = float(max(0.0, np.mean(m_vals[-3:]))) if m_vals else 0.0
            m_vals.append(ma_p)

            rows.append({
                "periode": dt,
                "SKU_ID": sku_id,
                "FCT_Prod": f_prod,
                "prod_source": prod_src,
                "FCT_Stock": f_stock,
                "proyeksi_defisit": deficit,
                "ma_prod_pred": ma_p,
            })

        step_update = step_ctx_sorted.copy()
        step_update["act_prod"] = resolved_prod
        step_update["stok_akhir"] = [stock_state[str(s)] for s in step_update["SKU_ID"]]
        hist_prod = pd.concat([hist_prod, step_update], ignore_index=True)

    return pd.DataFrame(rows)


def compute_production_stock_metrics(eval_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Menghitung metrik evaluasi Production (ACT_Prod) dan Proyeksi Stok (FCT_Stock)."""
    prod_rows = []
    prod_candidates = [
        ("AI Production (Hurdle)", "FCT_Prod"),
        ("FCT Prod Eksisting", "fct_prod"),
        ("MA-3 Production", "ma_prod_pred"),
    ]
    for name, col in prod_candidates:
        if col not in eval_df.columns or "act_prod" not in eval_df.columns:
            continue
        valid = eval_df[eval_df["act_prod"].notna() & eval_df[col].notna()]
        if valid.empty:
            continue
        y = valid["act_prod"].to_numpy(dtype=float)
        y_hat = valid[col].to_numpy(dtype=float)
        y_sum = max(float(np.sum(y)), 1e-9)
        err = y - y_hat
        prod_rows.append({
            "target": "ACT_Prod",
            "model": name,
            "n_obs": int(len(valid)),
            "MAE": round(float(np.mean(np.abs(err))), 2),
            "RMSE": round(float(np.sqrt(np.mean(err ** 2))), 2),
            "WAPE_pct": round(float(100.0 * np.sum(np.abs(err)) / y_sum), 2),
            "Cum_Bias_pct": round(float(100.0 * (np.sum(y_hat) - y_sum) / y_sum), 2),
        })

    stock_rows = []
    if "stok_akhir" in eval_df.columns and "FCT_Stock" in eval_df.columns:
        valid_s = eval_df[eval_df["stok_akhir"].notna() & eval_df["FCT_Stock"].notna()]
        if not valid_s.empty:
            y = valid_s["stok_akhir"].to_numpy(dtype=float)
            y_hat = valid_s["FCT_Stock"].to_numpy(dtype=float)
            y_sum = max(float(np.sum(y)), 1e-9)
            err = y - y_hat
            stock_rows.append({
                "target": "Stock_Akhir",
                "model": "Neraca Stok AI (FCT_Prod - FCT_Sales)",
                "n_obs": int(len(valid_s)),
                "MAE": round(float(np.mean(np.abs(err))), 2),
                "RMSE": round(float(np.sqrt(np.mean(err ** 2))), 2),
                "WAPE_pct": round(float(100.0 * np.sum(np.abs(err)) / y_sum), 2),
                "Cum_Bias_pct": round(float(100.0 * (np.sum(y_hat) - y_sum) / y_sum), 2),
            })

    return pd.DataFrame(prod_rows), pd.DataFrame(stock_rows)


def recursive_ml_forecast(
    history_df: pd.DataFrame,
    future_context_df: pd.DataFrame,
    quantile_model: QuantileLGBMForecaster,
    hurdle_model: TwoStageHurdleForecaster,
    sbc_df: pd.DataFrame,
    sku_categories: list[str],
    prod_categories: list[str],
    feature_cols: list[str] | None = None,
    tweedie_model: TweedieLGBMForecaster | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Menjalankan prediksi multi-bulan secara rekursif untuk LightGBM (Tweedie & Kuantil) serta Hurdle."""
    active_cols = feature_cols or FEATURE_COLS
    hist = history_df.copy()
    hist["periode"] = pd.to_datetime(hist["periode"])

    fut = future_context_df.copy()
    fut["periode"] = pd.to_datetime(fut["periode"])
    future_periods = sorted(fut["periode"].unique())

    sbc_map = sbc_df.set_index("SKU_ID")["pola_utama"].to_dict()

    # Lintasan riwayat terpisah agar prediksi rekursif tiap model konsisten dengan lag-nya
    hist_lgbm = hist.copy()
    hist_tweedie = hist.copy()
    hist_hurdle = hist.copy()

    pred_rows = []
    step_feature_rows = []

    for dt in future_periods:
        step_ctx = fut[fut["periode"] == dt].copy()

        # 1) Step LightGBM Kuantil
        combined_lgbm = pd.concat([hist_lgbm, step_ctx], ignore_index=True)
        feat_lgbm = make_step_features(combined_lgbm, dt, sku_categories, prod_categories, feature_cols=active_cols)
        q_preds = quantile_model.predict_quantiles(feat_lgbm[active_cols])

        # 2) Step LightGBM Tweedie
        if tweedie_model is not None:
            combined_tweedie = pd.concat([hist_tweedie, step_ctx], ignore_index=True)
            feat_tweedie = make_step_features(combined_tweedie, dt, sku_categories, prod_categories, feature_cols=active_cols)
            tweedie_preds = tweedie_model.predict(feat_tweedie[active_cols])
        else:
            feat_tweedie = feat_lgbm
            tweedie_preds = q_preds["q50"].to_numpy()

        # 3) Step Hurdle Dua Tahap
        combined_hurdle = pd.concat([hist_hurdle, step_ctx], ignore_index=True)
        feat_hurdle = make_step_features(combined_hurdle, dt, sku_categories, prod_categories, feature_cols=active_cols)
        prob_occ, cond_mag, hurdle_exp = hurdle_model.predict_components(feat_hurdle[active_cols])

        for i in range(len(feat_lgbm)):
            sku_id = feat_lgbm.loc[i, "SKU_ID"]
            pola = sbc_map.get(sku_id, "Lancar")
            pred_rows.append({
                "periode": dt,
                "SKU_ID": sku_id,
                "nama_produk": feat_lgbm.loc[i, "nama_produk"],
                "kategori": feat_lgbm.loc[i, "kategori"],
                "pola_sbc": pola,
                "lgbm_tweedie": float(tweedie_preds[i]),
                "lgbm_q50": float(q_preds.loc[i, "q50"]),
                "hurdle_prob_occ": float(prob_occ[i]),
                "hurdle_cond_mag": float(cond_mag[i]),
                "hurdle_pred": float(hurdle_exp[i]),
            })

        # Simpan fitur langkah untuk analisis SHAP (lgbm_q50 untuk Putus-putus, hurdle untuk Lancar)
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

        step_tw_update = step_ctx.sort_values("SKU_ID").reset_index(drop=True).copy()
        step_tw_update["jumlah_terjual"] = tweedie_preds
        hist_tweedie = pd.concat([hist_tweedie, step_tw_update], ignore_index=True)

        step_hurdle_update = step_ctx.sort_values("SKU_ID").reset_index(drop=True).copy()
        step_hurdle_update["jumlah_terjual"] = hurdle_exp
        hist_hurdle = pd.concat([hist_hurdle, step_hurdle_update], ignore_index=True)

    return pd.DataFrame(pred_rows), pd.concat(step_feature_rows, ignore_index=True)


def rolling_1step_ml_forecast(
    full_df: pd.DataFrame,
    test_periods: pd.DatetimeIndex,
    quantile_model: QuantileLGBMForecaster,
    hurdle_model: TwoStageHurdleForecaster,
    sbc_df: pd.DataFrame,
    sku_categories: list[str],
    prod_categories: list[str],
    feature_cols: list[str] | None = None,
    tweedie_model: TweedieLGBMForecaster | None = None,
    hybrid_model: HybridBOOILGBMForecaster | None = None,
) -> pd.DataFrame:
    """Prediksi rolling 1-bulan ke depan memakai fitur lag aktual sampai bulan t-1 (tanpa kebocoran)."""
    active_cols = feature_cols or FEATURE_COLS
    feat_df = _compute_lag_features_frame(full_df)
    if hybrid_model is not None and hybrid_model._is_fitted:
        feat_df = add_hybrid_bo_oi_features(feat_df)

    test_feat = feat_df[feat_df["periode"].isin(test_periods)].copy()
    test_feat["SKU_ID_cat"] = pd.Categorical(test_feat["SKU_ID"], categories=sku_categories)
    test_feat["kategori_cat"] = pd.Categorical(test_feat["kategori"], categories=prod_categories)

    for col in active_cols:
        if col not in CATEGORICAL_COLS:
            test_feat[col] = test_feat[col].astype("float32")

    test_feat = test_feat.sort_values(["periode", "SKU_ID"]).reset_index(drop=True)
    q_preds = quantile_model.predict_quantiles(test_feat[active_cols])
    tweedie_preds = (
        tweedie_model.predict(test_feat[active_cols])
        if tweedie_model is not None
        else q_preds["q50"].to_numpy()
    )
    prob_occ, cond_mag, hurdle_exp = hurdle_model.predict_components(test_feat[active_cols])

    sbc_map = sbc_df.set_index("SKU_ID")["pola_utama"].to_dict()
    out = pd.DataFrame({
        "periode": test_feat["periode"],
        "SKU_ID": test_feat["SKU_ID"],
        "pola_sbc": test_feat["SKU_ID"].map(sbc_map).fillna("Lancar"),
        "lgbm_tweedie": tweedie_preds,
        "lgbm_q50": q_preds["q50"].to_numpy(),
        "hurdle_prob_occ": prob_occ,
        "hurdle_cond_mag": cond_mag,
        "hurdle_pred": hurdle_exp,
        "ma_pred": test_feat["rolling_mean_3"].fillna(0.0).to_numpy(),
    })
    if hybrid_model is not None and hybrid_model._is_fitted:
        out["hybrid_pred"] = hybrid_model.predict(test_feat)
    return out


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
    candidate_models = [
        ("LightGBM Tweedie", "lgbm_tweedie"),
        ("LightGBM Kuantil (q50)", "lgbm_q50"),
        ("Hurdle Dua Tahap", "hurdle_pred"),
        ("Hybrid (AI + Forecast PIC)", "hybrid_pred"),
        ("ARIMA", "arima_pred"),
        ("Rata-rata Bergerak (MA-3)", "ma_pred"),
        ("Forecast Eksisting (PIC)", "fct_existing"),
        ("Rekap Order Akhir Bulan (BO+OI)", "bo_oi_recap"),
        ("Business Plan (BP)", "bp_sales"),
    ]
    model_cols = [
        (name, col) for name, col in candidate_models
        if col in eval_df.columns and eval_df[col].notna().any()
    ]

    subsets = [("Keseluruhan", eval_df)]
    for pola in ("Lancar", "Putus-putus"):
        sub = eval_df[eval_df["pola_sbc"] == pola]
        if not sub.empty:
            subsets.append((f"Pola {pola}", sub))

    rows = []
    for seg_name, sub in subsets:
        for model_name, col in model_cols:
            valid = sub[sub[col].notna()]
            if valid.empty:
                continue
            y = valid["jumlah_terjual"].to_numpy(dtype=float)
            y_sum = max(float(np.sum(y)), 1e-9)
            y_hat = valid[col].to_numpy(dtype=float)
            err = y - y_hat
            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err ** 2)))
            wape = float(100.0 * np.sum(np.abs(err)) / y_sum)
            cum_bias = float(100.0 * (np.sum(y_hat) - y_sum) / y_sum)
            pinball_q50 = _pinball_loss(y, y_hat, 0.50)
            rows.append({
                "segmen": seg_name,
                "model": model_name,
                "n_obs": int(len(valid)),
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
    feature_cols: list[str] | None = None,
    tweedie_model: TweedieLGBMForecaster | None = None,
    intermittent_model_key: str = "hurdle",
) -> pd.DataFrame:
    """Menghitung faktor penggerak SHAP per SKU sesuai model terpilih (default Hurdle untuk seluruh pola)."""
    active_cols = feature_cols or FEATURE_COLS
    sbc_map = sbc_df.set_index("SKU_ID")["pola_utama"].to_dict()

    lgbm_feat = step_features_df[step_features_df["model_source"] == "lgbm_q50"].reset_index(drop=True)
    hurdle_feat = step_features_df[step_features_df["model_source"] == "hurdle"].reset_index(drop=True)

    expl_lgbm = shap.TreeExplainer(quantile_model.models[0.50])
    shap_lgbm = _extract_shap_matrix(expl_lgbm, lgbm_feat[active_cols]) if not lgbm_feat.empty else np.empty((0, len(active_cols)))

    if hurdle_model._const_prob is None:
        expl_occ = shap.TreeExplainer(hurdle_model.clf)
        shap_occ = _extract_shap_matrix(expl_occ, hurdle_feat[active_cols]) if not hurdle_feat.empty else np.empty((0, len(active_cols)))
    else:
        shap_occ = np.zeros((len(hurdle_feat), len(active_cols)), dtype=float)

    if not hurdle_model._empty_pos:
        expl_mag = shap.TreeExplainer(hurdle_model.reg)
        shap_mag = _extract_shap_matrix(expl_mag, hurdle_feat[active_cols]) if not hurdle_feat.empty else np.empty((0, len(active_cols)))
    else:
        shap_mag = np.zeros((len(hurdle_feat), len(active_cols)), dtype=float)

    # Gabungkan kontribusi lokal tiap SKU rata-rata selama horizon ramalan
    rows = []
    for sku_id, pola in sorted(sbc_map.items()):
        use_hurdle = (pola == "Lancar") or (intermittent_model_key == "hurdle")
        if use_hurdle:
            mask = (hurdle_feat["SKU_ID"] == sku_id).to_numpy()
            if not mask.any():
                continue
            sku_shap = 0.5 * shap_occ[mask].mean(axis=0) + 0.5 * shap_mag[mask].mean(axis=0)
            sku_vals = hurdle_feat.loc[mask, active_cols].mean(numeric_only=True)
            model_used = "Hurdle Dua Tahap"
            satuan_efek = "skor_gabungan_log"
        else:
            mask = (lgbm_feat["SKU_ID"] == sku_id).to_numpy()
            if not mask.any():
                continue
            sku_shap = shap_lgbm[mask].mean(axis=0)
            sku_vals = lgbm_feat.loc[mask, active_cols].mean(numeric_only=True)
            model_used = "LightGBM Kuantil (q50)"
            satuan_efek = "skor_log_ramalan"

        order = np.argsort(np.abs(sku_shap))[::-1][:top_k]
        for rank_idx, feat_idx in enumerate(order, start=1):
            fname = active_cols[feat_idx]
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


def _round_half_up_nonneg(val: float) -> int:
    """Membulatkan setengah ke atas ke integer non-negatif sesuai kontrak EWS."""
    if pd.isna(val):
        return 0
    return int(Decimal(str(max(0.0, float(val)))).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def map_sku_to_ews(sku_id: str) -> str:
    """Memetakan SKU_ID modul Ramalan (LGT-nn / LGT-nnn / SKU-nnn) ke format baku SKU-nn."""
    s = str(sku_id).strip()
    m = re.match(r"^(?:LGT|SKU)-0*(\d+)$", s, flags=re.IGNORECASE)
    if m:
        return f"SKU-{int(m.group(1)):02d}"
    return s


def build_ews_contract_frame(fc_df: pd.DataFrame) -> pd.DataFrame:
    """Menyusun tabel keluaran sesuai Kontrak Data Terpadu untuk Tabel Input 2 EWS."""
    if fc_df.empty:
        return pd.DataFrame(columns=["SKU", "Bulan", "FCT_Prod", "FCT_Sales", "Pola_SBC"])

    out = pd.DataFrame({
        "SKU": fc_df["SKU_ID"].map(map_sku_to_ews),
        "Bulan": pd.to_datetime(fc_df["periode"]).dt.strftime("%Y-%m"),
        "FCT_Prod": [
            _round_half_up_nonneg(v) for v in (fc_df["FCT_Prod"] if "FCT_Prod" in fc_df.columns else np.zeros(len(fc_df)))
        ],
        "FCT_Sales": [_round_half_up_nonneg(v) for v in fc_df["FCT_Sales"]],
        "Pola_SBC": fc_df["pola_sbc"].astype(str),
    })
    return out.sort_values(["SKU", "Bulan"]).reset_index(drop=True)
