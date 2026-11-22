"""Uji kontrak EWS, pemisahan BO+OI, fallback rencana produksi PPIC, dan keluaran lighting_dummy."""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from scripts.generate_lighting_dummy import _build_lighting_specs
from src.loader import clean_demand_frame
from src.models import (
    ProductionForecaster,
    build_ews_contract_frame,
    map_sku_to_ews,
    recursive_production_and_stock_forecast,
)


ROOT = Path(__file__).resolve().parent.parent


class Phase1ContractTests(unittest.TestCase):
    def test_lighting_dummy_uses_two_digit_sku_ids(self) -> None:
        specs = _build_lighting_specs()
        self.assertEqual(len(specs), 20)
        ids = [s["SKU_ID"] for s in specs]
        self.assertEqual(ids[0], "LGT-01")
        self.assertEqual(ids[-1], "LGT-20")

    def test_sku_mapping_and_ews_contract_rounding(self) -> None:
        self.assertEqual(map_sku_to_ews("LGT-01"), "SKU-01")
        self.assertEqual(map_sku_to_ews("LGT-012"), "SKU-12")
        self.assertEqual(map_sku_to_ews("SKU-05"), "SKU-05")

        fc_df = pd.DataFrame({
            "periode": ["2026-01-01", "2026-02-01"],
            "SKU_ID": ["LGT-01", "LGT-12"],
            "FCT_Prod": [2.5, 0.49],
            "FCT_Sales": [10.49, 10.5],
            "pola_sbc": ["Lancar", "Putus-putus"],
        })
        ews_df = build_ews_contract_frame(fc_df)
        self.assertEqual(list(ews_df.columns), ["SKU", "Bulan", "FCT_Prod", "FCT_Sales", "Pola_SBC"])
        self.assertEqual(ews_df["SKU"].tolist(), ["SKU-01", "SKU-12"])
        self.assertEqual(ews_df["Bulan"].tolist(), ["2026-01", "2026-02"])
        # Pembulatan setengah ke atas (half-up)
        self.assertEqual(ews_df["FCT_Prod"].tolist(), [3, 0])
        self.assertEqual(ews_df["FCT_Sales"].tolist(), [10, 11])

    def test_loader_separates_bo_oi_from_fct_existing(self) -> None:
        raw = pd.DataFrame({
            "DATE": ["2025-01-01", "2025-02-01"],
            "Item Code": ["LGT-01", "LGT-01"],
            "Line": ["Lighting-LED", "Lighting-LED"],
            "Sales": [100, 120],
            "BO+OI": [98, 119],
        })
        cleaned = clean_demand_frame(raw)
        self.assertIn("bo_oi_recap", cleaned.columns)
        self.assertNotIn("fct_existing", cleaned.columns)

    def test_ppic_plan_overrides_ai_production_fallback(self) -> None:
        hist = pd.DataFrame({
            "periode": pd.date_range("2025-01-01", periods=6, freq="MS"),
            "SKU_ID": ["LGT-01"] * 6,
            "nama_produk": ["Bulb"] * 6,
            "kategori": ["Lighting-LED"] * 6,
            "jumlah_terjual": [100, 110, 105, 115, 120, 110],
            "act_prod": [100, 100, 120, 100, 120, 100],
            "stok_akhir": [200, 190, 205, 190, 190, 180],
        })
        sales_fc = pd.DataFrame({
            "periode": pd.to_datetime(["2026-01-01", "2026-02-01"]),
            "SKU_ID": ["LGT-01", "LGT-01"],
            "nama_produk": ["Bulb", "Bulb"],
            "kategori": ["Lighting-LED", "Lighting-LED"],
            "FCT_Sales": [100.0, 100.0],
        })
        prod_model = ProductionForecaster().fit(hist, ["LGT-01"], ["Lighting-LED"])
        planned_map = {(pd.Timestamp("2026-01-01"), "LGT-01"): 250.0}
        res = recursive_production_and_stock_forecast(
            history_df=hist,
            sales_forecast_df=sales_fc,
            initial_stock_map={"LGT-01": 180.0},
            prod_model=prod_model,
            sku_categories=["LGT-01"],
            prod_categories=["Lighting-LED"],
            planned_prod_map=planned_map,
        )
        row_jan = res[res["periode"] == pd.Timestamp("2026-01-01")].iloc[0]
        row_feb = res[res["periode"] == pd.Timestamp("2026-02-01")].iloc[0]
        self.assertEqual(row_jan["prod_source"], "rencana_ppic")
        self.assertEqual(float(row_jan["FCT_Prod"]), 250.0)
        self.assertEqual(float(row_jan["FCT_Stock"]), 330.0)
        self.assertEqual(row_feb["prod_source"], "ai_fallback")

    def test_generated_lighting_outputs_conform_to_ews_contract(self) -> None:
        ews_path = ROOT / "outputs" / "lighting_dummy" / "ews_input_jan2026_jun2026.csv"
        fc_path = ROOT / "outputs" / "lighting_dummy" / "forecast_jan2026_jun2026.csv"
        self.assertTrue(ews_path.exists(), f"File tidak ditemukan: {ews_path}")
        self.assertTrue(fc_path.exists(), f"File tidak ditemukan: {fc_path}")

        ews_df = pd.read_csv(ews_path)
        self.assertEqual(list(ews_df.columns), ["SKU", "Bulan", "FCT_Prod", "FCT_Sales", "Pola_SBC"])
        self.assertEqual(len(ews_df), 120)
        self.assertTrue(ews_df["SKU"].str.match(r"^SKU-\d{2}$").all())
        self.assertTrue(ews_df["Bulan"].str.match(r"^\d{4}-\d{2}$").all())
        self.assertTrue((ews_df["FCT_Sales"] >= 0).all())
        self.assertTrue((ews_df["FCT_Prod"] >= 0).all())

        # Pastikan tidak ada lagi prediksi Putus-putus yang kolaps ke <1 semua
        putus_sales = ews_df[ews_df["Pola_SBC"] == "Putus-putus"]["FCT_Sales"]
        self.assertGreater(float(putus_sales.mean()), 10.0)


if __name__ == "__main__":
    unittest.main()
