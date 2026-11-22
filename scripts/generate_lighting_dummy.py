"""Generator 20 SKU dummy Lighting (10 Lancar + 10 Putus-putus) meniru sifat variasi data aktual WD.

Menghasilkan workbook Excel sesuai template StockWise AI:
  data/lighting_dummy.xlsx
    - 1_Demand_Bulanan   : 20 SKU x 60 bulan (2021-01 s.d. 2025-12 = 1.200 baris)
    - 2_Inventory_Snapshot: 20 SKU posisi stok per 2025-12-31 (siap untuk horizon 2026-01 s.d. 2026-06)
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


SEED = 2026
START_PERIOD = "2021-01-01"
N_MONTHS = 60  # 2021-01 s.d. 2025-12 -> horizon 6 bulan: 2026-01 s.d. 2026-06

LEBARAN_PEAK_MONTHS = {
    (2021, 5),
    (2022, 4), (2022, 5),
    (2023, 3), (2023, 4),
    (2024, 3), (2024, 4),
    (2025, 3),
}


def _build_lighting_specs() -> list[dict]:
    """20 SKU Lighting (10 Lancar, 10 Putus-putus) dengan rentang volume heterogen seperti WD."""
    lancar_specs = [
        ("LGT-01", "LED Bulb 9W Warm White", "Lighting-LED", 18500, 0.005, 0.28, 0.24, 0.03),
        ("LGT-02", "LED Bulb 12W Cool Daylight", "Lighting-LED", 24000, 0.004, 0.30, 0.22, 0.02),
        ("LGT-03", "Downlight Panel 15W Round", "Lighting-Fixture", 4200, 0.006, 0.24, 0.28, 0.04),
        ("LGT-04", "LED Tube T8 18W 120cm", "Lighting-LED", 8600, 0.002, 0.22, 0.25, 0.03),
        ("LGT-05", "Downlight Outbow 12W Square", "Lighting-Fixture", 1450, 0.003, 0.20, 0.32, 0.05),
        ("LGT-06", "Batten Linear LED 20W", "Lighting-Fixture", 980, 0.004, 0.26, 0.35, 0.06),
        ("LGT-07", "Floodlight LED 50W IP65", "Lighting-Industrial", 640, 0.005, 0.34, 0.38, 0.05),
        ("LGT-08", "Emergency Exit Lamp LED", "Lighting-Industrial", 320, 0.002, 0.18, 0.36, 0.08),
        ("LGT-09", "Smart LED Bulb WiFi 10W", "Lighting-Smart", 510, 0.009, 0.32, 0.40, 0.06),
        ("LGT-10", "Tracklight Spotlight 15W", "Lighting-Fixture", 190, 0.003, 0.25, 0.42, 0.07),
    ]

    putus_specs = [
        ("LGT-11", "Highbay Industrial 150W", "Lighting-Industrial", 180, 0.36, 0.45, 0.55),
        ("LGT-12", "Streetlight PJU LED 90W", "Lighting-Outdoor", 420, 0.32, 0.50, 0.65),
        ("LGT-13", "Floodlight Stadium 200W", "Lighting-Outdoor", 95, 0.28, 0.48, 0.70),
        ("LGT-14", "Explosion Proof Lamp 60W", "Lighting-Industrial", 45, 0.25, 0.35, 0.50),
        ("LGT-15", "Solar Streetlight All-in-One 60W", "Lighting-Outdoor", 260, 0.40, 0.55, 0.60),
        ("LGT-16", "Architectural Wall Washer 36W", "Lighting-Outdoor", 140, 0.30, 0.40, 0.58),
        ("LGT-17", "Cleanroom Troffer LED 40W", "Lighting-Industrial", 310, 0.44, 0.38, 0.48),
        ("LGT-18", "Smart DALI Dimmer Driver 50W", "Lighting-Smart", 220, 0.48, 0.35, 0.45),
        ("LGT-19", "UV-C Germicidal Fixture 30W", "Lighting-Industrial", 70, 0.26, 0.42, 0.68),
        ("LGT-20", "Garden Bollard Light 12W", "Lighting-Outdoor", 160, 0.42, 0.40, 0.52),
    ]

    specs = []
    for sku_id, nama, kat, base, trend, seas, noise_cv, zero_p in lancar_specs:
        specs.append({
            "SKU_ID": sku_id,
            "nama_produk": nama,
            "kategori": kat,
            "pola": "Lancar",
            "base": base,
            "trend": trend,
            "seas": seas,
            "noise_cv": noise_cv,
            "occ_prob": 1.0 - zero_p,
            "std_months": 2.0,
        })
    for sku_id, nama, kat, base, occ_p, seas, noise_cv in putus_specs:
        specs.append({
            "SKU_ID": sku_id,
            "nama_produk": nama,
            "kategori": kat,
            "pola": "Putus-putus",
            "base": base,
            "trend": 0.002,
            "seas": seas,
            "noise_cv": noise_cv,
            "occ_prob": occ_p,
            "std_months": 1.6,
        })
    return specs


def generate_lighting_dataset(output_xlsx: Path) -> tuple[list[dict], list[dict]]:
    """Membangkitkan 1.200 baris historis + 20 baris snapshot inventori Lighting."""
    rng = np.random.default_rng(SEED)
    periods = pd_date_range(START_PERIOD, N_MONTHS)
    specs = _build_lighting_specs()

    demand_rows: list[dict] = []
    inv_rows: list[dict] = []

    for s_idx, spec in enumerate(specs):
        base = float(spec["base"])
        # Regime shift acak untuk beberapa SKU agar tidak terlalu mulus
        regime_month = int(rng.integers(24, 48))
        regime_mult = float(rng.choice([0.78, 1.0, 1.0, 1.25]))

        # Stok awal pada akhir 2020
        stock_prev = round(base * float(rng.uniform(1.1, 2.8)))
        std_stok = max(20.0, round(base * spec["std_months"] * spec["occ_prob"]))
        min_stok = round(0.6 * std_stok)

        recent_sales: list[float] = []
        last_to: float | None = None
        last_stock = stock_prev

        for t_idx, per in enumerate(periods):
            tahun, bulan = per.year, per.month
            is_leb = (tahun, bulan) in LEBARAN_PEAK_MONTHS
            is_nataru = bulan in (11, 12)
            cyc = 1.0 + 0.12 * np.sin(2.0 * np.pi * (bulan - 1) / 12.0)
            peak_boost = 1.0 + (spec["seas"] if (is_leb or is_nataru) else 0.0)
            trend_mult = ((1.0 + spec["trend"]) ** t_idx) * (regime_mult if t_idx >= regime_month else 1.0)

            # 1) Sales aktual (non-smooth: lognormal shock + sesekali outlier proyek)
            if rng.random() < spec["occ_prob"]:
                sigma = float(np.sqrt(np.log1p(spec["noise_cv"] ** 2)))
                mu = float(np.log(max(base * cyc * peak_boost * trend_mult, 1.0)) - 0.5 * sigma ** 2)
                sales_raw = float(rng.lognormal(mu, sigma))
                # Outlier proyek mendadak (~4% peluang pada Putus-putus, ~2% pada Lancar)
                if rng.random() < (0.04 if spec["pola"] == "Putus-putus" else 0.02):
                    sales_raw *= float(rng.uniform(1.6, 2.5))
                sales = float(round(max(1.0, sales_raw)))
            else:
                sales = 0.0

            # 2) Forecast PIC bulan sebelumnya (tidak sempurna, WAPE ~22-35%)
            ma_ref = float(np.mean(recent_sales[-3:])) if recent_sales else base * spec["occ_prob"]
            if rng.random() < 0.04:
                fct_existing = None
            else:
                # Campuran sinyal masa lalu + sinyal order parsial + human noise
                human_noise = float(rng.normal(1.0, 0.22 if spec["pola"] == "Lancar" else 0.38))
                fct_existing = float(max(0.0, round((0.45 * sales + 0.55 * ma_ref) * human_noise)))

            bp_sales = None if rng.random() < 0.08 else float(max(0.0, round(base * trend_mult * float(rng.uniform(0.85, 1.15)))))

            # 3) Keputusan produksi pabrik (ACT_Prod) dipengaruhi stok bulan lalu & ekspektasi sales
            target_inv = std_stok + 0.8 * max(ma_ref, sales * 0.5)
            needed = max(0.0, target_inv - stock_prev + (fct_existing if fct_existing is not None else ma_ref))
            if spec["pola"] == "Putus-putus":
                # Produksi batch: tidak memproduksi setiap bulan
                if stock_prev < min_stok or (needed > 0.5 * base and rng.random() < 0.55):
                    fct_prod_val = float(round(max(base * 0.8, needed * float(rng.uniform(0.85, 1.25)))))
                else:
                    fct_prod_val = 0.0
            else:
                fct_prod_val = float(round(max(0.0, needed * float(rng.uniform(0.90, 1.12)))))

            # Realisasi produksi kadang meleset / tertunda terhadap FCT Prod (serta sesekali NaN)
            if rng.random() < 0.03:
                act_prod: float | None = None
                prod_eff = 0.0
            elif fct_prod_val == 0.0:
                act_prod = 0.0 if rng.random() < 0.92 else float(round(base * 0.3))
                prod_eff = act_prod
            else:
                prod_eff = float(max(0.0, round(fct_prod_val * float(rng.normal(0.97, 0.14)))))
                act_prod = prod_eff

            # 4) Neraca stok aktual (dengan sesekali stock adjustment / scrap / retur kecil)
            adj = 0.0
            if rng.random() < 0.03:
                adj = float(round(rng.uniform(-0.08, 0.06) * max(stock_prev, 10.0)))
            stock_curr = float(max(0.0, round(stock_prev + prod_eff - sales + adj)))

            # Sesekali kosong (NaN) di stok_akhir seperti profil WD
            stok_cell: float | None = None if rng.random() < 0.02 else stock_curr

            # Turn Over (hari): #DIV/0! saat sales == 0
            if sales == 0.0:
                to_cell: float | str | None = "#DIV/0!" if rng.random() < 0.7 else None
            else:
                to_val = round(stock_curr / (sales / 30.0), 1)
                to_cell = to_val
                last_to = to_val

            demand_rows.append({
                "periode": per,
                "SKU_ID": spec["SKU_ID"],
                "nama_produk": spec["nama_produk"],
                "kategori": spec["kategori"],
                "jumlah_terjual": sales,
                "stok_akhir": stok_cell,
                "bp_sales": bp_sales,
                "fct_existing": fct_existing,
                "fct_prod": fct_prod_val,
                "act_prod": act_prod,
                "std_stok": std_stok if (t_idx % 6 == 5) else None,
                "min_stok_60pct": min_stok if (t_idx % 6 == 5) else None,
                "turn_over_hari": to_cell,
            })

            recent_sales.append(sales)
            stock_prev = stock_curr
            last_stock = stock_curr

        # Buat beberapa SKU sengaja overstock (untuk demo EWS) dan beberapa normal/rendah
        if s_idx in (2, 6, 11, 14, 17):
            last_stock = round(last_stock + std_stok * 2.5)

        inv_rows.append({
            "tanggal_snapshot": dt.date(2025, 12, 31),
            "SKU_ID": spec["SKU_ID"],
            "nama_produk": spec["nama_produk"],
            "kategori": spec["kategori"],
            "lokasi_gudang": "Gudang Cileungsi",
            "stok_tersedia": last_stock,
            "safety_stock": min_stok,
            "reorder_point": std_stok,
            "umur_stok_hari": last_to if last_to is not None else "#DIV/0!",
        })

    _save_excel(output_xlsx, demand_rows, inv_rows)
    return demand_rows, inv_rows


def pd_date_range(start: str, n_months: int) -> list[dt.date]:
    y, m, _ = (int(p) for p in start.split("-"))
    out = []
    for _ in range(n_months):
        out.append(dt.date(y, m, 1))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def _save_excel(output_xlsx: Path, demand_rows: list[dict], inv_rows: list[dict]) -> None:
    wb = openpyxl.Workbook()
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    thin = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    # Sheet 1: 1_Demand_Bulanan
    ws1 = wb.active
    ws1.title = "1_Demand_Bulanan"
    cols1 = list(demand_rows[0].keys())
    for c_idx, name in enumerate(cols1, start=1):
        c = ws1.cell(row=1, column=c_idx, value=name)
        c.fill = header_fill
        c.font = header_font
        c.alignment = Alignment(horizontal="center", vertical="center")
        c.border = border
        ws1.column_dimensions[get_column_letter(c_idx)].width = 18
    for r_idx, row in enumerate(demand_rows, start=2):
        for c_idx, col in enumerate(cols1, start=1):
            val = row[col]
            cell = ws1.cell(row=r_idx, column=c_idx, value=val)
            if col == "periode":
                cell.number_format = "yyyy-mm-dd"
    ws1.freeze_panes = "A2"

    # Sheet 2: 2_Inventory_Snapshot
    ws2 = wb.create_sheet("2_Inventory_Snapshot")
    cols2 = list(inv_rows[0].keys())
    for c_idx, name in enumerate(cols2, start=1):
        c = ws2.cell(row=1, column=c_idx, value=name)
        c.fill = header_fill
        c.font = header_font
        c.alignment = Alignment(horizontal="center", vertical="center")
        c.border = border
        ws2.column_dimensions[get_column_letter(c_idx)].width = 20
    for r_idx, row in enumerate(inv_rows, start=2):
        for c_idx, col in enumerate(cols2, start=1):
            val = row[col]
            cell = ws2.cell(row=r_idx, column=c_idx, value=val)
            if col == "tanggal_snapshot":
                cell.number_format = "yyyy-mm-dd"
    ws2.freeze_panes = "A2"

    # Sheet 3: 3_Transaksi_Distributor (kosong sesuai kondisi saat ini)
    ws3 = wb.create_sheet("3_Transaksi_Distributor")
    tx_cols = ["transaction_ID", "tanggal_transaksi", "distributor_ID", "nama_distributor", "wilayah", "SKU_ID", "kategori", "qty", "harga_satuan", "nilai_transaksi"]
    for c_idx, name in enumerate(tx_cols, start=1):
        c = ws3.cell(row=1, column=c_idx, value=name)
        c.fill = header_fill
        c.font = header_font

    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_xlsx)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    out = root / "data" / "lighting_dummy.xlsx"
    d_rows, i_rows = generate_lighting_dataset(out)
    print(f"Dummy Lighting berhasil dibuat: {out} ({len(d_rows):,} baris demand, {len(i_rows)} SKU snapshot)")
