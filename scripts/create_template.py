"""Generator template Excel (.xlsx) untuk data aktual StockWise AI."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation


HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
SUBHEADER_FILL = PatternFill("solid", fgColor="D9E1F2")
EXAMPLE_FILL = PatternFill("solid", fgColor="FFF2CC")
GUIDE_HEADER_FILL = PatternFill("solid", fgColor="2F5597")
ZEBRA_FILL = PatternFill("solid", fgColor="F9FAFC")

HEADER_FONT = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
SUBHEADER_FONT = Font(name="Calibri", size=9, italic=True, color="1F3864")
BODY_FONT = Font(name="Calibri", size=10, color="111111")
BOLD_FONT = Font(name="Calibri", size=10, bold=True, color="111111")

THIN_SIDE = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN_SIDE, right=THIN_SIDE, top=THIN_SIDE, bottom=THIN_SIDE)


def _style_data_sheet(
    ws,
    columns: list[dict],
    examples: list[list],
) -> None:
    """Mengatur header (baris 1), metadata tipe data (baris 2), contoh (baris 3+), dan format kolom."""
    ws.freeze_panes = "A3"
    ws.views.sheetView[0].showGridLines = True

    for col_idx, col in enumerate(columns, start=1):
        # Baris 1: Nama kolom exact untuk pipeline
        c1 = ws.cell(row=1, column=col_idx, value=col["name"])
        c1.fill = HEADER_FILL
        c1.font = HEADER_FONT
        c1.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c1.border = BORDER

        # Baris 2: Tipe data, status wajib/opsional, dan sumber kolom aktual
        req_tag = "WAJIB" if col["required"] else "OPSIONAL"
        meta_text = f"{col['dtype']} | {req_tag}"
        if col.get("source"):
            meta_text += f"\n[{col['source']}]"
        c2 = ws.cell(row=2, column=col_idx, value=meta_text)
        c2.fill = SUBHEADER_FILL
        c2.font = SUBHEADER_FONT
        c2.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c2.border = BORDER

        # Lebar kolom
        ws.column_dimensions[get_column_letter(col_idx)].width = col.get("width", 18)

    ws.row_dimensions[1].height = 24
    ws.row_dimensions[2].height = 30

    # Baris 3+: Contoh dummy
    for r_offset, row_vals in enumerate(examples):
        r_idx = 3 + r_offset
        ws.row_dimensions[r_idx].height = 20
        for col_idx, val in enumerate(row_vals, start=1):
            cell = ws.cell(row=r_idx, column=col_idx, value=val)
            cell.fill = EXAMPLE_FILL
            cell.font = BODY_FONT
            cell.border = BORDER
            num_fmt = columns[col_idx - 1].get("num_fmt")
            if num_fmt:
                cell.number_format = num_fmt
            if columns[col_idx - 1]["dtype"].startswith(("INT", "FLOAT")):
                cell.alignment = Alignment(horizontal="right", vertical="center")
            elif columns[col_idx - 1]["dtype"] == "DATE":
                cell.alignment = Alignment(horizontal="center", vertical="center")
            else:
                cell.alignment = Alignment(horizontal="left", vertical="center")

    # Pre-format baris kosong 5..200 agar tipe format siap saat diisi user
    for r_idx in range(3 + len(examples), 201):
        for col_idx, col in enumerate(columns, start=1):
            cell = ws.cell(row=r_idx, column=col_idx)
            if col.get("num_fmt"):
                cell.number_format = col["num_fmt"]

    # Data validation untuk kolom biner (0/1)
    for col_idx, col in enumerate(columns, start=1):
        if col.get("binary"):
            col_letter = get_column_letter(col_idx)
            dv = DataValidation(type="list", formula1='"0,1"', allow_blank=True)
            dv.error = "Isi hanya dengan 0 atau 1."
            dv.errorTitle = "Input Tidak Valid"
            ws.add_data_validation(dv)
            dv.add(f"{col_letter}3:{col_letter}5000")


def build_template(output_path: Path) -> None:
    """Membuat workbook template Excel untuk data aktual StockWise AI."""
    wb = Workbook()

    # =========================================================================
    # Sheet 0: Panduan & Kamus Kolom
    # =========================================================================
    ws_guide = wb.active
    ws_guide.title = "0_Panduan_Skema"

    intro_rows = [
        ("TEMPLATE DATA AKTUAL - STOCKWISE AI (2020-2026)", ""),
        ("Cara Pakai Aman (Privasi):", "1. Salin file ini ke folder DI LUAR workspace sebelum diisi data asli perusahaan."),
        ("", "2. Jangan mengubah nama kolom pada Baris 1 di setiap sheet data."),
        ("", "3. Baris 2 adalah keterangan tipe data & sumber kolom; baris data dimulai dari Baris 3 (timpa contoh berwarna kuning)."),
        ("", "4. Jika ada error Excel seperti #DIV/0! pada kolom Turn Over, kosongkan selnya (biarkan blank) agar terbaca sebagai NaN."),
        ("", "5. Kolom bertanda WAJIB harus terisi; kolom OPSIONAL boleh dikosongkan jika tidak tersedia di data aktual Anda."),
    ]
    for r_idx, (k, v) in enumerate(intro_rows, start=1):
        c_k = ws_guide.cell(row=r_idx, column=1, value=k)
        c_v = ws_guide.cell(row=r_idx, column=2, value=v)
        c_k.font = Font(name="Calibri", size=12 if r_idx == 1 else 10, bold=True, color="1F4E78" if r_idx == 1 else "111111")
        c_v.font = BODY_FONT

    guide_headers = ["Sheet", "Nama Kolom", "Tipe Data", "Status", "Format / Contoh", "Sumber di Sheet Aktual (WD SMI/SMII)", "Keterangan"]
    start_row = 8
    for c_idx, h in enumerate(guide_headers, start=1):
        cell = ws_guide.cell(row=start_row, column=c_idx, value=h)
        cell.fill = GUIDE_HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = BORDER
    ws_guide.row_dimensions[start_row].height = 24

    # Definisi kolom untuk semua sheet data
    demand_cols = [
        {"name": "periode", "dtype": "DATE", "required": True, "source": "Header Bulan (Oct-20 -> 2020-10-01)", "width": 16, "num_fmt": "yyyy-mm-dd", "desc": "Tanggal awal bulan (YYYY-MM-01). Urutkan per SKU."},
        {"name": "SKU_ID", "dtype": "STRING", "required": True, "source": "Item Code", "width": 18, "num_fmt": "@", "desc": "Kode unik item/produk."},
        {"name": "nama_produk", "dtype": "STRING", "required": False, "source": "Opsional (bisa sama dgn Item Code)", "width": 24, "num_fmt": "@", "desc": "Deskripsi/nama produk (boleh sama dengan SKU_ID jika tidak ada)."},
        {"name": "kategori", "dtype": "STRING", "required": True, "source": "Line", "width": 16, "num_fmt": "@", "desc": "Lini produksi / kategori produk."},
        {"name": "jumlah_terjual", "dtype": "FLOAT (>=0)", "required": True, "source": "Sales", "width": 18, "num_fmt": "#,##0.##", "desc": "TARGET UTAMA forecasting. Wajib isi 0 jika tidak ada penjualan di bulan tsb."},
        {"name": "stok_akhir", "dtype": "FLOAT (>=0)", "required": False, "source": "Stock", "width": 16, "num_fmt": "#,##0.##", "desc": "Stok akhir bulan berjalan."},
        {"name": "bp_sales", "dtype": "FLOAT (>=0)", "required": False, "source": "BP", "width": 16, "num_fmt": "#,##0.##", "desc": "Business Plan / target penjualan perusahaan."},
        {"name": "fct_existing", "dtype": "FLOAT (>=0)", "required": False, "source": "BO+OI", "width": 18, "num_fmt": "#,##0.##", "desc": "Forecast eksisting perusahaan (BO+OI) untuk benchmark di backtest."},
        {"name": "fct_prod", "dtype": "FLOAT (>=0)", "required": False, "source": "FCT Prod", "width": 16, "num_fmt": "#,##0.##", "desc": "Rencana produksi bulanan."},
        {"name": "act_prod", "dtype": "FLOAT (>=0)", "required": False, "source": "ACT Prod", "width": 16, "num_fmt": "#,##0.##", "desc": "Realisasi produksi bulanan."},
        {"name": "std_stok", "dtype": "FLOAT (>=0)", "required": False, "source": "STD", "width": 16, "num_fmt": "#,##0.##", "desc": "Standar stok / kebutuhan acuan."},
        {"name": "min_stok_60pct", "dtype": "FLOAT (>=0)", "required": False, "source": "60%XSTD Min", "width": 18, "num_fmt": "#,##0.##", "desc": "Batas stok minimum (60% x STD)."},
        {"name": "turn_over_hari", "dtype": "FLOAT (>=0)", "required": False, "source": "Turn Over (Hari)", "width": 18, "num_fmt": "#,##0.0", "desc": "Ketahanan stok (hari). Kosongkan jika di Excel bernilai #DIV/0!."},
        {"name": "harga", "dtype": "FLOAT (>=0)", "required": False, "source": "Opsional Eksternal", "width": 16, "num_fmt": "#,##0", "desc": "Harga satuan rata-rata (jika tersedia)."},
        {"name": "promo", "dtype": "INT (0/1)", "required": False, "source": "Opsional Eksternal", "width": 14, "num_fmt": "0", "binary": True, "desc": "1 = ada promo pada bulan tsb, 0 = tidak ada."},
        {"name": "diskon_pct", "dtype": "FLOAT (0-100)", "required": False, "source": "Opsional Eksternal", "width": 16, "num_fmt": "0.0", "desc": "Persentase diskon (0 s/d 100)."},
        {"name": "hari_libur_bulan", "dtype": "INT (>=0)", "required": False, "source": "Otomatis/Kalender", "width": 18, "num_fmt": "0", "desc": "Jumlah hari libur di bulan tsb (bisa diisi otomatis oleh skrip)."},
        {"name": "is_musim_puncak", "dtype": "INT (0/1)", "required": False, "source": "Otomatis/Kalender", "width": 18, "num_fmt": "0", "binary": True, "desc": "1 = musim puncak permintaan, 0 = normal."},
        {"name": "lead_time_hari", "dtype": "INT (>=0)", "required": False, "source": "Opsional Eksternal", "width": 16, "num_fmt": "0", "desc": "Lead time produksi/pengiriman dalam hari."},
    ]

    inv_cols = [
        {"name": "tanggal_snapshot", "dtype": "DATE", "required": True, "source": "Tanggal Cutoff (mis. 2026-10-31)", "width": 18, "num_fmt": "yyyy-mm-dd", "desc": "Tanggal posisi stok terakhir diambil."},
        {"name": "SKU_ID", "dtype": "STRING", "required": True, "source": "Item Code", "width": 18, "num_fmt": "@", "desc": "Kode unik item (harus cocok dengan sheet 1_Demand_Bulanan)."},
        {"name": "nama_produk", "dtype": "STRING", "required": False, "source": "Opsional", "width": 24, "num_fmt": "@", "desc": "Nama/deskripsi produk."},
        {"name": "kategori", "dtype": "STRING", "required": True, "source": "Line", "width": 16, "num_fmt": "@", "desc": "Lini produksi / kategori produk."},
        {"name": "lokasi_gudang", "dtype": "STRING", "required": False, "source": "Opsional (mis. Gudang Utama)", "width": 18, "num_fmt": "@", "desc": "Nama gudang penyimpanan."},
        {"name": "stok_tersedia", "dtype": "FLOAT (>=0)", "required": True, "source": "Stock Bulan Terakhir", "width": 18, "num_fmt": "#,##0.##", "desc": "Jumlah stok fisik saat tanggal snapshot."},
        {"name": "safety_stock", "dtype": "FLOAT (>=0)", "required": True, "source": "60%XSTD Min", "width": 18, "num_fmt": "#,##0.##", "desc": "Batas aman minimum stok untuk deteksi overstock/understock."},
        {"name": "reorder_point", "dtype": "FLOAT (>=0)", "required": False, "source": "STD (atau opsional)", "width": 18, "num_fmt": "#,##0.##", "desc": "Titik pemesanan/produksi ulang."},
        {"name": "umur_stok_hari", "dtype": "FLOAT (>=0)", "required": False, "source": "Turn Over (Hari)", "width": 18, "num_fmt": "#,##0", "desc": "Estimasi umur/ketahanan stok dalam hari."},
        {"name": "tanggal_kedaluwarsa", "dtype": "DATE", "required": False, "source": "Opsional", "width": 20, "num_fmt": "yyyy-mm-dd", "desc": "Kosongkan jika produk (mis. wearing device) tidak memiliki tanggal kedaluwarsa."},
    ]

    tx_cols = [
        {"name": "transaction_ID", "dtype": "STRING", "required": True, "source": "No. DO / Invoice (Opsional)", "width": 18, "num_fmt": "@", "desc": "ID unik transaksi (wajib hanya jika sheet ini diisi untuk modul RFM)."},
        {"name": "tanggal_transaksi", "dtype": "DATE", "required": True, "source": "Tanggal Transaksi", "width": 18, "num_fmt": "yyyy-mm-dd", "desc": "Tanggal transaksi pembelian oleh distributor/pelanggan."},
        {"name": "distributor_ID", "dtype": "STRING", "required": True, "source": "Kode Customer/Distributor", "width": 18, "num_fmt": "@", "desc": "ID unik distributor/pelanggan B2B."},
        {"name": "nama_distributor", "dtype": "STRING", "required": False, "source": "Nama Customer", "width": 24, "num_fmt": "@", "desc": "Nama distributor (boleh dianonimkan, mis. CUST-01)."},
        {"name": "wilayah", "dtype": "STRING", "required": False, "source": "Area/Region", "width": 16, "num_fmt": "@", "desc": "Wilayah distributor."},
        {"name": "SKU_ID", "dtype": "STRING", "required": True, "source": "Item Code", "width": 18, "num_fmt": "@", "desc": "Kode item yang dibeli."},
        {"name": "kategori", "dtype": "STRING", "required": False, "source": "Line", "width": 16, "num_fmt": "@", "desc": "Lini/kategori item."},
        {"name": "qty", "dtype": "FLOAT (>0)", "required": True, "source": "Qty Terjual", "width": 14, "num_fmt": "#,##0.##", "desc": "Jumlah unit terjual pada transaksi tsb."},
        {"name": "harga_satuan", "dtype": "FLOAT (>=0)", "required": False, "source": "Harga Satuan", "width": 16, "num_fmt": "#,##0", "desc": "Harga per unit."},
        {"name": "nilai_transaksi", "dtype": "FLOAT (>=0)", "required": True, "source": "Total Nilai (Rp)", "width": 18, "num_fmt": "#,##0", "desc": "Total nilai transaksi (untuk komponen Monetary RFM)."},
    ]

    all_sheets_meta = [
        ("1_Demand_Bulanan", demand_cols),
        ("2_Inventory_Snapshot", inv_cols),
        ("3_Transaksi_Distributor (Opsional)", tx_cols),
    ]

    cur_row = start_row + 1
    for sheet_label, cols in all_sheets_meta:
        for idx, col in enumerate(cols):
            vals = [
                sheet_label,
                col["name"],
                col["dtype"],
                "WAJIB" if col["required"] else "OPSIONAL",
                col["num_fmt"],
                col["source"],
                col["desc"],
            ]
            for c_idx, val in enumerate(vals, start=1):
                cell = ws_guide.cell(row=cur_row, column=c_idx, value=val)
                cell.font = BOLD_FONT if c_idx == 2 else BODY_FONT
                cell.border = BORDER
                if idx % 2 == 1:
                    cell.fill = ZEBRA_FILL
                cell.alignment = Alignment(
                    horizontal="center" if c_idx in (3, 4, 5) else "left",
                    vertical="center",
                    wrap_text=True,
                )
            ws_guide.row_dimensions[cur_row].height = 24
            cur_row += 1

    guide_widths = [26, 22, 16, 12, 16, 30, 56]
    for c_idx, w in enumerate(guide_widths, start=1):
        ws_guide.column_dimensions[get_column_letter(c_idx)].width = w
    ws_guide.freeze_panes = f"A{start_row + 1}"

    # =========================================================================
    # Sheet 1: 1_Demand_Bulanan
    # =========================================================================
    ws_dem = wb.create_sheet("1_Demand_Bulanan")
    demand_examples = [
        [dt.date(2020, 10, 1), "WD-001", "Wearing Device A (Contoh Lancar)", "Line-1", 120, 350, 130, 125, 140, 135, 300, 180, 87.5, None, 0, 0, 1, 0, 7],
        [dt.date(2020, 10, 1), "WD-002", "Wearing Device B (Contoh Putus-putus)", "Line-2", 0, 95, 10, 5, 0, 0, 80, 48, None, None, 0, 0, 1, 0, 14],
    ]
    _style_data_sheet(ws_dem, demand_cols, demand_examples)

    # =========================================================================
    # Sheet 2: 2_Inventory_Snapshot
    # =========================================================================
    ws_inv = wb.create_sheet("2_Inventory_Snapshot")
    inv_examples = [
        [dt.date(2026, 10, 31), "WD-001", "Wearing Device A (Contoh)", "Line-1", "Gudang Utama", 850, 180, 300, 120, None],
        [dt.date(2026, 10, 31), "WD-002", "Wearing Device B (Contoh)", "Line-2", "Gudang Utama", 95, 48, 80, 180, None],
    ]
    _style_data_sheet(ws_inv, inv_cols, inv_examples)

    # =========================================================================
    # Sheet 3: 3_Transaksi_Distributor (Opsional untuk RFM)
    # =========================================================================
    ws_tx = wb.create_sheet("3_Transaksi_Distributor")
    tx_examples = [
        ["TRX-0001", dt.date(2026, 10, 15), "CUST-01", "Distributor Contoh 1", "Jawa Barat", "WD-001", "Line-1", 50, 100000, 5000000],
    ]
    _style_data_sheet(ws_tx, tx_cols, tx_examples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    out = root / "templates" / "StockWise_Template_Data_Aktual.xlsx"
    build_template(out)
    print(f"Template berhasil dibuat di: {out}")
