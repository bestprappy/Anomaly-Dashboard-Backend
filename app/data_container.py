"""
DataBillContainer
==================

Ingests the 5 raw billing exports (PEA: BFKT, TUC / MEA: BFKT, TUC, TMV),
cleans + reshapes them into one long "master" table, and exposes the
EDA aggregations the dashboard needs.

The cleaning logic here is a direct port of the exploratory work done in
`True_billing.ipynb` (PEA) and `MEA_cleaning.ipynb` (MEA), generalised so it
works against uploaded CSV/XLSX file objects instead of hardcoded paths.

Master table schema (self.master_df), one row per site per month:

    provider        'PEA' | 'MEA'
    company         'BFKT' | 'TUC' | 'TMV'
    Meter_No        str   (kept as string, some meters have leading zeros)
    Site_ID         str   (raw, upper-stripped)
    site_type       str   normalised type / status label
    Rate_CAT        str
    TOU_TOD         str
    Province        str
    month           int   YYYYMM (Gregorian)
    date            datetime64 (first day of month)
    bill_amount     float baht
    kwh             float
    bill_class      'zero' | 'maintenance' | 'active'
    is_shutdown     bool  (MEA only concept, False for PEA rows)
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
import time
from typing import Optional, Union, BinaryIO

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FileLike = Union[str, bytes, BinaryIO]

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

SITE_ID_RE = re.compile(
    r'^[A-Z]{2,5}\d{2,5}(?:[A-Z]{1,3}\d?)?$'
)

MEA_METER_COL = 'Meter_No'
MEA_SITE_COL_CANDIDATES = ['MSC/RMSC/IBC/WIFI/Decom', 'MSC/RMSC/IBC/WIFI/DeCom']
MEA_SITE_ID = 'Site_ID'

PEA_SITE_TYPE_COL_CANDIDATES = ['MSC/RMSC/IBC/WIFI/DeCom', 'MSC/RMSC/IBC/WIFI/DN/PN']

LAST_MONTH_WINDOWS = (3, 6, 9)


def _read_any(file: FileLike, header=0) -> pd.DataFrame:
    """Read a csv or xlsx upload into a DataFrame, tolerant of file-likes / bytes."""
    if isinstance(file, (bytes, bytearray)):
        file = io.BytesIO(file)
    if hasattr(file, "read"):
        pos = file.tell() if hasattr(file, "tell") else None
        try:
            df = pd.read_csv(file, header=header, low_memory=False)
            if df.shape[1] <= 1:  # single-column result usually means it wasn't really CSV
                raise ValueError("suspicious single-column parse")
            return df
        except Exception:
            if pos is not None:
                file.seek(pos)
            return pd.read_excel(file, header=header)
    if str(file).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(file, header=header)
    return pd.read_csv(file, header=header, low_memory=False)

def _clean_numeric_string(series: pd.Series) -> pd.Series:
    """
    Clean PEA/MEA numeric columns before pd.to_numeric:
      - strip whitespace/quotes
      - treat '-' (dash placeholder for zero) as 0
      - strip thousands-separator commas
    """
    s = series.astype(str).str.strip().str.strip('"').str.strip()
    s = s.replace(r'^-+$', '0', regex=True)   # bare dash(es) -> 0
    s = s.str.replace(',', '', regex=False)
    s = s.str.strip()
    return s

def _fix_numeric_col(col) -> str:
    try:
        return str(int(float(col)))
    except Exception:
        return str(col)


def _normalise_mea_site_type(val) -> str:
    v = str(val).strip().upper()
    if v in ('0', 'NAN', ''):
        return 'NORMAL'
    if v == 'WIFI':
        return 'WIFI'
    if 'ยกเลิก' in str(val):
        return 'WIFI_CANCEL'
    if 'ไม่จ่าย' in str(val):
        return 'WIFI_NOPAY'
    if v.startswith('WIFI'):
        return 'WIFI_OTHER'
    if v in ('DECOM', 'DECOMM'):
        return 'DECOM'
    if v == 'MSC':
        return 'MSC'
    return v


def classify_bill(amount) -> str:
    if pd.isna(amount) or amount == 0:
        return 'zero'
    elif amount < 200:
        return 'maintenance'
    else:
        return 'active'


@dataclass
class LoadReport:
    """Per-file diagnostics captured during ingestion, surfaced in the EDA tab."""
    provider: str
    company: str
    rows_raw: int = 0
    rows_after_clean: int = 0
    removed_rows: int = 0
    notes: list = field(default_factory=list)


class DataBillContainer:
    """
    Holds raw + cleaned billing data for all 5 uploaded files and produces
    every aggregation the dashboard's EDA / ML tabs need.
    """

    REQUIRED_FILES = ["pea_bfkt", "pea_tuc", "mea_bfkt", "mea_tuc", "mea_tmv"]

    def __init__(self):
        self.raw_frames: dict[str, pd.DataFrame] = {}
        self.load_reports: dict[str, LoadReport] = {}
        self.master_df: Optional[pd.DataFrame] = None
        self._loaded_keys: set[str] = set()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def load_files(self, files: dict[str, FileLike]) -> None:
        if "pea_bfkt" in files:
            t0=time.time(); self._load_pea("BFKT", files["pea_bfkt"]); logger.info(f"pea_bfkt load={time.time()-t0:.1f}s")
        if "pea_tuc" in files:
            t0=time.time(); self._load_pea("TUC", files["pea_tuc"]); logger.info(f"pea_tuc load={time.time()-t0:.1f}s")
        if "mea_bfkt" in files:
            t0=time.time(); self._load_mea("BFKT", files["mea_bfkt"]); logger.info(f"mea_bfkt load={time.time()-t0:.1f}s")
        if "mea_tuc" in files:
            t0=time.time(); self._load_mea("TUC", files["mea_tuc"]); logger.info(f"mea_tuc load={time.time()-t0:.1f}s")
        if "mea_tmv" in files:
            t0=time.time(); self._load_mea("TMV", files["mea_tmv"]); logger.info(f"mea_tmv load={time.time()-t0:.1f}s")

    def is_ready(self) -> bool:
        return self.master_df is not None and len(self.master_df) > 0

    def missing_files(self) -> list[str]:
        return [k for k in self.REQUIRED_FILES if k not in self._loaded_keys]

    # ------------------------- PEA loader ------------------------------

    def _load_pea(self, company: str, file: FileLike) -> None:
        key = f"pea_{company.lower()}"
        report = LoadReport(provider="PEA", company=company)

        raw = _read_any(file, header=0)
        # first row holds the "real" column names in the original export
        raw = raw.rename(columns=raw.iloc[0]).iloc[1:].reset_index(drop=True)
        report.rows_raw = len(raw)

        # normalise the site-id / meter column names across BFKT vs TUC exports
        rename_map = {}
        for c in raw.columns:
            cs = str(c)
            if 'หมายเลขผู้ใช้ไฟฟ้า' in cs:
                rename_map[c] = 'user_number'
        raw = raw.rename(columns=rename_map)
        raw.columns = [_fix_numeric_col(c) for c in raw.columns]

        if 'Site_ID' in raw.columns:
            raw['Site_ID'] = (
                raw['Site_ID'].astype(str).str.strip().str.upper()
                .replace({'0.0': '0', 'NAN': '0', 'NONE': '0', '': '0'})
            )
        if 'Meter_No.' in raw.columns and MEA_METER_COL not in raw.columns:
            raw = raw.rename(columns={'Meter_No.': MEA_METER_COL})

        # Split the two numeric blocks (amount block, then unit block) using the
        # two 'avg' marker columns present in every PEA export.
        cols = raw.columns.tolist()
        avg_idx = [i for i, c in enumerate(cols) if str(c).strip().lower() == 'avg']
        new_cols = []
        if len(avg_idx) >= 2:
            split_idx = avg_idx[1]
            for i, col in enumerate(cols):
                if i < avg_idx[0]:
                    new_cols.append(col)
                elif i < split_idx:
                    new_cols.append(f"{col}_amount" if str(col).isdigit() else
                                     ("avg_amount" if str(col).strip().lower() == 'avg' else col))
                else:
                    new_cols.append(f"{col}_unit" if str(col).isdigit() else
                                     ("avg_unit" if str(col).strip().lower() == 'avg' else col))
            raw.columns = new_cols
        else:
            report.notes.append("Could not find both 'avg' marker columns — "
                                 "amount/unit split may be unreliable.")

        # drop stray duplicate unit column some exports carry (e.g. '256905_unit')
        stray = [c for c in raw.columns if str(c).endswith('_unit')
                 and raw[c].isna().all()]
        if stray:
            raw = raw.drop(columns=stray)

        amount_cols = [c for c in raw.columns if str(c).endswith('_amount') and c != 'avg_amount']
        unit_cols = [c for c in raw.columns if str(c).endswith('_unit') and c != 'avg_unit']

        # drop rows whose most-recent 3 unit columns are all zero/NaN (deceased sites)
        recent_unit_cols = sorted(unit_cols, key=lambda c: int(c.split('_')[0]))[-3:]
        if recent_unit_cols:
            mask_bad = raw[recent_unit_cols].apply(
                lambda x: pd.to_numeric(x, errors='coerce').fillna(0) == 0
            ).all(axis=1)
            before = len(raw)
            raw = raw[~mask_bad].copy()
            report.notes.append(f"Dropped {before - len(raw)} rows with 0 usage "
                                 f"in last 3 known months ({recent_unit_cols}).")

        raw[amount_cols + unit_cols] = (
            raw[amount_cols + unit_cols]
            .apply(_clean_numeric_string)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )

        type_col = next((c for c in PEA_SITE_TYPE_COL_CANDIDATES if c in raw.columns), None)
        if type_col:
            raw[type_col] = raw[type_col].fillna("0").astype(str).str.strip().replace({"0.0": "0"})
            raw = raw.rename(columns={type_col: 'site_type'})
        else:
            raw['site_type'] = "0"

        # drop obviously invalid / decommissioned site-type categories
        drop_types = {"Decom", "DECOM", "RMSC", "MSC", "PN"}
        before = len(raw)
        raw = raw[~raw['site_type'].isin(drop_types)]
        report.notes.append(f"Dropped {before - len(raw)} rows with invalid site_type "
                             f"({sorted(drop_types)}).")

        # drop Site_ID == '0' (no site attached to meter)
        if 'Site_ID' in raw.columns:
            before = len(raw)
            raw = raw[raw['Site_ID'] != '0']
            report.notes.append(f"Dropped {before - len(raw)} rows with Site_ID == '0'.")

        raw['site_type'] = raw['site_type'].replace('0', 'NORMAL')
        raw['company'] = company
        raw['provider'] = 'PEA'

        report.rows_after_clean = len(raw)
        report.removed_rows = report.rows_raw - report.rows_after_clean
        self.raw_frames[key] = raw
        self.load_reports[key] = report
        self._loaded_keys.add(key)

    # ------------------------- MEA loader ------------------------------

    def _load_mea(self, company: str, file: FileLike) -> None:
        key = f"mea_{company.lower()}"
        report = LoadReport(provider="MEA", company=company)

        raw = _read_any(file, header=1)  # real header is row index 1 in MEA exports
        report.rows_raw = len(raw)

        # drop the trailing summary row(s) — Meter_No must be numeric there
        raw = raw[pd.to_numeric(raw[MEA_METER_COL], errors='coerce').notna()]
        raw[MEA_METER_COL] = pd.to_numeric(raw[MEA_METER_COL], errors='coerce')
        raw = raw.dropna(subset=[MEA_METER_COL])

        site_col = next((c for c in MEA_SITE_COL_CANDIDATES if c in raw.columns), None)
        raw['site_type'] = raw[site_col].apply(_normalise_mea_site_type) if site_col else 'NORMAL'

        before = len(raw)
        raw = raw[raw['site_type'] != 'MSC']
        report.notes.append(f"Dropped {before - len(raw)} MSC rows.")

        amt_cols, unit_cols = self._mea_month_columns(raw)

        # drop ghost rows (every monthly amount is NaN) then dedupe by meter,
        # keeping the row with the most non-NaN months ("first" tie-break)
        ghost_mask = raw[amt_cols].isna().all(axis=1) if amt_cols else pd.Series(False, index=raw.index)
        before = len(raw)
        raw = raw[~ghost_mask].copy()
        report.notes.append(f"Dropped {before - len(raw)} ghost rows (all-NaN monthly amounts).")

        if raw.duplicated(MEA_METER_COL).any():
            raw['_non_nan'] = raw[amt_cols].notna().sum(axis=1) if amt_cols else 0
            before = len(raw)
            raw = (raw.sort_values('_non_nan', ascending=False)
                      .drop_duplicates(subset=MEA_METER_COL, keep='first')
                      .drop(columns='_non_nan'))
            report.notes.append(f"Deduplicated {before - len(raw)} repeated Meter_No rows "
                                 f"(kept richest record, method='first').")

        raw['company'] = company
        raw['provider'] = 'MEA'

        report.rows_after_clean = len(raw)
        report.removed_rows = report.rows_raw - report.rows_after_clean
        self.raw_frames[key] = raw
        self.load_reports[key] = report
        self._loaded_keys.add(key)

    @staticmethod
    def _mea_month_columns(df: pd.DataFrame):
        """
        Split the monthly columns into (amount_cols, unit_cols).

        The original Excel exports let us tell the two blocks apart by dtype
        (Excel keeps a bare numeric header like 201901 as an int for the
        amount block, while the duplicate-named unit-block header gets
        pandas' '.1' mangling and becomes a string). CSV uploads lose that
        distinction — every header is a plain string — so we fall back to
        position: the amount block and unit block are each internally
        chronological, so the point where the running max month resets is
        the block boundary.
        """
        all_cols = df.columns.tolist()

        amt_cols = [c for c in all_cols
                    if isinstance(c, (int, np.integer)) and 190001 <= int(c) <= 209912]
        unit_cols = [c for c in all_cols
                     if isinstance(c, str) and re.match(r'^\d{6}(\.\d+)?$', c.strip())]
        if amt_cols:
            return amt_cols, unit_cols

        # --- CSV fallback: positional split ---
        candidates = []
        for c in all_cols:
            base = str(c).split('.')[0].strip()
            if re.match(r'^\d{6}$', base) and 190001 <= int(base) <= 209912:
                candidates.append(c)
        if not candidates:
            return [], []

        first_month = int(str(candidates[0]).split('.')[0])
        boundary = len(candidates)
        running_max = -1
        for i, c in enumerate(candidates):
            m = int(str(c).split('.')[0])
            if i > 0 and m <= running_max and m == first_month:
                boundary = i
                break
            running_max = max(running_max, m)

        return candidates[:boundary], candidates[boundary:]

    # ------------------------------------------------------------------
    # Build the combined long master table
    # ------------------------------------------------------------------

    def build_master(self) -> pd.DataFrame:
        """Melt every loaded raw frame to long format and stack into one table."""
        long_frames = []

        for key, raw in self.raw_frames.items():
            provider = raw['provider'].iloc[0]
            company = raw['company'].iloc[0]

            if provider == 'PEA':
                long_frames.append(self._melt_pea(raw))
            else:
                long_frames.append(self._melt_mea(raw))

        if not long_frames:
            self.master_df = pd.DataFrame()
            return self.master_df

        master = pd.concat(long_frames, ignore_index=True, sort=False)
        master['Site_ID'] = master['Site_ID'].astype(str).str.strip().str.upper()
        master['bill_class'] = master['bill_amount'].apply(classify_bill)
        master['date'] = pd.to_datetime(master['month'].astype(int).astype(str), format='%Y%m')
        master = master.sort_values(['provider', 'company', 'Site_ID', 'date']).reset_index(drop=True)

        self.master_df = master
        return master

    def _melt_pea(self, raw: pd.DataFrame) -> pd.DataFrame:
        candidate_id_cols = ['Meter_No', MEA_METER_COL, 'Site_ID', 'site_type',
                              'RATE_CAT', 'Rate_CAT', 'TOU&TOD', 'Province',
                              'company', 'provider']
        seen = set()
        id_cols = []
        for c in candidate_id_cols:
            if c in raw.columns and c not in seen:
                id_cols.append(c)
                seen.add(c)
        amount_cols = [c for c in raw.columns if str(c).endswith('_amount') and c != 'avg_amount']
        unit_cols = [c for c in raw.columns if str(c).endswith('_unit') and c != 'avg_unit']

        amt_long = raw.melt(id_vars=id_cols, value_vars=amount_cols,
                             var_name='month_raw', value_name='bill_amount')
        amt_long['month_key'] = amt_long['month_raw'].str.replace('_amount', '', regex=False)

        unit_long = raw.melt(id_vars=id_cols, value_vars=unit_cols,
                              var_name='month_raw', value_name='kwh')
        unit_long['month_key'] = unit_long['month_raw'].str.replace('_unit', '', regex=False)

        merged = amt_long.merge(
            unit_long[id_cols + ['month_key', 'kwh']],
            on=id_cols + ['month_key'], how='left'
        )
        merged = merged.drop(columns=['month_raw'])

        # month_key is Buddhist-era YYYYMM (e.g. 256902) -> convert to Gregorian
        be_year = merged['month_key'].str[:4].astype(int)
        mm = merged['month_key'].str[4:6].astype(int)
        merged['month'] = (be_year - 543) * 100 + mm
        merged = merged.drop(columns=['month_key'])

        merged = merged.rename(columns={'RATE_CAT': 'Rate_CAT', 'TOU&TOD': 'TOU_TOD'})
        if 'TOU_TOD' not in merged.columns:
            merged['TOU_TOD'] = np.nan
        if 'Rate_CAT' not in merged.columns:
            merged['Rate_CAT'] = np.nan
        if 'Province' not in merged.columns:
            merged['Province'] = np.nan
        if MEA_METER_COL not in merged.columns and 'Meter_No' in merged.columns:
            merged = merged.rename(columns={'Meter_No': MEA_METER_COL})

        merged['is_shutdown'] = False  # PEA pipeline does not compute shutdown flag
        keep = [MEA_METER_COL, 'Site_ID', 'site_type', 'Rate_CAT', 'TOU_TOD',
                'Province', 'month', 'bill_amount', 'kwh', 'company', 'provider', 'is_shutdown']
        keep = [c for c in keep if c in merged.columns]
        return merged[keep]

    def _melt_mea(self, raw: pd.DataFrame) -> pd.DataFrame:
        amt_cols, unit_cols = self._mea_month_columns(raw)
        logger.info(f"[{raw['company'].iloc[0]}] amt_cols={len(amt_cols)} unit_cols={len(unit_cols)}")
        logger.info(f"[{raw['company'].iloc[0]}] sample amt_cols: {amt_cols[:5]}")
        id_cols = [c for c in [MEA_METER_COL, 'Site_ID', 'company', 'provider', 'site_type',
                                'Rate_CAT', 'TOU&TOD', 'Province', 'Input_Date', 'Remark']
                   if c in raw.columns]

        melted_amt = raw[id_cols + amt_cols].melt(
            id_vars=id_cols, value_vars=amt_cols, var_name='month_raw', value_name='bill_amount')
        melted_amt['month'] = melted_amt['month_raw'].astype(int)
        melted_amt = melted_amt.drop(columns=['month_raw'])

        if unit_cols:
            melted_kwh = raw[id_cols + unit_cols].melt(
                id_vars=id_cols, value_vars=unit_cols, var_name='month_raw', value_name='kwh')
            melted_kwh['month'] = melted_kwh['month_raw'].apply(lambda c: int(str(c).strip()[:6]))
            melted_kwh = melted_kwh.drop(columns=['month_raw'])
            merged = melted_amt.merge(melted_kwh, on=id_cols + ['month'], how='left')
        else:
            merged = melted_amt.copy()
            merged['kwh'] = np.nan

        merged[MEA_METER_COL] = merged[MEA_METER_COL].astype('int64').astype(str)
        merged['bill_amount'] = pd.to_numeric(merged['bill_amount'], errors='coerce')
        merged['kwh'] = pd.to_numeric(merged['kwh'], errors='coerce')

        # --- fill spine + shutdown detection (ported from MEA_cleaning.ipynb) ---
        merged = merged.sort_values([MEA_METER_COL, 'company', 'month']).reset_index(drop=True)

        all_months = sorted(merged['month'].unique())
        keys = merged[[MEA_METER_COL, 'company']].drop_duplicates()
        logger.info(f"[{raw['company'].iloc[0]}] unique_meters={len(keys)} unique_months={len(all_months)} "
                    f"spine_size={len(keys)*len(all_months)}")
        spine = keys.merge(pd.DataFrame({'month': all_months}), how='cross')
        merged = spine.merge(merged, on=[MEA_METER_COL, 'company', 'month'], how='left')

        meta_cols = [c for c in ['Site_ID', 'site_type', 'Rate_CAT', 'TOU&TOD',
                                  'Province', 'Input_Date', 'Remark'] if c in merged.columns]
        if meta_cols:
            merged[meta_cols] = merged.groupby([MEA_METER_COL, 'company'])[meta_cols].ffill().bfill()
        merged['provider'] = 'MEA'

        merged['bill_amount'] = merged['bill_amount'].fillna(0).round(2)
        merged['kwh'] = merged['kwh'].fillna(0).round(2)

        bill_class = merged['bill_amount'].apply(classify_bill)
        is_zero = (bill_class == 'zero').astype(int)

        def trailing_zero_run(s: pd.Series) -> pd.Series:
            return s[::-1].cummin()[::-1]

        merged['_is_zero'] = is_zero
        trailing = merged.groupby([MEA_METER_COL, 'company'])['_is_zero'].transform(trailing_zero_run)
        trailing_len = merged.groupby([MEA_METER_COL, 'company'])['trailing_zero' if False else '_is_zero'] \
            .transform(lambda s: trailing_zero_run(s).sum())
        immediate = merged['site_type'] == 'DECOM' if 'site_type' in merged.columns else False
        consecutive_shutdown = (trailing == 1) & (trailing_len >= 3)
        merged['is_shutdown'] = (immediate | consecutive_shutdown) if 'site_type' in merged.columns else consecutive_shutdown
        merged.loc[merged[MEA_METER_COL].astype(str) == '0', 'is_shutdown'] = True
        merged = merged.drop(columns=['_is_zero'])

        merged = merged.rename(columns={'TOU&TOD': 'TOU_TOD'})
        keep = [MEA_METER_COL, 'Site_ID', 'site_type', 'Rate_CAT', 'TOU_TOD',
                'Province', 'month', 'bill_amount', 'kwh', 'company', 'provider', 'is_shutdown']
        keep = [c for c in keep if c in merged.columns]
        return merged[keep]

    # ------------------------------------------------------------------
    # EDA
    # ------------------------------------------------------------------

    def _df(self) -> pd.DataFrame:
        if self.master_df is None:
            raise RuntimeError("Call build_master() after load_files() first.")
        return self.master_df

    def eda_bill_range(self) -> dict:
        df = self._df()
        months = sorted(df['month'].unique())
        return {
            "min_month": int(months[0]) if months else None,
            "max_month": int(months[-1]) if months else None,
            "n_months": len(months),
            "per_provider": {
                p: {
                    "min_month": int(g['month'].min()),
                    "max_month": int(g['month'].max()),
                    "n_months": g['month'].nunique(),
                }
                for p, g in df.groupby('provider')
            },
        }

    def eda_duplicates(self) -> dict:
        """
        Duplicate site names across the 5 raw files (metadata level, one row
        per meter per file, BEFORE the melt). We report:
          - malformed_site_ids: Site_IDs that don't match the CBR4017 / CBR4017A pattern
          - duplicate_site_ids: any Site_ID appearing on more than one raw row
            (across the uploaded files) — this is what the eventual
            groupby(...).first() merge will silently collapse.
        """
        raws = [df[['Site_ID', 'company', 'provider']] for df in self.raw_frames.values()
                if 'Site_ID' in df.columns]
        if not raws:
            return {"malformed_site_ids": [], "duplicate_site_ids": []}
        combined = pd.concat(raws, ignore_index=True)
        combined['Site_ID'] = combined['Site_ID'].astype(str).str.strip().str.upper()

        malformed = sorted(combined.loc[~combined['Site_ID'].str.match(SITE_ID_RE), 'Site_ID'].unique().tolist())

        dup_mask = combined.duplicated('Site_ID', keep=False)
        dup_detail = (
            combined[dup_mask]
            .groupby('Site_ID')
            .agg(occurrences=('Site_ID', 'size'),
                 providers=('provider', lambda s: sorted(set(s))),
                 companies=('company', lambda s: sorted(set(s))))
            .reset_index()
            .sort_values('occurrences', ascending=False)
        )

        return {
            "malformed_site_ids": malformed,
            "malformed_count": len(malformed),
            "duplicate_site_ids": dup_detail.to_dict(orient='records'),
            "duplicate_count": len(dup_detail),
        }

    def eda_common_sites(self) -> dict:
        pea_sites = {c: set(df['Site_ID'].astype(str).str.upper())
                     for c, df in self.raw_frames.items() if c.startswith('pea_')}
        mea_sites = {c: set(df['Site_ID'].astype(str).str.upper())
                     for c, df in self.raw_frames.items() if c.startswith('mea_') and 'Site_ID' in df.columns}

        def pairwise_common(d: dict):
            out = {}
            keys = list(d.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    common = sorted(d[keys[i]] & d[keys[j]])
                    out[f"{keys[i]}__{keys[j]}"] = {"count": len(common), "site_ids": common}
            return out

        pea_all = set().union(*pea_sites.values()) if pea_sites else set()
        mea_all = set().union(*mea_sites.values()) if mea_sites else set()
        cross_common = sorted(pea_all & mea_all)

        result = {
            "within_pea": pairwise_common(pea_sites),
            "within_mea": pairwise_common(mea_sites),
            "pea_mea_cross_common": {"count": len(cross_common), "site_ids": cross_common},
        }
        if len(mea_sites) == 3:
            all3 = set.intersection(*mea_sites.values())
            result["mea_all_three_common"] = {"count": len(all3), "site_ids": sorted(all3)}
        return result

    def eda_site_types(self) -> dict:
        df = self._df()
        out = {}
        for provider, g in df.groupby('provider'):
            counts = (g.drop_duplicates([MEA_METER_COL, 'company', 'Site_ID'])['site_type']
                        .value_counts().to_dict())
            out[provider] = counts
        return out

    def eda_last_month_missing(self, windows=LAST_MONTH_WINDOWS) -> dict:
        """
        For each window N in `windows`, find sites whose last N months are ALL
        missing/zero kwh — i.e. likely shut-down / stopped-reporting stations.
        """
        df = self._df()
        result = {}
        for provider, g in df.groupby('provider'):
            months = sorted(g['month'].unique())
            per_window = {}
            for n in windows:
                last_n = months[-n:] if len(months) >= n else months
                sub = g[g['month'].isin(last_n)]
                pivot = sub.pivot_table(index='Site_ID', columns='month', values='kwh', aggfunc='first')
                pivot = pivot.reindex(columns=last_n)
                missing_mask = pivot.isna() | (pivot == 0)
                all_missing = missing_mask.all(axis=1)
                sites = all_missing[all_missing].index.tolist()
                per_window[n] = {"months_checked": [int(m) for m in last_n],
                                  "count": len(sites), "site_ids": sites}
            result[provider] = per_window
        return result

    def eda_maintenance_sites(self, months_window: int = 6) -> dict:
        """
        Bill-amount value_counts restricted to the 'maintenance' bucket
        (0 < bill_amount < 200), plus which sites are currently in maintenance
        (i.e. had a maintenance-range bill in the last N months) with their
        provider (PEA/MEA) and company.
        """
        df = self._df()
        maint = df[df['bill_class'] == 'maintenance']

        vc = maint['bill_amount'].round(2).value_counts().sort_index()

        cutoff = df['date'].max() - pd.DateOffset(months=months_window)
        recent_maint = maint[maint['date'] >= cutoff]

        site_rows = (
            recent_maint
            .sort_values('date')
            .drop_duplicates(subset=['Site_ID', 'provider', 'company'], keep='last')
            [['Site_ID', 'provider', 'company', 'site_type', 'bill_amount', 'date']]
            .rename(columns={'date': 'last_maintenance_month'})
            .sort_values(['provider', 'company', 'Site_ID'])
        )

        sites = [
            {
                "site_id": r.Site_ID,
                "provider": r.provider,   # PEA or MEA
                "company": r.company,     # BFKT / TUC / TMV
                "site_type": r.site_type,
                "bill_amount": float(r.bill_amount),
                "last_maintenance_month": r.last_maintenance_month.strftime("%Y-%m"),
            }
            for r in site_rows.itertuples(index=False)
        ]

        return {
            "total_maintenance_rows": int(len(maint)),
            "unique_amounts": int(vc.shape[0]),
            "value_counts": [{"amount": float(k), "count": int(v)} for k, v in vc.items()],
            "maintenance_sites_last_{}_months".format(months_window): sites,
            "maintenance_site_count": len(sites),
        }

    def eda_error_rates(self) -> dict:
        """Sanity-check / error-rate summary a business user can hand to the billing vendor."""
        df = self._df()
        total = len(df)
        zero_bill = int((df['bill_amount'] == 0).sum())
        mismatch_bill_no_kwh = int(((df['bill_amount'] > 0) & (df['kwh'].fillna(0) == 0)).sum())
        mismatch_kwh_no_bill = int(((df['kwh'].fillna(0) > 0) & (df['bill_amount'] == 0)).sum())
        missing_kwh = int(df['kwh'].isna().sum())
        negative_values = int(((df['bill_amount'] < 0) | (df['kwh'].fillna(0) < 0)).sum())

        return {
            "total_rows": total,
            "zero_bill_rate": round(zero_bill / total, 4) if total else 0,
            "bill_without_kwh_rate": round(mismatch_bill_no_kwh / total, 4) if total else 0,
            "kwh_without_bill_rate": round(mismatch_kwh_no_bill / total, 4) if total else 0,
            "missing_kwh_rate": round(missing_kwh / total, 4) if total else 0,
            "negative_value_rows": negative_values,
            "load_reports": {k: vars(v) for k, v in self.load_reports.items()},
        }

    def eda_summary(self) -> dict:
        return {
            "bill_range": self.eda_bill_range(),
            "duplicates": self.eda_duplicates(),
            "common_sites": self.eda_common_sites(),
            "site_types": self.eda_site_types(),
            "last_month_missing": self.eda_last_month_missing(),
            "maintenance_sites": self.eda_maintenance_sites(),
            "error_rates": self.eda_error_rates(),
        }

    # ------------------------------------------------------------------
    # Site trend query
    # ------------------------------------------------------------------

    def get_site_trend(self, site_id: str, metric: str = "kwh",
                        start_month: Optional[int] = None,
                        end_month: Optional[int] = None) -> dict:
        if metric not in ("kwh", "bill_amount"):
            raise ValueError("metric must be 'kwh' or 'bill_amount'")
        df = self._df()
        sub = df[df['Site_ID'].astype(str).str.upper() == str(site_id).upper()].copy()
        if start_month:
            sub = sub[sub['month'] >= int(start_month)]
        if end_month:
            sub = sub[sub['month'] <= int(end_month)]
        sub = sub.sort_values('month')
        if sub.empty:
            return {"site_id": site_id, "found": False, "series": []}
        return {
            "site_id": site_id,
            "found": True,
            "provider": sub['provider'].iloc[0],
            "company": sub['company'].iloc[0],
            "site_type": sub['site_type'].iloc[0] if 'site_type' in sub.columns else None,
            "metric": metric,
            "series": [
                {"month": int(row['month']),
                 "value": None if pd.isna(row[metric]) else float(row[metric])}
                for _, row in sub.iterrows()
            ],
        }

    def list_site_ids(self, provider: Optional[str] = None) -> list[str]:
        df = self._df()
        if provider:
            df = df[df['provider'] == provider]
        return sorted(df['Site_ID'].dropna().astype(str).unique().tolist())