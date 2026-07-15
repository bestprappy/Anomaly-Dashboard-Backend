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

import csv
import gc
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

# Export tools use several spellings for the two identifier headers.  Match
# them by a separator-insensitive key and keep one canonical spelling inside
# the ingestion pipeline.
IDENTIFIER_COLUMN_ALIASES = {
    'siteid': MEA_SITE_ID,
    'meterno': MEA_METER_COL,
    'meternumber': MEA_METER_COL,
}

PEA_SITE_TYPE_COL_CANDIDATES = ['MSC/RMSC/IBC/WIFI/DeCom', 'MSC/RMSC/IBC/WIFI/DN/PN']

LAST_MONTH_WINDOWS = (3, 6, 9)

# anomalous bill patterns surfaced by eda_meter_patterns, most severe first
METER_PATTERN_ORDER = ('shutdown', 'gap')


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


def _identifier_column_key(column) -> str:
    """Return a stable key for common identifier-header variations."""
    text = str(column).replace('\ufeff', '').replace('\u00a0', ' ').strip().casefold()
    return re.sub(r'[^a-z0-9]+', '', text)


def _canonicalise_identifier_columns(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Rename known Site_ID/Meter_No variants, rejecting ambiguous inputs."""
    matches: dict[str, list[object]] = {}
    for column in df.columns:
        canonical = IDENTIFIER_COLUMN_ALIASES.get(_identifier_column_key(column))
        if canonical is not None:
            matches.setdefault(canonical, []).append(column)

    ambiguous = {name: columns for name, columns in matches.items() if len(columns) > 1}
    if ambiguous:
        details = '; '.join(
            f"{name}: {', '.join(repr(str(column)) for column in columns)}"
            for name, columns in ambiguous.items()
        )
        raise ValueError(f"{source} file has ambiguous identifier columns ({details}).")

    rename_map = {
        columns[0]: canonical
        for canonical, columns in matches.items()
        if columns[0] != canonical
    }
    return df.rename(columns=rename_map)


def _require_columns(df: pd.DataFrame, required: tuple[str, ...], source: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if not missing:
        return

    raise ValueError(
        f"{source} file is missing required column(s): {', '.join(missing)}. "
        "Accepted Site_ID headers include Site_ID, Site ID, and SiteID; "
        "accepted Meter_No headers include Meter_No, Meter No, and Meter Number."
    )


def _drop_missing_site_ids(
    df: pd.DataFrame,
    report: "LoadReport",
    source: str,
) -> pd.DataFrame:
    """Normalize Site_ID values and remove rows that have no usable ID."""
    site_ids = (
        df[MEA_SITE_ID]
        .astype('string')
        .str.replace('\u00a0', ' ', regex=False)
        .str.strip()
        .str.upper()
    )
    missing = site_ids.isna() | site_ids.isin({'', '0', '0.0', 'NAN', 'NONE', 'NULL'})
    removed = int(missing.sum())

    cleaned = df.loc[~missing].copy()
    cleaned[MEA_SITE_ID] = site_ids.loc[~missing].astype(str)
    if removed:
        report.notes.append(f"Dropped {removed} rows with a missing Site_ID.")
    if cleaned.empty:
        raise ValueError(f"{source} file has no rows with a usable Site_ID.")
    return cleaned

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


def _meter_str(val) -> Optional[str]:
    """Meter number as a clean display string ('112.0' -> '112'), or None."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    return re.sub(r'\.0+$', '', s) or None


def classify_bill(amount) -> str:
    if pd.isna(amount) or amount == 0:
        return 'zero'
    if 0 < amount < 200:
        return 'maintenance'
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
        self.long_frames: dict[str, pd.DataFrame] = {}
        self.site_frames: dict[str, pd.DataFrame] = {}
        self.load_reports: dict[str, LoadReport] = {}
        self.master_df: Optional[pd.DataFrame] = None
        self._loaded_keys: set[str] = set()
        self.dropped_latest_month: Optional[int] = None
        # most-recent eda_meter_patterns result, keyed by window; cleared on rebuild
        self._meter_patterns_cache: dict[int, dict] = {}

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def load_files(self, files: dict[str, FileLike]) -> None:
        if files:
            self.master_df = None
            gc.collect()

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

    def has_loaded_data(self) -> bool:
        return bool(self.long_frames)

    def is_ready(self) -> bool:
        return self.has_loaded_data() and not self.missing_files()

    def loaded_files(self) -> list[str]:
        return sorted(self._loaded_keys)

    def missing_files(self) -> list[str]:
        return [k for k in self.REQUIRED_FILES if k not in self._loaded_keys]

    def rows_total(self) -> int:
        return len(self.master_df) if self.master_df is not None else 0

    # ------------------------- PEA loader ------------------------------

    def _load_pea(self, company: str, file: FileLike) -> None:
        key = f"pea_{company.lower()}"
        report = LoadReport(provider="PEA", company=company)

        raw = _read_any(file, header=0)
        if raw.empty:
            raise ValueError(f"PEA {company} file has no data rows.")
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
        source = f"PEA {company}"
        raw = _canonicalise_identifier_columns(raw, source)
        _require_columns(raw, (MEA_SITE_ID,), source)
        raw = _drop_missing_site_ids(raw, report, source)

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

        # Prune to just the columns the melt will use — the raw export carries
        # banner/summary columns that would otherwise ride along through every
        # copy below (the API lives on a 512 MB instance; prune early).
        meta_keep = [c for c in ('Meter_No', MEA_METER_COL, 'Site_ID',
                                  'RATE_CAT', 'Rate_CAT', 'TOU&TOD', 'Province',
                                  *PEA_SITE_TYPE_COL_CANDIDATES) if c in raw.columns]
        raw = raw[list(dict.fromkeys(meta_keep + amount_cols + unit_cols))]

        # Convert the numeric block column-by-column into float32. The old
        # whole-block `.apply(_clean_numeric_string)` briefly duplicated the
        # entire numeric block as strings *and* left it float64 — both fatal
        # at 512 MB. Collecting into a dict then concatenating once also
        # avoids block-manager fragmentation.
        converted = {}
        for col in amount_cols + unit_cols:
            s = raw[col]
            if s.dtype == object:
                s = pd.to_numeric(_clean_numeric_string(s), errors="coerce")
            else:
                s = pd.to_numeric(s, errors="coerce")
            converted[col] = s.fillna(0.0).astype(np.float32)
        raw = pd.concat(
            [raw.drop(columns=amount_cols + unit_cols),
             pd.DataFrame(converted, index=raw.index)],
            axis=1,
        )
        del converted
        gc.collect()

        # drop rows whose most-recent 3 unit columns are all zero/NaN (deceased
        # sites) — NaNs are already 0.0 after the conversion above
        recent_unit_cols = sorted(unit_cols, key=lambda c: int(c.split('_')[0]))[-3:]
        if recent_unit_cols:
            mask_bad = raw[recent_unit_cols].eq(0).all(axis=1)
            before = len(raw)
            raw = raw[~mask_bad].copy()
            report.notes.append(f"Dropped {before - len(raw)} rows with 0 usage "
                                 f"in last 3 known months ({recent_unit_cols}).")

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

        raw['site_type'] = raw['site_type'].replace('0', 'NORMAL')
        raw['company'] = company
        raw['provider'] = 'PEA'

        report.rows_after_clean = len(raw)
        report.removed_rows = report.rows_raw - report.rows_after_clean
        self.site_frames[key] = raw[[MEA_SITE_ID, 'company', 'provider']].copy()
        # Melt now and discard the wide raw frame — it is by far the biggest
        # object in memory and nothing downstream needs it after this point.
        self.long_frames[key] = self._melt_pea(raw)
        del raw
        gc.collect()
        self.load_reports[key] = report
        self._loaded_keys.add(key)

    # ------------------------- MEA loader ------------------------------

    def _load_mea(self, company: str, file: FileLike) -> None:
        key = f"mea_{company.lower()}"
        report = LoadReport(provider="MEA", company=company)

        raw = _read_any(file, header=1)  # real header is row index 1 in MEA exports
        report.rows_raw = len(raw)
        source = f"MEA {company}"
        raw = _canonicalise_identifier_columns(raw, source)
        _require_columns(raw, (MEA_METER_COL, MEA_SITE_ID), source)

        # drop the trailing summary row(s) — Meter_No must be numeric there
        raw = raw[pd.to_numeric(raw[MEA_METER_COL], errors='coerce').notna()]
        raw[MEA_METER_COL] = pd.to_numeric(raw[MEA_METER_COL], errors='coerce')
        raw = raw.dropna(subset=[MEA_METER_COL])
        raw = _drop_missing_site_ids(raw, report, source)

        site_col = next((c for c in MEA_SITE_COL_CANDIDATES if c in raw.columns), None)
        raw['site_type'] = raw[site_col].apply(_normalise_mea_site_type) if site_col else 'NORMAL'

        before = len(raw)
        raw = raw[raw['site_type'] != 'MSC']
        report.notes.append(f"Dropped {before - len(raw)} MSC rows.")

        amt_cols, unit_cols = self._mea_month_columns(raw)

        # Prune to the columns the melt keeps, then convert the monthly block
        # to float32 column-by-column *before* melting (512 MB instance: the
        # old post-melt to_numeric carried the whole block through the melt as
        # objects). No fillna here — the ghost/dedupe steps below rely on NaN.
        meta_keep = [c for c in (MEA_METER_COL, 'Site_ID', 'site_type', 'Rate_CAT',
                                  'TOU&TOD', 'Province', 'Input_Date', 'Remark')
                     if c in raw.columns]
        raw = raw[list(dict.fromkeys(meta_keep + amt_cols + unit_cols))]

        converted = {}
        for col in amt_cols + unit_cols:
            converted[col] = pd.to_numeric(raw[col], errors='coerce').astype(np.float32)
        raw = pd.concat(
            [raw.drop(columns=amt_cols + unit_cols),
             pd.DataFrame(converted, index=raw.index)],
            axis=1,
        )
        del converted
        gc.collect()

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
        self.site_frames[key] = raw[[MEA_SITE_ID, 'company', 'provider']].copy()
        self.long_frames[key] = self._melt_mea(raw)
        del raw
        gc.collect()
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

        # each block is internally chronological, so the first month that fails
        # to increase marks the start of the unit block
        boundary = len(candidates)
        running_max = -1
        for i, c in enumerate(candidates):
            m = int(str(c).split('.')[0])
            if i > 0 and m <= running_max:
                boundary = i
                break
            running_max = max(running_max, m)

        return candidates[:boundary], candidates[boundary:]

    # ------------------------------------------------------------------
    # Build the combined long master table
    # ------------------------------------------------------------------

    def _drop_incomplete_latest_month(self, master: pd.DataFrame) -> pd.DataFrame:
        """
        The most recent calendar month in a fresh export is almost always
        mid-billing-cycle: most sites haven't been billed yet, so kwh/bill_amount
        reads as artificially low or zero. That month isn't "real" data yet, so
        we drop it here — once, at the source — rather than let every EDA tab
        and the ML pipeline separately have to work around a half-finished month.
        """
        if master.empty:
            self.dropped_latest_month = None
            return master

        latest_month = int(master['month'].max())
        before = len(master)
        # boolean indexing already returns a new frame; an extra .copy() would
        # briefly hold the master table in memory twice
        trimmed = master[master['month'] < latest_month]
        dropped_rows = before - len(trimmed)

        self.dropped_latest_month = latest_month
        logger.info(
            f"Dropped {dropped_rows} rows from incomplete latest month {latest_month} "
            f"(billing cycle likely still open)."
        )
        return trimmed


    def build_master(self) -> pd.DataFrame:
        self._meter_patterns_cache = {}
        if not self.long_frames:
            self.master_df = pd.DataFrame()
            return self.master_df

        missing_site_id = [
            key for key, frame in self.long_frames.items()
            if MEA_SITE_ID not in frame.columns
        ]
        if missing_site_id:
            raise ValueError(
                "Loaded file(s) are missing the canonical Site_ID column: "
                + ", ".join(sorted(missing_site_id))
            )

        self.master_df = None
        gc.collect()

        master = pd.concat(
            self.long_frames.values(),
            ignore_index=True,
            sort=False,
            copy=False,
        )
        master['Site_ID'] = master['Site_ID'].astype(str).str.strip().str.upper()
        master['bill_class'] = master['bill_amount'].apply(classify_bill)
        master['date'] = pd.to_datetime(master['month'].astype(int).astype(str), format='%Y%m')

        master = self._drop_incomplete_latest_month(master)

        self.master_df = master
        gc.collect()
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

        # Melt on a compact integer row key instead of the ~9 string id
        # columns. Melting duplicates every id column per month, which made
        # the two long frames the biggest transient allocation of the whole
        # upload on the 512 MB instance; the metadata now joins back exactly
        # once at the end.
        raw = raw.copy()
        raw['_row'] = np.arange(len(raw), dtype=np.int32)

        amt_long = raw.melt(id_vars=['_row'], value_vars=amount_cols,
                             var_name='month_raw', value_name='bill_amount')
        amt_long['month_key'] = amt_long['month_raw'].str.replace('_amount', '', regex=False)
        amt_long = amt_long.drop(columns=['month_raw'])

        unit_long = raw.melt(id_vars=['_row'], value_vars=unit_cols,
                              var_name='month_raw', value_name='kwh')
        unit_long['month_key'] = unit_long['month_raw'].str.replace('_unit', '', regex=False)
        unit_long = unit_long.drop(columns=['month_raw'])

        merged = amt_long.merge(unit_long, on=['_row', 'month_key'], how='left')
        del amt_long, unit_long
        gc.collect()

        # month_key is Buddhist-era YYYYMM (e.g. 256902) -> convert to Gregorian.
        # Some exports already use Gregorian years, so only shift years >= 2400.
        year = merged['month_key'].str[:4].astype(int)
        mm = merged['month_key'].str[4:6].astype(int)
        year = np.where(year >= 2400, year - 543, year)
        merged['month'] = (year * 100 + mm).astype(np.int32)
        merged = merged.drop(columns=['month_key'])

        merged = merged.merge(raw[id_cols + ['_row']], on='_row', how='left')
        merged = merged.drop(columns=['_row'])

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

        # Melt on just (meter, company); the remaining metadata joins back
        # exactly once at the end. Dragging every meta column through two
        # melts, a merge and the spine used to multiply the string columns
        # by the month count — the biggest transient of an MEA upload.
        key_cols = [MEA_METER_COL, 'company']
        meta_cols = [c for c in ['Site_ID', 'site_type', 'Rate_CAT', 'TOU&TOD',
                                  'Province', 'Input_Date', 'Remark'] if c in raw.columns]

        melted_amt = raw[key_cols + amt_cols].melt(
            id_vars=key_cols, value_vars=amt_cols, var_name='month_raw', value_name='bill_amount')
        melted_amt['month'] = melted_amt['month_raw'].astype(int).astype(np.int32)
        melted_amt = melted_amt.drop(columns=['month_raw'])

        if unit_cols:
            melted_kwh = raw[key_cols + unit_cols].melt(
                id_vars=key_cols, value_vars=unit_cols, var_name='month_raw', value_name='kwh')
            melted_kwh['month'] = melted_kwh['month_raw'].apply(
                lambda c: int(str(c).strip()[:6])).astype(np.int32)
            melted_kwh = melted_kwh.drop(columns=['month_raw'])
            merged = melted_amt.merge(melted_kwh, on=key_cols + ['month'], how='left')
            del melted_kwh
        else:
            merged = melted_amt.copy()
            merged['kwh'] = pd.Series(np.nan, index=merged.index, dtype='float32')
        del melted_amt
        gc.collect()

        merged[MEA_METER_COL] = merged[MEA_METER_COL].astype('int64').astype(str)

        # --- fill spine + shutdown detection (ported from MEA_cleaning.ipynb) ---
        merged = merged.sort_values(key_cols + ['month']).reset_index(drop=True)

        all_months = sorted(merged['month'].unique())
        keys = merged[key_cols].drop_duplicates()
        logger.info(f"[{raw['company'].iloc[0]}] unique_meters={len(keys)} unique_months={len(all_months)} "
                    f"spine_size={len(keys)*len(all_months)}")
        spine = keys.merge(
            pd.DataFrame({'month': all_months}).astype({'month': 'int32'}), how='cross')
        merged = spine.merge(merged, on=key_cols + ['month'], how='left')
        del spine, keys
        gc.collect()

        # raw is one row per meter after the dedupe in _load_mea, so a single
        # per-meter merge fills the spine-created gap rows with the same
        # values the old per-group ffill().bfill() produced.
        if meta_cols:
            meta = raw[[MEA_METER_COL] + meta_cols].copy()
            meta[MEA_METER_COL] = meta[MEA_METER_COL].astype('int64').astype(str)
            merged = merged.merge(meta, on=MEA_METER_COL, how='left')
            del meta
        merged['provider'] = 'MEA'

        merged['bill_amount'] = merged['bill_amount'].fillna(0).round(2)
        merged['kwh'] = merged['kwh'].fillna(0).round(2)

        bill_class = merged['bill_amount'].apply(classify_bill)
        is_zero = (bill_class == 'zero').astype(int)

        def trailing_zero_run(s: pd.Series) -> pd.Series:
            return s[::-1].cummin()[::-1]

        merged['_is_zero'] = is_zero
        grouped_zero = merged.groupby([MEA_METER_COL, 'company'])['_is_zero']
        trailing = grouped_zero.transform(trailing_zero_run)
        trailing_len = grouped_zero.transform(lambda s: trailing_zero_run(s).sum())
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

    def ensure_master(self) -> pd.DataFrame:
        if self.master_df is None:
            if not self.has_loaded_data():
                raise RuntimeError("Call load_files() first.")
            return self.build_master()
        return self.master_df

    def _df(self) -> pd.DataFrame:
        return self.ensure_master()

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
        raws = list(self.site_frames.values())
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
                     for c, df in self.site_frames.items() if c.startswith('pea_')}
        mea_sites = {c: set(df['Site_ID'].astype(str).str.upper())
                     for c, df in self.site_frames.items() if c.startswith('mea_')}

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
        if MEA_METER_COL not in maint.columns:
            maint = maint.assign(**{MEA_METER_COL: np.nan})

        vc = maint['bill_amount'].round(2).value_counts().sort_index()

        cutoff = df['date'].max() - pd.DateOffset(months=months_window)
        recent_maint = maint[maint['date'] >= cutoff]

        site_rows = (
            recent_maint
            .sort_values('date')
            .drop_duplicates(subset=['Site_ID', 'provider', 'company'], keep='last')
            [['Site_ID', MEA_METER_COL, 'provider', 'company', 'site_type',
              'bill_amount', 'date']]
            .rename(columns={'date': 'last_maintenance_month'})
            .sort_values(['provider', 'company', 'Site_ID'])
        )

        sites = [
            {
                "site_id": r.Site_ID,
                "meter_no": _meter_str(r.Meter_No),
                "provider": r.provider,   # PEA or MEA
                "company": r.company,     # BFKT / TUC / TMV
                "site_type": None if pd.isna(r.site_type) else str(r.site_type),
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

    def _meter_patterns(self, window: int) -> dict:
        """
        Build (and cache) the full per-meter datasheet for the last `window`
        billed months: one row per unique meter with its monthly bill amounts
        and a pattern label:

          shutdown — no bill at all in any of the months (site gone)
          gap      — billed some months, zero/missing the others
          normal   — billed every month

        Rows without a usable meter number fall back to their Site_ID as
        identity and report meter_no = NA. The cache holds only the most
        recent window and is cleared whenever the master table is rebuilt.
        """
        cached = self._meter_patterns_cache.get(window)
        if cached is not None:
            return cached

        df = self._df()
        counts = {p: 0 for p in (*METER_PATTERN_ORDER, 'normal')}
        meta_cols = ['meter_no', 'site_id', 'provider', 'company',
                     'site_type', 'pattern']
        result = {"months": [], "unique_meters": 0,
                  "unique_meters_per_provider": {}, "counts": counts,
                  "counts_per_company": [],
                  "table": pd.DataFrame(columns=meta_cols)}
        if df.empty:
            self._meter_patterns_cache = {window: result}
            return result

        if MEA_METER_COL in df.columns:
            meter = (df[MEA_METER_COL].astype(str).str.strip()
                     .str.replace(r'\.0+$', '', regex=True))
        else:
            meter = pd.Series('', index=df.index)
        bad = meter.isin(('', '0', 'nan', 'NaN', 'NAN', 'None', 'NONE'))
        work = df[['provider', 'company', 'Site_ID', 'site_type',
                   'month', 'bill_amount']].copy()
        work['meter_key'] = meter.mask(bad, 'SITE:' + df['Site_ID'].astype(str))

        distinct = work.drop_duplicates(['provider', 'meter_key'])
        unique_total = int(len(distinct))
        unique_per_provider = {str(p): int(n)
                               for p, n in distinct.groupby('provider').size().items()}

        all_months: set[int] = set()
        frames: list[pd.DataFrame] = []

        # PEA and MEA exports don't necessarily end on the same month, so each
        # provider is classified against its own last-N window; a month a
        # provider never billed stays NaN in the combined table (vs 0 = no bill).
        for provider, g in work.groupby('provider'):
            months = sorted(int(m) for m in g['month'].unique())[-window:]
            all_months.update(months)
            sub = g[g['month'].isin(months)]
            meter_universe = pd.Index(
                g['meter_key'].drop_duplicates(), name='meter_key')

            # max-aggregation both dedupes repeated meter rows and preserves
            # the "was anything billed this month?" signal the classes need
            pivot = sub.pivot_table(index='meter_key', columns='month',
                                    values='bill_amount', aggfunc='max')
            # A company's export can lag another company under the same
            # provider. Keep its known meters in the recent provider window;
            # no recent rows is a real all-zero (shutdown) pattern.
            pivot = (pivot.reindex(index=meter_universe, columns=months)
                     .fillna(0.0).round(2))

            is_zero = pivot.eq(0)
            pattern = pd.Series('normal', index=pivot.index)
            pattern[is_zero.any(axis=1) & (~is_zero).any(axis=1)] = 'gap'
            pattern[is_zero.all(axis=1)] = 'shutdown'

            for name, n in pattern.value_counts().items():
                counts[str(name)] = counts.get(str(name), 0) + int(n)

            meta = (g.sort_values('month')
                    .drop_duplicates('meter_key', keep='last')
                    .set_index('meter_key'))
            idx = pivot.index.to_series().astype(str)
            frame = pd.DataFrame({
                'meter_no': idx.mask(idx.str.startswith('SITE:'), other=pd.NA),
                'site_id': meta['Site_ID'].reindex(pivot.index).astype(str),
                'provider': str(provider),
                'company': meta['company'].reindex(pivot.index),
                'site_type': meta['site_type'].reindex(pivot.index),
                'pattern': pattern,
            }, index=pivot.index)
            frames.append(pd.concat([frame, pivot], axis=1))

        months_sorted = sorted(all_months)
        table = pd.concat(frames, sort=False)
        order = {p: i for i, p in enumerate((*METER_PATTERN_ORDER, 'normal'))}
        table['_ord'] = table['pattern'].map(order)
        table = (table.sort_values(['_ord', 'provider', 'company', 'site_id'])
                 .drop(columns='_ord')
                 .reset_index(drop=True))
        table = table[meta_cols + months_sorted]

        # pattern composition per provider/company — feeds the chart view
        grouped = (table.groupby(['provider', 'company'], dropna=False)['pattern']
                   .value_counts().unstack(fill_value=0))
        per_company = []
        for (prov, comp), row in grouped.iterrows():
            entry = {"provider": str(prov),
                     "company": None if pd.isna(comp) else str(comp)}
            for p in (*METER_PATTERN_ORDER, 'normal'):
                entry[p] = int(row.get(p, 0))
            entry["total"] = int(row.sum())
            per_company.append(entry)
        per_company.sort(key=lambda e: (e["provider"], e["company"] or ""))

        result = {"months": months_sorted, "unique_meters": unique_total,
                  "unique_meters_per_provider": unique_per_provider,
                  "counts": counts, "counts_per_company": per_company,
                  "table": table}
        self._meter_patterns_cache = {window: result}
        return result

    @staticmethod
    def _meter_pattern_rows(table: pd.DataFrame, pattern: Optional[str]) -> pd.DataFrame:
        if pattern and not table.empty:
            return table[table['pattern'] == pattern]
        return table

    def eda_meter_patterns(self, window: int = 3,
                           pattern: Optional[str] = None,
                           limit: Optional[int] = None,
                           offset: int = 0) -> dict:
        """
        Paged JSON view over the per-meter datasheet built by
        _meter_patterns(). `pattern` filters to one class; `limit`/`offset`
        page through the (sorted) rows so the frontend never has to download
        every meter at once.
        """
        data = self._meter_patterns(window)
        table = self._meter_pattern_rows(data['table'], pattern)
        months = data['months']

        total = int(len(table))
        if offset:
            table = table.iloc[offset:]
        if limit is not None:
            table = table.iloc[:max(int(limit), 0)]

        records = [
            {
                "meter_no": None if pd.isna(row['meter_no']) else str(row['meter_no']),
                "site_id": str(row['site_id']),
                "provider": str(row['provider']),
                "company": None if pd.isna(row['company']) else str(row['company']),
                "site_type": None if pd.isna(row['site_type']) else str(row['site_type']),
                "pattern": str(row['pattern']),
                "monthly": [{"month": int(m), "bill_amount": round(float(row[m]), 2)}
                            for m in months if not pd.isna(row.get(m))],
            }
            for row in table.to_dict(orient='records')
        ]

        return {
            "window": window,
            "months": months,
            "unique_meters": data['unique_meters'],
            "unique_meters_per_provider": data['unique_meters_per_provider'],
            "counts": data['counts'],
            "counts_per_company": data['counts_per_company'],
            "total_records": total,
            "offset": int(offset),
            "records": records,
        }

    def meter_patterns_csv(self, window: int = 3,
                           pattern: Optional[str] = None,
                           chunk_rows: int = 1000):
        """Yield the meter-pattern datasheet as CSV text, in row chunks so a
        full export streams without materialising one giant string."""
        data = self._meter_patterns(window)
        table = self._meter_pattern_rows(data['table'], pattern)
        months = data['months']

        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator='\r\n')
        writer.writerow(['Meter No', 'Site ID', 'Provider', 'Company', 'Type',
                         'Pattern'] + [f"{str(m)[:4]}-{str(m)[4:]}" for m in months])

        for start in range(0, len(table), chunk_rows):
            for row in table.iloc[start:start + chunk_rows].to_dict(orient='records'):
                writer.writerow(
                    ['' if pd.isna(row[c]) else str(row[c])
                     for c in ('meter_no', 'site_id', 'provider', 'company',
                               'site_type', 'pattern')]
                    + ['' if pd.isna(row.get(m)) else f"{float(row[m]):.2f}"
                       for m in months])
            yield buf.getvalue()
            buf.seek(0)
            buf.truncate(0)

        remainder = buf.getvalue()
        if remainder:
            yield remainder

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
            "site_type": (None if 'site_type' not in sub.columns or pd.isna(sub['site_type'].iloc[0])
                          else str(sub['site_type'].iloc[0])),
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
