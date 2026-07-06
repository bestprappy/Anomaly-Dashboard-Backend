# Billing EDA Dashboard — Backend

FastAPI backend for the PEA/MEA billing dashboard. Accepts the 5 raw bill
exports, cleans + reshapes them with `DataBillContainer` (ported from
`True_billing.ipynb` and `MEA_cleaning.ipynb`), and serves the EDA
aggregations the frontend needs.

## Project layout

```
app/
  data_container.py   # DataBillContainer — ingestion, cleaning, EDA aggregations
  schemas.py           # Pydantic response models
  main.py              # FastAPI app + routes
requirements.txt
```

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs for interactive Swagger docs.

## The 5 expected uploads

| form field  | provider | company |
|-------------|----------|---------|
| `pea_bfkt`  | PEA      | BFKT    |
| `pea_tuc`   | PEA      | TUC     |
| `mea_bfkt`  | MEA      | BFKT    |
| `mea_tuc`   | MEA      | TUC     |
| `mea_tmv`   | MEA      | TMV     |

Each should be a CSV export of the original wide-format bill sheet (one row
per meter, one column per month for baht amount, one column per month for
kWh). `_read_any()` also accepts `.xlsx` directly if that's easier for a
given vendor export.

**Important CSV caveat:** the original Excel-based cleaning notebooks told
the "amount" block and "kWh" block of month-columns apart using Excel's
int-vs-string header typing (a quirk of how `pandas.read_excel` mangles
duplicate headers). CSV has no such typing — every header is a string — so
`DataBillContainer._mea_month_columns()` falls back to a **positional**
split: it walks the monthly columns in file order and cuts the block at the
point where the month sequence resets back to the first month. This assumes
the source sheet lists the amount block first, then the kWh block, each in
chronological order — true for the samples we tested against. If a vendor
ever exports the two blocks interleaved instead of stacked, this logic will
need revisiting.

## API

All endpoints are prefixed `/api`.

### Upload
- `POST /api/upload` — multipart form with any/all of the 5 fields above.
  Can be called incrementally (e.g. upload 2 files now, 3 more later); each
  call rebuilds the master table from whatever has been loaded so far.
- `GET /api/upload/status` — what's loaded / still missing.

### EDA
- `GET /api/eda/summary` — everything below, in one call (good for an
  overview dashboard that renders several cards from a single fetch).
- `GET /api/eda/bill-range` — min/max month covered, per provider and overall.
- `GET /api/eda/duplicates` — malformed `Site_ID`s (don't match
  `[A-Z]{2,4}\d{3,5}[A-Z]?`, e.g. `CBR4017` / `CBR4017A`) and `Site_ID`s that
  appear on more than one raw row across the 5 files — i.e. what a
  `groupby(Site_ID).first()` merge would silently collapse. Hand this list to
  whoever owns the site master data.
- `GET /api/eda/common-sites` — site overlap within PEA (BFKT∩TUC), within
  MEA (all pairs + 3-way), and across PEA↔MEA (same physical site billed by
  both authorities — worth flagging since it may indicate a
  double-connection or a site mid-migration between providers).
- `GET /api/eda/site-types` — count of each site status (NORMAL, WIFI,
  WIFI_CANCEL, DECOM, ...) per provider.
- `GET /api/eda/missing-consequence?windows=3,6,9` — sites with zero/missing
  kWh for *all* of the last N months, for N in the given windows. These are
  candidate "silently shut down but never marked DECOM" stations.
- `GET /api/eda/maintenance-sites` — value_counts of bill amounts inside the
  0–200 baht "maintenance" bucket. Recurring exact amounts (e.g. many sites
  billed exactly ฿49.39) usually mean a flat maintenance/rental fee — useful
  to show a vendor as "these X sites are billed a suspicious flat fee".
- `GET /api/eda/error-rates` — headline sanity-check numbers: zero-bill
  rate, bill-without-kWh rate, kWh-without-bill rate, missing-kWh rate,
  negative values, plus the row-by-row ingestion log (`load_reports`) so you
  can see exactly what got dropped from each raw file and why.

### Site lookup
- `GET /api/sites?provider=PEA|MEA` — all Site_IDs (for a search/autocomplete
  box).
- `GET /api/site/{site_id}/trend?metric=kwh|bill_amount&start_month=&end_month=`
  — a single site's monthly series, optionally windowed by `YYYYMM` bounds.
  Feed this straight into a line chart.

## Why these particular EDA metrics

The brief asked "is this enough, and would a business person be able to hand
this to the vendor?" A few additions beyond the original list, and why:

- **`error_rates`** — the four sanity ratios (zero-bill rate, bill-without-kWh,
  kWh-without-bill, missing-kWh) are the numbers a business user would
  actually quote back to a billing vendor as "X% of your invoices don't
  reconcile with metered usage." Without them the EDA tab shows *lists* of
  problems but no headline severity number.
- **`load_reports`** in `error_rates` — every row silently dropped during
  cleaning (ghost rows, MSC/RMSC exclusions, ambiguous site-type rows) is
  logged with a plain-English reason. This is what turns "the dashboard
  disagrees with the raw file" complaints into "here's exactly why, row by
  row."
- **Cross-provider common sites** (PEA∩MEA) — not in the original list, but
  a physical site billed twice, once by each authority, is exactly the kind
  of double-charge a business owner would want to catch.
- Duplicates are split into **malformed IDs** vs **true duplicate rows**,
  since they need different fixes: a malformed ID is a typo to correct at
  the source; a true duplicate is a merge decision (`keep='first'`) that
  silently drops billing history for the other row(s) — worth a human
  reviewing before the merge in case the dropped row wasn't actually a
  duplicate.

## Deploying

**Render (this API):**
1. Push this folder to a GitHub repo.
2. New Web Service on Render → connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

**Note on state:** the API keeps the processed table in memory
(`STATE["container"]` in `main.py`). That's fine for a single small Render
instance, but it resets on every redeploy/restart and won't work if you
scale to multiple instances. For anything beyond a single-user internal
tool, persist `container.master_df` (e.g. to S3 or a database) right after
`build_master()` and reload it on startup instead of relying on the
in-memory singleton.

**GitHub Pages (frontend):** point your static dashboard's fetch calls at
the Render URL, e.g. `https://your-api.onrender.com/api/eda/summary`. CORS
is currently wide open (`allow_origins=["*"]`) — tighten that to your Pages
origin before going to production.