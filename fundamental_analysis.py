"""Module 2: Fundamental Analysis Model

Free-data US fundamentals pipeline for 2016-01-01 to 2026-01-31.

What this script does
---------------------
1) Reads a list of US tickers from data/tickers.csv by default.
2) Loads ticker -> CIK mapping from a local cache if available.
3) Falls back to SEC ticker-mapping files if the cache is missing.
4) Pulls SEC companyfacts JSON for each company.
5) Extracts annual fundamentals with filing-date metadata.
6) Builds Piotroski F-Score.
7) Builds Ohlson O-score and default probability.
8) Optionally pulls FRED GDPDEF for the macro term.
9) Processes tickers in batches so the sequencing mirrors Module 1.
10) Exports a clean panel to CSV / Parquet.

Requirements
------------
pip install pandas numpy requests pyarrow

Environment variables
---------------------
SEC_USER_AGENT = 'Your Name your.email@example.com'
FRED_API_KEY   = optional, needed for GDPDEF fetch from FRED API

Files expected
--------------
data/tickers.csv
  - Your universe file from Module 1.
  - Must contain a ticker column named ticker/tickers/symbol/symbols.

data/ticker_cik_map.csv
  - Local cache of ticker -> CIK mapping.
  - Created automatically if SEC download succeeds.

Notes
-----
- SEC expects a descriptive User-Agent.
- This implementation uses annual data (FY / 10-K style) because it is
  the most stable choice for F-Score and Ohlson.
- The output keeps filing_date and period_end so later point-in-time
  alignment with daily prices is easier.
- If the SEC mapping file is unavailable, the script will use the cached
  local mapping file. If neither exists, it raises a clear error.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DEFAULT_TICKERS_CSV = os.path.join(DATA_DIR, "tickers.csv")
DEFAULT_TICKER_CIK_MAP_CSV = os.path.join(DATA_DIR, "ticker_cik_map.csv")
DEFAULT_OUT_CSV = os.path.join(OUTPUT_DIR, "module2_fundamentals.csv")
DEFAULT_OUT_PARQUET = os.path.join(OUTPUT_DIR, "module2_fundamentals.parquet")

SEC_HEADERS = {
    "User-Agent": os.getenv("SEC_USER_AGENT", "FundamentalResearchBot your.email@example.com"),
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json,text/plain,*/*",
}

# SEC documented ticker/CIK mapping files.
SEC_TICKER_MAP_URLS = [
    "https://www.sec.gov/files/company_tickers.json",
    "https://www.sec.gov/files/company_tickers_exchange.json",
]

SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"


# -------------------------------
# Configuration
# -------------------------------

@dataclass
class ModelConfig:
    tickers: List[str]
    start_date: str = "2016-01-01"
    end_date: str = "2026-01-31"
    batch_size: int = 50
    fred_series_id: str = "GDPDEF"

    @property
    def start_year(self) -> int:
        return pd.Timestamp(self.start_date).year

    @property
    def end_year(self) -> int:
        return pd.Timestamp(self.end_date).year


# -------------------------------
# Small utilities
# -------------------------------

EPS = 1e-12


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(SEC_HEADERS)
    return s


def chunked(seq: Sequence[str], size: int) -> List[List[str]]:
    size = max(1, int(size))
    return [list(seq[i : i + size]) for i in range(0, len(seq), size)]


def download_json(url: str, session: Optional[requests.Session] = None, timeout: int = 60, retries: int = 3) -> dict:
    sess = session or _session()
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            r = sess.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
            else:
                raise last_err
    raise last_err  # pragma: no cover


def normalize_ticker_map_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize SEC ticker mapping dataframe to columns: ticker, cik, title."""
    if df.empty:
        return df

    cols = {c.lower(): c for c in df.columns}
    ticker_col = cols.get("ticker")
    title_col = cols.get("title")
    cik_col = cols.get("cik") or cols.get("cik_str")

    if ticker_col is None or cik_col is None:
        raise ValueError("Ticker map file must contain ticker and CIK columns")

    out = pd.DataFrame()
    out["ticker"] = df[ticker_col].astype(str).str.upper().str.strip()
    out["cik"] = pd.to_numeric(df[cik_col], errors="coerce").astype("Int64")
    out["title"] = df[title_col].astype(str).fillna("") if title_col is not None else ""
    out = out.dropna(subset=["ticker", "cik"]).drop_duplicates(subset=["ticker"], keep="first")
    out["cik"] = out["cik"].astype(int)
    return out.reset_index(drop=True)


def load_ticker_cik_map(session: Optional[requests.Session] = None, cache_path: str = DEFAULT_TICKER_CIK_MAP_CSV) -> pd.DataFrame:
    """Load ticker -> CIK mapping from local cache first, then SEC fallback.

    If SEC download succeeds, the mapping is cached to disk for later runs.
    """
    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path)
        cached = normalize_ticker_map_df(cached)
        print(f"Loaded ticker map from local cache: {cache_path}")
        return cached

    sess = session or _session()
    last_error: Optional[Exception] = None

    for url in SEC_TICKER_MAP_URLS:
        try:
            data = download_json(url, sess)
            rows = []
            if isinstance(data, dict):
                for _, v in data.items():
                    if not isinstance(v, dict):
                        continue
                    ticker = v.get("ticker")
                    cik = v.get("cik_str", v.get("cik"))
                    title = v.get("title", "")
                    if ticker is None or cik is None:
                        continue
                    rows.append({"ticker": ticker, "cik": int(cik), "title": title})
            elif isinstance(data, list):
                for v in data:
                    if not isinstance(v, dict):
                        continue
                    ticker = v.get("ticker")
                    cik = v.get("cik_str", v.get("cik"))
                    title = v.get("title", "")
                    if ticker is None or cik is None:
                        continue
                    rows.append({"ticker": ticker, "cik": int(cik), "title": title})
            df = pd.DataFrame(rows)
            if df.empty:
                raise ValueError(f"Ticker map downloaded from {url} but no usable rows were parsed")
            df = normalize_ticker_map_df(df)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df.to_csv(cache_path, index=False)
            print(f"Downloaded ticker map from SEC and cached to: {cache_path}")
            return df
        except Exception as e:
            last_error = e

    raise RuntimeError(
        "Could not load ticker-CIK map from SEC and no local cache exists. "
        f"Create {cache_path} once, then rerun. Last error: {last_error}"
    )


def get_companyfacts(cik: int, session: Optional[requests.Session] = None, pause_s: float = 0.15) -> dict:
    """Fetch companyfacts JSON for a single CIK."""
    time.sleep(pause_s)
    sess = session or _session()
    url = SEC_COMPANYFACTS_URL.format(cik=int(cik))
    return download_json(url, sess)


def _normalize_fact_df(df: pd.DataFrame, concept: str, taxonomy: str, unit: str) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["concept"] = concept
    out["taxonomy"] = taxonomy
    out["unit"] = unit
    for col in ["fy", "fp", "form", "filed", "end", "frame"]:
        if col not in out.columns:
            out[col] = pd.NA
    out["filed"] = pd.to_datetime(out["filed"], errors="coerce")
    out["end"] = pd.to_datetime(out["end"], errors="coerce")
    out["fy"] = pd.to_numeric(out["fy"], errors="coerce").astype("Int64")
    return out


def extract_concept_facts(
    companyfacts: dict,
    candidates: Sequence[Tuple[str, str]],
    unit_preferences: Sequence[str],
    annual_only: bool = True,
    forms: Sequence[str] = ("10-K", "10-K/A"),
) -> pd.DataFrame:
    """Extract a single concept from companyfacts using fallback tag candidates."""
    facts = companyfacts.get("facts", {})

    for taxonomy, tag in candidates:
        if taxonomy not in facts:
            continue
        tax_obj = facts[taxonomy]
        if tag not in tax_obj:
            continue
        units = tax_obj[tag].get("units", {})
        unit_names = [u for u in unit_preferences if u in units]
        if not unit_names:
            unit_names = list(units.keys())

        frames = []
        for unit in unit_names:
            rows = units.get(unit, [])
            if not rows:
                continue
            df = pd.DataFrame(rows)
            if df.empty:
                continue
            df = _normalize_fact_df(df, tag, taxonomy, unit)
            if annual_only:
                if "fp" in df.columns:
                    df = df[df["fp"].eq("FY")]
                if "form" in df.columns:
                    df = df[df["form"].isin(list(forms))]
            if not df.empty:
                frames.append(df)

        if frames:
            out = pd.concat(frames, ignore_index=True)
            out = out.dropna(subset=["fy", "val"])
            out = out.sort_values(["fy", "filed", "end"])
            out = out.groupby("fy", as_index=False).tail(1)
            return out.reset_index(drop=True)

    return pd.DataFrame()


# -------------------------------
# Field extraction / normalization
# -------------------------------

FACT_TAGS = {
    "assets": [("us-gaap", "Assets")],
    "liabilities": [("us-gaap", "Liabilities")],
    "current_assets": [("us-gaap", "AssetsCurrent")],
    "current_liabilities": [("us-gaap", "LiabilitiesCurrent")],
    "long_term_debt": [
        ("us-gaap", "LongTermDebtAndCapitalLeaseObligations"),
        ("us-gaap", "LongTermDebt"),
        ("us-gaap", "LongTermDebtNoncurrent"),
    ],
    "net_income": [("us-gaap", "NetIncomeLoss")],
    "cfo": [("us-gaap", "NetCashProvidedByUsedInOperatingActivities")],
    "revenue": [
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
        ("us-gaap", "SalesRevenueNet"),
        ("us-gaap", "Revenues"),
    ],
    "gross_profit": [("us-gaap", "GrossProfit")],
    "cost_of_revenue": [("us-gaap", "CostOfRevenue"), ("us-gaap", "CostOfGoodsSold")],
    "shares_outstanding": [
        ("dei", "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "CommonStockSharesOutstanding"),
    ],
}


def build_company_annual_panel(companyfacts: dict) -> pd.DataFrame:
    """Return a yearly panel from a single company's companyfacts JSON.

    The panel includes filing_date and period_end so price data can be aligned
    point-in-time later.
    """
    extracted: Dict[str, pd.DataFrame] = {}

    for key, candidates in FACT_TAGS.items():
        unit_prefs = ["USD", "shares", "pure"]
        if key == "shares_outstanding":
            unit_prefs = ["shares", "USD", "pure"]
        extracted[key] = extract_concept_facts(companyfacts, candidates, unit_prefs)

    years: set[int] = set()
    for df in extracted.values():
        if not df.empty and "fy" in df.columns:
            years.update(int(y) for y in df["fy"].dropna().astype(int).tolist())

    if not years:
        return pd.DataFrame()

    panel = pd.DataFrame({"fy": sorted(years)})

    meta_frames = []
    for key, df in extracted.items():
        if df.empty:
            panel[key] = np.nan
            continue

        temp = df[[c for c in ["fy", "val", "filed", "end"] if c in df.columns]].copy()
        temp = temp.rename(columns={"val": key})
        value_only = temp[["fy", key]].groupby("fy", as_index=False).tail(1)
        panel = panel.merge(value_only, on="fy", how="left")

        meta = temp[["fy", "filed", "end"]].copy()
        meta_frames.append(meta)

    if meta_frames:
        meta_all = pd.concat(meta_frames, ignore_index=True)
        meta_all["filed"] = pd.to_datetime(meta_all["filed"], errors="coerce")
        meta_all["end"] = pd.to_datetime(meta_all["end"], errors="coerce")
        meta_all = meta_all.sort_values(["fy", "filed", "end"])
        meta_all = meta_all.groupby("fy", as_index=False).agg(
            filing_date=("filed", "max"),
            period_end=("end", "max"),
        )
        panel = panel.merge(meta_all, on="fy", how="left")
    else:
        panel["filing_date"] = pd.NaT
        panel["period_end"] = pd.NaT

    for col in panel.columns:
        if col not in {"fy", "filing_date", "period_end"}:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")

    panel["working_capital"] = panel["current_assets"] - panel["current_liabilities"]
    panel["gross_profit_calc"] = np.where(
        panel["gross_profit"].notna(),
        panel["gross_profit"],
        panel["revenue"] - panel["cost_of_revenue"],
    )
    panel["gross_margin"] = panel["gross_profit_calc"] / panel["revenue"].replace(0, np.nan)
    panel["asset_turnover"] = panel["revenue"] / panel["assets"].replace(0, np.nan)
    panel["roa"] = panel["net_income"] / panel["assets"].replace(0, np.nan)
    panel["cfo_ratio"] = panel["cfo"] / panel["assets"].replace(0, np.nan)
    panel["leverage_ratio"] = panel["long_term_debt"] / panel["assets"].replace(0, np.nan)
    panel["current_ratio"] = panel["current_assets"] / panel["current_liabilities"].replace(0, np.nan)
    panel["shares_proxy"] = panel["shares_outstanding"]
    panel = panel.sort_values("fy").reset_index(drop=True)
    return panel


# -------------------------------
# FRED helper
# -------------------------------

def fetch_fred_series(
    series_id: str,
    api_key: Optional[str],
    start_date: str,
    end_date: str,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """Fetch a FRED series via the official API."""
    if not api_key:
        return pd.DataFrame(columns=["date", "value", "series_id"])

    sess = session or requests.Session()
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
    }
    r = sess.get(FRED_OBSERVATIONS_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json().get("observations", [])

    rows = []
    for row in data:
        val = row.get("value")
        try:
            val = float(val)
        except Exception:
            val = np.nan
        rows.append({"date": row.get("date"), "value": val, "series_id": series_id})

    out = pd.DataFrame(rows)
    if not out.empty:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["year"] = out["date"].dt.year
    return out


# -------------------------------
# Scoring
# -------------------------------

def compute_piotroski_f_score(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy().sort_values("fy").reset_index(drop=True)

    df["prev_roa"] = df["roa"].shift(1)
    df["prev_leverage_ratio"] = df["leverage_ratio"].shift(1)
    df["prev_current_ratio"] = df["current_ratio"].shift(1)
    df["prev_shares_proxy"] = df["shares_proxy"].shift(1)
    df["prev_gross_margin"] = df["gross_margin"].shift(1)
    df["prev_asset_turnover"] = df["asset_turnover"].shift(1)
    df["prev_net_income"] = df["net_income"].shift(1)

    df["f_roa_positive"] = (df["roa"] > 0).astype("Int64")
    df["f_cfo_positive"] = (df["cfo"] > 0).astype("Int64")
    df["f_delta_roa"] = (df["roa"] > df["prev_roa"]).astype("Int64")
    df["f_cfo_gt_ni"] = (df["cfo"] > df["net_income"]).astype("Int64")
    df["f_lower_leverage"] = (df["leverage_ratio"] < df["prev_leverage_ratio"]).astype("Int64")
    df["f_higher_current_ratio"] = (df["current_ratio"] > df["prev_current_ratio"]).astype("Int64")
    df["f_no_new_shares"] = (df["shares_proxy"] <= df["prev_shares_proxy"]).astype("Int64")
    df["f_higher_gross_margin"] = (df["gross_margin"] > df["prev_gross_margin"]).astype("Int64")
    df["f_higher_asset_turnover"] = (df["asset_turnover"] > df["prev_asset_turnover"]).astype("Int64")

    f_cols = [
        "f_roa_positive",
        "f_cfo_positive",
        "f_delta_roa",
        "f_cfo_gt_ni",
        "f_lower_leverage",
        "f_higher_current_ratio",
        "f_no_new_shares",
        "f_higher_gross_margin",
        "f_higher_asset_turnover",
    ]
    df[f_cols] = df[f_cols].fillna(0).astype(int)
    df["f_score"] = df[f_cols].sum(axis=1)
    df["f_score_coverage_years"] = df[["prev_roa", "prev_leverage_ratio", "prev_current_ratio"]].notna().sum(axis=1)
    return df


def compute_ohlson_score(panel: pd.DataFrame, gdpdef: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    df = panel.copy().sort_values("fy").reset_index(drop=True)

    df["ta"] = df["assets"].clip(lower=EPS)
    df["tl"] = df["liabilities"].clip(lower=EPS)
    df["wc_ta"] = df["working_capital"] / df["ta"]
    df["cl_ca"] = df["current_liabilities"] / df["current_assets"].replace(0, np.nan)
    df["tl_ta"] = df["tl"] / df["ta"]
    df["nita"] = df["net_income"] / df["ta"]
    df["futl"] = df["cfo"] / df["tl"]
    df["oeneg"] = (df["tl"] > df["ta"]).astype(int)
    df["intwo"] = ((df["net_income"] < 0) & (df["prev_net_income"] < 0)).astype(int)
    df["chin"] = (df["net_income"] - df["prev_net_income"]) / (
        df["net_income"].abs() + df["prev_net_income"].abs() + EPS
    )
    df["size"] = np.log(df["ta"])

    df["ohlson_linear"] = (
        -1.32
        - 0.407 * df["size"]
        + 6.03 * df["tl_ta"]
        - 1.43 * df["wc_ta"]
        + 0.076 * df["cl_ca"]
        - 1.72 * df["oeneg"]
        - 2.37 * df["nita"]
        - 1.83 * df["futl"]
        + 0.285 * df["intwo"]
        - 0.521 * df["chin"]
    )

    if gdpdef is not None and not gdpdef.empty:
        g = gdpdef.copy()
        if "year" not in g.columns:
            g["year"] = pd.to_datetime(g["date"], errors="coerce").dt.year
        annual = g.groupby("year", as_index=False)["value"].mean().rename(columns={"value": "gdpdef"})
        df = df.merge(annual, left_on="fy", right_on="year", how="left").drop(columns=["year"], errors="ignore")
        base = df["gdpdef"].dropna().iloc[0] if df["gdpdef"].notna().any() else np.nan
        df["gdpdef_rel"] = df["gdpdef"] / base if pd.notna(base) else np.nan
    else:
        df["gdpdef"] = np.nan
        df["gdpdef_rel"] = np.nan

    df["ohlson_default_prob"] = 1.0 / (1.0 + np.exp(-df["ohlson_linear"]))
    return df


# -------------------------------
# End-to-end pipeline
# -------------------------------

def build_model_for_ticker(
    ticker: str,
    ticker_map: pd.DataFrame,
    session: Optional[requests.Session] = None,
    fred_api_key: Optional[str] = None,
    start_date: str = "2016-01-01",
    end_date: str = "2026-01-31",
) -> pd.DataFrame:
    sess = session or _session()
    t = ticker.upper().strip()
    row = ticker_map.loc[ticker_map["ticker"].eq(t)]
    if row.empty:
        raise ValueError(f"Ticker not found in ticker map: {ticker}")

    cik = int(row.iloc[0]["cik"])
    companyfacts = get_companyfacts(cik, sess)
    panel = build_company_annual_panel(companyfacts)
    if panel.empty:
        return panel

    panel = panel[(panel["fy"] >= pd.Timestamp(start_date).year - 1) & (panel["fy"] <= pd.Timestamp(end_date).year)]
    panel = panel.sort_values("fy").reset_index(drop=True)

    panel.insert(0, "ticker", t)
    panel.insert(1, "cik", cik)
    panel.insert(2, "company_name", row.iloc[0]["title"])

    panel = compute_piotroski_f_score(panel)

    gdp = fetch_fred_series(
        series_id="GDPDEF",
        api_key=fred_api_key,
        start_date=str(pd.Timestamp(start_date) - pd.DateOffset(years=1)).split(" ")[0],
        end_date=end_date,
        session=sess,
    )
    panel = compute_ohlson_score(panel, gdp)

    panel = panel[(panel["fy"] >= pd.Timestamp(start_date).year) & (panel["fy"] <= pd.Timestamp(end_date).year)]
    panel = panel.sort_values(["ticker", "fy"]).reset_index(drop=True)
    return panel


def build_universe_model_batched(
    tickers: Sequence[str],
    start_date: str = "2016-01-01",
    end_date: str = "2026-01-31",
    batch_size: int = 50,
    out_csv: Optional[str] = None,
    out_parquet: Optional[str] = None,
    ticker_cik_map_path: str = DEFAULT_TICKER_CIK_MAP_CSV,
) -> pd.DataFrame:
    sess = _session()
    ticker_map = load_ticker_cik_map(sess, cache_path=ticker_cik_map_path)
    fred_api_key = os.getenv("FRED_API_KEY")

    tickers = [str(t).upper().strip() for t in tickers if str(t).strip()]
    batches = chunked(tickers, batch_size)

    all_rows: List[pd.DataFrame] = []

    print(f"Universe size: {len(tickers)}")
    print(f"Processing fundamentals in {len(batches)} batches...")

    for i, batch in enumerate(batches, start=1):
        print(f"[Batch {i}] {batch[0]} → {batch[-1]}")
        batch_rows: List[pd.DataFrame] = []

        for ticker in batch:
            try:
                df = build_model_for_ticker(
                    ticker=ticker,
                    ticker_map=ticker_map,
                    session=sess,
                    fred_api_key=fred_api_key,
                    start_date=start_date,
                    end_date=end_date,
                )
                if not df.empty:
                    batch_rows.append(df)
            except Exception as e:
                print(f"[WARN] {ticker}: {e}")

        if batch_rows:
            batch_df = pd.concat(batch_rows, ignore_index=True)
            print(f"  Wrote {len(batch_df):,} rows")
            all_rows.append(batch_df)
        else:
            print("  Wrote 0 rows")

    if not all_rows:
        return pd.DataFrame()

    out = pd.concat(all_rows, ignore_index=True)

    front = ["ticker", "cik", "company_name", "fy", "filing_date", "period_end"]
    preferred = [
        "assets",
        "liabilities",
        "current_assets",
        "current_liabilities",
        "working_capital",
        "long_term_debt",
        "net_income",
        "cfo",
        "revenue",
        "gross_profit",
        "cost_of_revenue",
        "gross_profit_calc",
        "roa",
        "cfo_ratio",
        "leverage_ratio",
        "current_ratio",
        "asset_turnover",
        "f_score",
        "ohlson_linear",
        "ohlson_default_prob",
        "gdpdef",
        "gdpdef_rel",
    ]
    cols = [c for c in front + preferred + [c for c in out.columns if c not in front + preferred] if c in out.columns]
    out = out[cols].sort_values(["ticker", "fy"]).reset_index(drop=True)

    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        out.to_csv(out_csv, index=False)
    if out_parquet:
        os.makedirs(os.path.dirname(out_parquet) or ".", exist_ok=True)
        out.to_parquet(out_parquet, index=False)

    return out


# -------------------------------
# CLI
# -------------------------------

def read_tickers_from_csv(path: str) -> List[str]:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Ticker CSV is empty: {path}")

    ticker_col = None
    for candidate in ["ticker", "tickers", "symbol", "symbols"]:
        if candidate in df.columns:
            ticker_col = candidate
            break

    if ticker_col is None:
        raise ValueError("CSV must contain a ticker column named ticker/tickers/symbol/symbols")

    return [str(x).upper().strip() for x in df[ticker_col].dropna().tolist()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Module 2: Fundamental analysis model (F-Score + Ohlson)")
    p.add_argument("--tickers", nargs="*", help="List of tickers, e.g. AAPL MSFT JPM")
    p.add_argument("--tickers-csv", type=str, help="CSV file with a ticker column")
    p.add_argument("--ticker-cik-map", type=str, default=DEFAULT_TICKER_CIK_MAP_CSV, help="Local ticker->CIK cache file")
    p.add_argument("--start-date", type=str, default="2016-01-01")
    p.add_argument("--end-date", type=str, default="2026-01-31")
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--out-csv", type=str, default=DEFAULT_OUT_CSV)
    p.add_argument("--out-parquet", type=str, default=DEFAULT_OUT_PARQUET)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.tickers_csv:
        tickers = read_tickers_from_csv(args.tickers_csv)
    elif args.tickers:
        tickers = [t.upper().strip() for t in args.tickers]
    elif os.path.exists(DEFAULT_TICKERS_CSV):
        tickers = read_tickers_from_csv(DEFAULT_TICKERS_CSV)
        print(f"No tickers passed. Using default file: {DEFAULT_TICKERS_CSV}")
    else:
        raise SystemExit(
            f"Provide either --tickers, --tickers-csv, or place a ticker file at {DEFAULT_TICKERS_CSV}"
        )

    result = build_universe_model_batched(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        batch_size=args.batch_size,
        out_csv=args.out_csv,
        out_parquet=args.out_parquet,
        ticker_cik_map_path=args.ticker_cik_map,
    )

    print("Done.")
    if not result.empty:
        print(result.head(10).to_string(index=False))
        print(f"Rows: {len(result):,}")
        print(f"Tickers: {result['ticker'].nunique()}")
    else:
        print("No rows returned.")


if __name__ == "__main__":
    main()
