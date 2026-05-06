# ============================================================
# data_ingestion.py
# Robust ingestion for:
# - S&P 500 tickers
# - Daily adjusted prices (yfinance)
# - Fama-French 3-factor monthly data
#
# Windows-safe, batch-safe, research-grade
# ============================================================

import argparse
import os
import time
import tempfile
import warnings
import zipfile
import io
from datetime import datetime
from typing import List
from io import StringIO

import pandas as pd
import yfinance as yf
import requests

# ------------------------------------------------------------
# 0. Silence Warnings (New)
# ------------------------------------------------------------
# These filters block the "Timestamp.utcnow is deprecated" noise from yfinance
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message="Timestamp.utcnow is deprecated")


# ------------------------------------------------------------
# 1. Get S&P 500 tickers (robust)
# ------------------------------------------------------------
def get_sp500_tickers(fallback_csv=None) -> List[str]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    # 1   Local CSV fallback
    if fallback_csv and os.path.exists(fallback_csv):
        df = pd.read_csv(fallback_csv)
        col = "Symbol" if "Symbol" in df.columns else "Ticker"
        return df[col].str.replace(".", "-", regex=False).str.upper().tolist()

    # 2   Wikipedia (with headers)
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).str.upper().tolist()
        print(f"Fetched {len(tickers)} tickers from Wikipedia.")
        return tickers
    except Exception as e:
        print("Wikipedia fetch failed:", repr(e))

    # 3   GitHub fallback
    try:
        gh = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        resp = requests.get(gh, headers=headers, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        tickers = df["Symbol"].str.replace(".", "-", regex=False).str.upper().tolist()
        print(f"Fetched {len(tickers)} tickers from GitHub.")
        return tickers
    except Exception as e:
        print("GitHub fallback failed:", repr(e))

    raise RuntimeError(
        "Unable to fetch S&P 500 tickers. "
        "Provide a local CSV with column 'Symbol' via --universe_csv."
    )


# ------------------------------------------------------------
# 2. Download daily prices (Windows-safe)
# ------------------------------------------------------------
def download_prices(
        tickers: List[str],
        start: str,
        end: str,
        out_csv: str,
        batch_size: int = 50,
        sleep_between_batches: float = 1.0,
):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    print(f"Downloading prices for {len(tickers)} tickers...")

    # Temp file to avoid Windows file-lock issues
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix="prices_tmp_", suffix=".csv", dir=os.path.dirname(out_csv)
    )
    os.close(tmp_fd)

    wrote_header = False

    def append_to_tmp(df):
        nonlocal wrote_header
        df.to_csv(tmp_path, mode="a", header=not wrote_header, index=False)
        wrote_header = True

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i: i + batch_size]
        print(f"[Batch {i // batch_size + 1}] {batch[0]} → {batch[-1]}")

        try:
            # Silence yfinance progress bar to keep output clean
            data = yf.download(
                batch,
                start=start,
                end=end,
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception as e:
            print("Batch failed:", repr(e))
            continue

        rows = []

        if isinstance(data.columns, pd.MultiIndex):
            for tk in batch:
                if tk not in data:
                    continue
                df = data[tk].reset_index()
                # Rename columns robustly
                cols_map = {}
                for c in df.columns:
                    if c in ['Adj Close', 'Close', 'adj_close']:
                        cols_map[c] = 'adj_close'
                    elif c in ['Volume', 'volume']:
                        cols_map[c] = 'volume'
                    elif c in ['Date', 'date']:
                        cols_map[c] = 'date'

                df = df.rename(columns=cols_map)

                if "adj_close" not in df.columns:
                    continue

                # Ensure we have the minimal columns
                if 'volume' not in df.columns: df['volume'] = 0

                df = df[["date", "adj_close", "volume"]]
                df["ticker"] = tk
                rows.append(df)
        else:
            # Single ticker case or flat index
            df = data.reset_index()
            cols_map = {}
            for c in df.columns:
                if c in ['Adj Close', 'Close', 'adj_close']:
                    cols_map[c] = 'adj_close'
                elif c in ['Volume', 'volume']:
                    cols_map[c] = 'volume'
                elif c in ['Date', 'date']:
                    cols_map[c] = 'date'
            df = df.rename(columns=cols_map)

            if "adj_close" in df.columns:
                if 'volume' not in df.columns: df['volume'] = 0
                df = df[["date", "adj_close", "volume"]]
                df["ticker"] = batch[0]
                rows.append(df)

        if rows:
            out = pd.concat(rows, ignore_index=True)
            out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
            append_to_tmp(out)
            print(f"  Wrote {len(out)} rows")

        time.sleep(sleep_between_batches)

    # Atomic replace
    try:
        if os.path.exists(out_csv):
            os.remove(out_csv)
        os.replace(tmp_path, out_csv)
        print(f"Saved prices to {out_csv}")

    except PermissionError:
        print("WARNING: prices_daily.csv is open elsewhere.")
        print(f"Data saved to temp file: {tmp_path}")
        return tmp_path
    except Exception as e:
        print(f"Error saving file: {e}")
        return tmp_path

    return out_csv


# ------------------------------------------------------------
# 3. Fetch Fama-French 3-factor (Direct Download Fix)
# ------------------------------------------------------------
def fetch_ff_factors(start, end, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    print("Fetching Fama-French factors (Direct Download)...")
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            # Locate the CSV file in the zip
            csv_name = [n for n in z.namelist() if n.lower().endswith('.csv')][0]
            with z.open(csv_name) as f:
                # Skip header rows (usually 3)
                df = pd.read_csv(f, skiprows=3, index_col=0)
    except Exception as e:
        raise RuntimeError(f"Failed to download FF factors: {e}")

    # Clean up the index and format
    df.index.name = "date"
    df = df.reset_index()

    # Keep only valid monthly rows (YYYYMM)
    df['date'] = df['date'].astype(str).str.strip()
    df = df[df['date'].str.match(r'^\d{6}$')]

    # Convert to timestamp (End of Month)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m') + pd.offsets.MonthEnd(0)

    # Rename columns
    df = df.rename(columns={
        "Mkt-RF": "mkt_excess",
        "SMB": "smb",
        "HML": "hml",
        "RF": "rf"
    })

    # Convert from percent to decimal
    cols = ["mkt_excess", "smb", "hml", "rf"]
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce') / 100.0

    # Filter Date Range
    df = df[(df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))]

    # Select final columns
    df = df[["date", "mkt_excess", "smb", "hml", "rf"]]

    df.to_csv(out_csv, index=False)
    print(f"Saved Fama-French factors to {out_csv}")
    return out_csv


# ------------------------------------------------------------
# 4. Main
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2026-01-31")
    parser.add_argument("--universe_csv", default=None)
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--limit", type=int, default=None)  # Added for compatibility
    args = parser.parse_args()

    start = args.start
    end = args.end or datetime.today().strftime("%Y-%m-%d")

    tickers = get_sp500_tickers(args.universe_csv)
    if args.limit:
        tickers = tickers[:args.limit]

    # --- CLEAN AND SAVE TICKERS ---
    ticker_df = pd.DataFrame({"ticker": tickers})
    ticker_df["ticker"] = ticker_df["ticker"].astype(str).str.strip().str.upper()
    ticker_df = ticker_df.dropna(subset=["ticker"]).drop_duplicates(keep="first")
    tickers = ticker_df["ticker"].tolist()

    tickers_path = os.path.join(args.out_dir, "tickers.csv")
    os.makedirs(args.out_dir, exist_ok=True)
    ticker_df.to_csv(tickers_path, index=False)
    print(f"Saved {len(tickers)} tickers to tickers.csv")

    print(f"Universe size: {len(tickers)}")

    prices_csv = download_prices(
        tickers,
        start=start,
        end=end,
        out_csv=os.path.join(args.out_dir, "prices_daily.csv"),
    )

    ff_csv = fetch_ff_factors(
        start,
        end,
        out_csv=os.path.join(args.out_dir, "ff_factors_monthly.csv"),
    )

    print("\nDATA INGESTION COMPLETE")
    print("Files created:")
    print(" -", prices_csv)
    print(" -", ff_csv)