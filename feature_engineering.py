# ============================================================
# feature_engineering.py
# Produce time-safe monthly/quarterly features from daily prices + FF factors
# Outputs: data/features_quarterly.csv (rows: month-end decision date x ticker)
# ============================================================

import argparse
import os
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import yfinance as yf

# Silence warnings for clean output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ------------------------------------------------------------
# 1. Utilities
# ------------------------------------------------------------
def load_prices(prices_csv):
    print(f"Loading prices from {prices_csv}...")
    df = pd.read_csv(prices_csv)

    # Standardize column names
    df = df.rename(columns=str.lower)

    # Enforce Date type
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    required = {'date', 'ticker', 'adj_close'}
    if not required.issubset(set(df.columns)):
        raise RuntimeError(f"Prices CSV missing columns. Found: {list(df.columns)}; need {required}")

    return df


def make_monthly_from_daily(df_daily):
    """
    Aggregate daily adjusted prices into month-end prices and compute monthly returns.
    """
    print("Aggregating daily to monthly...")
    df = df_daily.copy()

    # Create Year-Month ID for grouping
    df['ym'] = df['date'].dt.to_period('M').dt.to_timestamp('M') + pd.offsets.MonthEnd(0)

    # Take the LAST price of the month per ticker
    # sort by date to ensure 'last' is actually the end of month
    df = df.sort_values(['ticker', 'date'])
    monthly = df.groupby(['ticker', 'ym'], as_index=False).agg({
        'adj_close': 'last',
        'volume': 'sum'
    })

    monthly = monthly.rename(columns={'ym': 'date'})

    # Compute Returns: Price(t) / Price(t-1) - 1
    monthly['ret'] = monthly.groupby('ticker')['adj_close'].pct_change()

    # Drop the first row (NaN return)
    monthly = monthly.dropna(subset=['ret'])

    return monthly


# ------------------------------------------------------------
# 2. Market Index
# ------------------------------------------------------------
def compute_monthly_index(index_symbol='^GSPC', start=None, end=None, prices_daily_df=None):
    """
    Get monthly returns for the benchmark index (S&P 500).
    """
    print(f"Computing market index ({index_symbol})...")

    # 1. Try to find index in existing data
    if prices_daily_df is not None:
        idx_df = prices_daily_df[prices_daily_df['ticker'] == index_symbol].copy()
        if not idx_df.empty:
            return make_monthly_from_daily(idx_df)[['date', 'ret']].rename(columns={'ret': 'index_ret'})

    # 2. Download if not found
    start = start or '2010-01-01'
    end = end or datetime.today().strftime('%Y-%m-%d')

    try:
        idx = yf.download(index_symbol, start=start, end=end, progress=False, auto_adjust=True)

        # Handle MultiIndex columns (yfinance > 0.2)
        if isinstance(idx.columns, pd.MultiIndex):
            # Try to find 'Close'
            try:
                idx = idx.xs('Close', level=0, axis=1)
            except KeyError:
                idx = idx.iloc[:, 0].to_frame('Close')  # Fallback to first col

        # Rename to generic
        if 'Close' in idx.columns:
            idx = idx.rename(columns={'Close': 'adj_close'})
        else:
            idx.columns = ['adj_close']  # Force first col

        idx = idx.reset_index()
        idx = idx.rename(columns={'Date': 'date', 'index': 'date'})
        idx['date'] = pd.to_datetime(idx['date'])

        # Resample to monthly
        idx.set_index('date', inplace=True)
        idx_m = idx['adj_close'].resample('ME').last().to_frame()
        idx_m = idx_m.reset_index()

        idx_m['index_ret'] = idx_m['adj_close'].pct_change()
        return idx_m[['date', 'index_ret']].dropna()

    except Exception as e:
        print(f"Warning: Index download failed ({e}). Using flat zero index returns (fallback).")
        # Fallback: create dummy index aligned to the user's data later
        return pd.DataFrame(columns=['date', 'index_ret'])


# ------------------------------------------------------------
# 3. Momentum Features (Vectorized)
# ------------------------------------------------------------
def momentum_features(monthly_df):
    """
    Compute standard momentum signals:
    - mom_3m: Return t-3 to t
    - mom_6m: Return t-6 to t
    - mom_12_ex1m: Return t-12 to t-1 (Standard academic Momentum)
    """
    print("Computing momentum features...")
    df = monthly_df.sort_values(['ticker', 'date']).copy()

    # Use Log returns for additive math (safer for long horizons)
    df['log_ret'] = np.log1p(df['ret'])

    g = df.groupby('ticker')['log_ret']

    # Rolling Sum of Log Returns = Total Log Return
    # shift(1) because we want features KNOWN at time t (based on past)
    # Actually, standard is: at time t, mom_3m uses prices P_t / P_{t-3} - 1.
    # So we include current row t in the window.

    # 3-Month Momentum
    df['mom_3m'] = g.rolling(3).sum().reset_index(0, drop=True)
    df['mom_3m'] = np.expm1(df['mom_3m'])  # Convert back to simple

    # 6-Month Momentum
    df['mom_6m'] = g.rolling(6).sum().reset_index(0, drop=True)
    df['mom_6m'] = np.expm1(df['mom_6m'])

    # 12-Month Momentum excluding most recent 1 month (Reversal protection)
    # Logic: Sum(t-11...t) - Ret(t)
    rolling_12 = g.rolling(12).sum().reset_index(0, drop=True)
    df['mom_12_ex1m'] = rolling_12 - df['log_ret']
    df['mom_12_ex1m'] = np.expm1(df['mom_12_ex1m'])

    df = df.drop(columns=['log_ret'])
    return df


# ------------------------------------------------------------
# 4. Rolling Fama-French Regression (Optimized)
# ------------------------------------------------------------
def rolling_ff_regression(monthly_df, ff_csv, window_months=36):
    """
    Compute rolling Alpha and Betas (Mkt, SMB, HML) using RollingOLS.
    Much faster than manual looping.
    """
    print(f"Computing rolling FF regressions (window={window_months})...")

    # Load FF
    ff = pd.read_csv(ff_csv)
    ff['date'] = pd.to_datetime(ff['date']) + pd.offsets.MonthEnd(0)

    # Merge
    merged = pd.merge(monthly_df, ff, on='date', how='inner')
    merged['excess_ret'] = merged['ret'] - merged['rf']

    # Sort for rolling
    merged = merged.sort_values(['ticker', 'date'])

    results = []

    # Group by ticker and apply RollingOLS
    # We use a loop over groups because RollingOLS doesn't natively support 'groupby'
    # but looping 500 tickers is instantaneous compared to looping 90k rows.

    # Define columns
    exog_vars = ['mkt_excess', 'smb', 'hml']

    total_tickers = merged['ticker'].nunique()
    count = 0

    for ticker, grp in merged.groupby('ticker'):
        # Ensure enough data
        if len(grp) < window_months:
            continue

        grp = grp.set_index('date')

        endog = grp['excess_ret']
        exog = sm.add_constant(grp[exog_vars])

        # Fit Rolling OLS
        model = RollingOLS(endog, exog, window=window_months)
        params = model.fit().params.copy()

        # params has the same index as grp (date)
        params['ticker'] = ticker
        results.append(params)

        count += 1
        if count % 100 == 0:
            print(f"  Processed {count}/{total_tickers} tickers...")

    if not results:
        raise RuntimeError("Rolling Regression failed: No data overlaps found between Prices and FF Factors.")

    # Combine results
    betas = pd.concat(results).reset_index()
    betas = betas.rename(columns={
        'const': 'alpha_roll',
        'mkt_excess': 'beta_mkt_roll',
        'smb': 'beta_smb_roll',
        'hml': 'beta_hml_roll'
    })

    return betas


# ------------------------------------------------------------
# 5. Labels & Final Merge
# ------------------------------------------------------------
def compute_next_quarter_labels(monthly_df, index_monthly_df):
    print("Computing next-quarter targets...")
    df = monthly_df.sort_values(['ticker', 'date']).copy()

    # Shift returns backward to get Future returns aligned with Current row
    # t+1, t+2, t+3
    r1 = df.groupby('ticker')['ret'].shift(-1)
    r2 = df.groupby('ticker')['ret'].shift(-2)
    r3 = df.groupby('ticker')['ret'].shift(-3)

    # Compound return: (1+r1)*(1+r2)*(1+r3) - 1
    df['next_q_ret'] = (1 + r1) * (1 + r2) * (1 + r3) - 1

    # Same for Index
    idx = index_monthly_df.sort_values('date').copy()
    i1 = idx['index_ret'].shift(-1)
    i2 = idx['index_ret'].shift(-2)
    i3 = idx['index_ret'].shift(-3)
    idx['next_q_index_ret'] = (1 + i1) * (1 + i2) * (1 + i3) - 1

    # Merge Index Target
    df = pd.merge(df, idx[['date', 'next_q_index_ret']], on='date', how='left')

    # Compute Excess
    df['next_q_excess'] = df['next_q_ret'] - df['next_q_index_ret']

    return df


def join_all(mom_df, ff_df, labels_df):
    """
    Robust clean merge of all features
    """
    print("Merging all features...")

    # Start with Momentum DF as base (it has ret and dates)
    base = labels_df.copy()

    # Merge Rolling Betas
    # Note: Rolling betas at date T are calculated using T-36...T.
    # Safe to use for predicting T+1.
    final = pd.merge(base, ff_df, on=['date', 'ticker'], how='left')

    # Merge Momentum (mom_df) - actually we computed momentum ON the base df earlier?
    # Ah, in main() we passed 'mom' to rolling.
    # Let's just ensure we have the momentum cols.
    # In this script structure, mom_df IS labels_df essentially.

    return final


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices_csv", default="data/prices_daily.csv")
    parser.add_argument("--ff_csv", default="data/ff_factors_monthly.csv")
    parser.add_argument("--out_csv", default="data/features_quarterly.csv")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--index_symbol", default="^GSPC")
    args = parser.parse_args()

    # 1. Load & Monthly Agg
    prices_daily = load_prices(args.prices_csv)
    monthly = make_monthly_from_daily(prices_daily)
    print(f"Monthly rows: {len(monthly)}; Tickers: {monthly['ticker'].nunique()}")

    # 2. Market Index
    index_monthly = compute_monthly_index(
        index_symbol=args.index_symbol,
        start=args.start,
        end=args.end,
        prices_daily_df=prices_daily
    )

    # 3. Momentum
    # (Adds columns to 'monthly')
    monthly = momentum_features(monthly)

    # 4. Rolling FF
    # (Returns new DF with betas)
    ff_roll = rolling_ff_regression(monthly, args.ff_csv, window_months=36)

    # 5. Labels
    # (Adds columns to 'monthly')
    monthly_labeled = compute_next_quarter_labels(monthly, index_monthly)

    # 6. Merge
    features = pd.merge(monthly_labeled, ff_roll, on=['date', 'ticker'], how='left')

    # 7. Filter for Quarter-Ends (Mar, Jun, Sep, Dec)
    # We only make decisions/trades at quarter ends
    features['month'] = features['date'].dt.month
    quarterly_mask = features['month'].isin([3, 6, 9, 12])
    features_q = features[quarterly_mask].copy()

    # 8. Final Clean & Save
    # Rename date to feature_as_of_date for clarity
    features_q = features_q.rename(columns={'date': 'feature_as_of_date'})

    # Select Columns
    keep_cols = [
        'feature_as_of_date', 'ticker',
        'ret', 'adj_close',  # Metadata
        'mom_3m', 'mom_6m', 'mom_12_ex1m',  # Momentum
        'alpha_roll', 'beta_mkt_roll', 'beta_smb_roll', 'beta_hml_roll',  # FF Factors
        'next_q_ret', 'next_q_excess'  # Labels
    ]

    # Ensure cols exist
    existing_cols = [c for c in keep_cols if c in features_q.columns]
    features_q = features_q[existing_cols]

    # Drop rows where Targets are NaN (cannot train on these)
    # (Usually the last 3 months of data won't have targets)
    features_q = features_q.dropna(subset=['next_q_excess', 'beta_mkt_roll'])

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    features_q.to_csv(args.out_csv, index=False)
    print(f"Feature engineering complete. Wrote {len(features_q)} rows to {args.out_csv}")