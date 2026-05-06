# ============================================================
# models_and_backtest.py
# Walk-forward ML (LightGBM), Purge+Embargo, Backtesting
# ============================================================

import os
import argparse
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# --- NEW Plotting Imports ---
import matplotlib.pyplot as plt
import seaborn as sns

# Silence warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 1. Robust Imports
try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor

    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("WARNING: LightGBM not found. Falling back to RandomForest.")

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# ------------------------------------------------------------
# Utils: Purging & Embargoing
# ------------------------------------------------------------
def apply_purge_embargo(train_df, test_date, label_months=3, embargo_months=1):
    """
    Strictly remove training samples that overlap with the test period.

    Test Date: T (Prediction made at T for returns T -> T+3)

    Training Sample S:
    - Features known at S.
    - Label is return S -> S+3.
    - Label available at S+3.

    Constraint: S+3 must be <= T (Outcome must be known before we predict).
    Embargo: S+3 <= T - Embargo (Buffer).

    So, Keep rows where: sample_date <= T - 3 months - 1 month
    """
    # Calculate the latest safe date for training features
    # train_date + 3 (label duration) + 1 (embargo) <= test_date
    # train_date <= test_date - 4 months
    cutoff_date = test_date - pd.DateOffset(months=label_months + embargo_months)

    # Filter
    safe_train = train_df[train_df['feature_as_of_date'] <= cutoff_date]
    return safe_train


# ------------------------------------------------------------
# Backtest Logic
# ------------------------------------------------------------
def run_backtest(preds, mode='decile', tc_bps=10):
    """
    Backtest the predictions.
    preds df cols: feature_as_of_date, ticker, y_pred, y_true, next_q_ret
    """
    dates = sorted(preds['feature_as_of_date'].unique())
    results = []

    print(f"Running backtest (Mode: {mode}, TC: {tc_bps} bps)...")

    for d in dates:
        slice_df = preds[preds['feature_as_of_date'] == d].copy()

        # Sort by prediction score (descending)
        slice_df = slice_df.sort_values('y_pred', ascending=False)
        n = len(slice_df)

        if n < 10: continue  # Skip if too few stocks

        # 1. Strategy Logic
        if mode == 'decile':
            # Long Top 10%, Short Bottom 10%
            decile_size = int(n * 0.10)
            longs = slice_df.iloc[:decile_size]
            shorts = slice_df.iloc[-decile_size:]

            long_ret = longs['next_q_ret'].mean()
            short_ret = shorts['next_q_ret'].mean()

            # Gross Return = Long - Short
            gross_ret = long_ret - short_ret

            # Costs: 2 legs (Long + Short) * 2 trades (Open + Close) * bps
            # Note: This assumes 100% turnover every quarter (conservative worst case)
            cost = 4 * (tc_bps / 10000.0)

        elif mode == 'long_only':
            # Long Top 10%
            decile_size = int(n * 0.10)
            longs = slice_df.iloc[:decile_size]

            long_ret = longs['next_q_ret'].mean()
            short_ret = 0.0
            gross_ret = long_ret

            cost = 2 * (tc_bps / 10000.0)

        elif mode == 'score_weighted':
            # Weight proportional to score
            # Shift scores to be positive
            min_score = slice_df['y_pred'].min()
            slice_df['w'] = slice_df['y_pred'] - min_score

            # If all scores identical, skip
            if slice_df['w'].sum() == 0:
                gross_ret = 0
            else:
                slice_df['w'] = slice_df['w'] / slice_df['w'].sum()
                gross_ret = (slice_df['w'] * slice_df['next_q_ret']).sum()

            long_ret = gross_ret  # simplificaton for reporting
            short_ret = 0.0
            cost = 2 * (tc_bps / 10000.0)

        net_ret = gross_ret - cost

        results.append({
            'date': d,
            'long_ret': long_ret,
            'short_ret': short_ret,
            'gross_ret': gross_ret,
            'net_ret': net_ret,
            'turnover': 1.0,  # Placeholder (assuming 100% rebal)
            'n_stocks': n
        })

    return pd.DataFrame(results)


def calc_metrics(res_df):
    if res_df.empty: return {}

    # Input returns are quarterly (approx 3 months)
    # Annualize factor = 4

    rets = res_df['net_ret']

    # Geometric Annual Return
    # (1 + r_q1)*(1 + r_q2)...
    cum_ret = (1 + rets).prod()
    n_years = len(rets) / 4.0
    if n_years == 0: n_years = 1

    ann_ret = (cum_ret ** (1 / n_years)) - 1

    # Annualized Volatility
    # Std(quarterly) * sqrt(4)
    ann_vol = rets.std() * np.sqrt(4)

    # Sharpe (Rf = 0 for simplicity or assume 3% annual)
    rf_annual = 0.03
    sharpe = (ann_ret - rf_annual) / (ann_vol + 1e-9)

    # Max Drawdown
    nav = (1 + rets).cumprod()
    peak = nav.cummax()
    dd = (nav - peak) / peak
    max_dd = dd.min()

    return {
        "Annualized Return": f"{ann_ret:.2%}",
        "Annualized Vol": f"{ann_vol:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_dd:.2%}"
    }


# ------------------------------------------------------------
# Walk-Forward Training
# ------------------------------------------------------------
def train_and_predict(features_csv, start_year=5, min_obs=200):
    print(f"Loading features from {features_csv}...")
    df = pd.read_csv(features_csv)
    df['feature_as_of_date'] = pd.to_datetime(df['feature_as_of_date'])

    # Identify Feature Columns
    # Exclude metadata and forward-looking labels
    exclude = ['feature_as_of_date', 'ticker', 'next_q_ret', 'next_q_excess', 'ret', 'adj_close']
    feat_cols = [c for c in df.columns if c not in exclude]

    # Ensure only numeric columns
    feat_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])]

    print(f"Features ({len(feat_cols)}): {feat_cols}")

    dates = sorted(df['feature_as_of_date'].unique())
    print(f"Total quarters available: {len(dates)}")

    # Expanding Window Loop
    # Need at least 'start_year' years of data to start
    start_idx = int(start_year * 4)

    if start_idx >= len(dates):
        print("Warning: Not enough history for requested start_years. Starting halfway.")
        start_idx = len(dates) // 2

    oos_preds = []
    model = None
    scaler = StandardScaler()

    print(f"Starting Walk-Forward Validation (Start Date: {dates[start_idx].date()})...")

    for i in range(start_idx, len(dates)):
        test_date = dates[i]

        # 1. Purge & Embargo
        # We can only train on data that was fully available BEFORE test_date
        train_df = apply_purge_embargo(df, test_date)

        test_df = df[df['feature_as_of_date'] == test_date].copy()

        if len(train_df) < min_obs:
            print(f"[{test_date.date()}] Skipping: Insufficient train data ({len(train_df)})")
            continue

        if test_df.empty:
            continue

        # 2. Prepare X, y
        X_train = train_df[feat_cols].fillna(0)
        y_train = train_df['next_q_excess']  # Predict Excess Return

        X_test = test_df[feat_cols].fillna(0)

        # 3. Scale
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 4. Train
        if HAS_LGB:
            # LightGBM is fast and handles non-linearities
            model = LGBMRegressor(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
            model.fit(X_train_scaled, y_train)
        else:
            model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
            model.fit(X_train_scaled, y_train)

        # 5. Predict
        preds = model.predict(X_test_scaled)

        # 6. Store
        test_df['y_pred'] = preds
        test_df['y_true'] = test_df['next_q_excess']

        # We need next_q_ret for backtesting PnL
        cols_to_save = ['feature_as_of_date', 'ticker', 'y_pred', 'y_true', 'next_q_ret']
        oos_preds.append(test_df[cols_to_save])

        print(f"[{test_date.date()}] Train: {len(X_train)} | Test: {len(X_test)} | Preds generated.")

    if not oos_preds:
        raise RuntimeError("No predictions generated. Check your dates or min_train_obs.")

    all_preds = pd.concat(oos_preds)
    return all_preds, model, scaler, feat_cols


# ------------------------------------------------------------
# Plotting Logic (For Research Paper)
# ------------------------------------------------------------
def plot_backtest_results(res_df, out_dir, mode):
    """Generates and saves high-res equity curve and drawdown plots."""
    if res_df.empty: return

    # Ensure dates are datetime objects
    res_df['date'] = pd.to_datetime(res_df['date'])
    res_df = res_df.sort_values('date')

    # Calculate Cumulative Returns & Drawdowns
    res_df['cum_net'] = (1 + res_df['net_ret']).cumprod()
    res_df['cum_gross'] = (1 + res_df['gross_ret']).cumprod()

    peak_net = res_df['cum_net'].cummax()
    res_df['drawdown_net'] = (res_df['cum_net'] - peak_net) / peak_net

    # Set plot style suitable for academic papers
    sns.set_theme(style="whitegrid", context="paper")

    # --- 1. Equity Curve Plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(res_df['date'], res_df['cum_net'], label='Net Return', color='#1f77b4', linewidth=2)
    plt.plot(res_df['date'], res_df['cum_gross'], label='Gross Return', color='#aec7e8', linestyle='--', linewidth=1.5)
    plt.title(f'Cumulative Returns: {mode.replace("_", " ").title()} Strategy', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Growth of $1', fontsize=12)
    plt.legend(loc='upper left', frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'equity_curve_{mode}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- 2. Drawdown Plot ---
    plt.figure(figsize=(10, 4))
    plt.fill_between(res_df['date'], res_df['drawdown_net'], 0, color='#d62728', alpha=0.3)
    plt.plot(res_df['date'], res_df['drawdown_net'], color='#d62728', linewidth=1)
    plt.title('Strategy Drawdown (Net Returns)', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Drawdown (%)', fontsize=12)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'drawdown_{mode}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved performance plots to {out_dir}/")

def plot_shap_importance(shap_df, out_dir, top_n=20):
    """Generates and saves a high-res bar plot of top SHAP features."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.figure(figsize=(10, 8))

    plot_data = shap_df.head(top_n)
    # Using a professional colormap
    sns.barplot(x='mean_shap_value', y='feature', data=plot_data, palette='viridis')

    plt.title(f'Top {top_n} Feature Importances (Mean Absolute SHAP)', fontsize=14, fontweight='bold')
    plt.xlabel('Mean |SHAP value| (Impact on Model Output)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'shap_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_csv", default="data/features_quarterly.csv")
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--mode", default="decile", choices=["decile", "long_only", "score_weighted"])
    parser.add_argument("--shap", action="store_true", help="Compute SHAP explanations")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Train & Predict
    preds, model, scaler, feat_cols = train_and_predict(args.features_csv)

    # Save Predictions
    pred_path = os.path.join(args.out_dir, "oos_predictions.csv")
    preds.to_csv(pred_path, index=False)
    print(f"\nSaved OOS predictions to {pred_path}")

    # 2. Backtest
    res = run_backtest(preds, mode=args.mode)

    bt_path = os.path.join(args.out_dir, f"backtest_{args.mode}.csv")
    res.to_csv(bt_path, index=False)
    print(f"Saved backtest results to {bt_path}")

    # 3. Report Metrics
    metrics = calc_metrics(res)
    print("-" * 30)
    print(f"BACKTEST RESULTS ({args.mode.upper()})")
    print("-" * 30)
    for k, v in metrics.items():
        print(f"{k:<20}: {v}")
    print("-" * 30)

    # ---> NEW: Generate performance plots <---
    plot_backtest_results(res, args.out_dir, args.mode)

    # 4. Save Model Artifacts
    with open(os.path.join(args.out_dir, "final_model.pkl"), "wb") as f:
        pickle.dump((model, scaler, feat_cols), f)

    # 5. SHAP (Optional)
    if args.shap and HAS_SHAP:
        print("\nComputing SHAP feature importance...")
        try:
            # Use a random sample of background data for speed
            # Load fresh data to ensure we have the raw values
            df_full = pd.read_csv(args.features_csv)
            # Filter to numeric features only
            X_raw = df_full[feat_cols].fillna(0)

            # Downsample for speed (SHAP is slow on large datasets)
            if len(X_raw) > 2000:
                X_sample = X_raw.sample(2000, random_state=42)
            else:
                X_sample = X_raw

            X_scaled = scaler.transform(X_sample)

            if HAS_LGB and isinstance(model, LGBMRegressor):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_scaled)

                # Mean Absolute SHAP
                mean_abs = np.abs(shap_values).mean(axis=0)

                shap_df = pd.DataFrame({
                    'feature': feat_cols,
                    'mean_shap_value': mean_abs
                }).sort_values('mean_shap_value', ascending=False)

                shap_path = os.path.join(args.out_dir, "shap_summary.csv")
                shap_df.to_csv(shap_path, index=False)
                print(f"Saved SHAP summary to {shap_path}")

                print("Top 5 Features:")
                print(shap_df.head(5))

                # ---> NEW: Generate SHAP visualization <---
                plot_shap_importance(shap_df, args.out_dir)

        except Exception as e:
            print(f"SHAP calculation failed: {e}")