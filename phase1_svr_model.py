"""
Phase A: SVR Model Feature Engineering
Creates a supervised dataset for one-step-ahead variance forecasting
and writes the feature dataset to outputs/SVR_OUTPUT/svr_feature_dataset.csv

Follows the user's instructions exactly for target and lagged features.
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import json
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
try:
    import yfinance as yf
    HAS_YFINANCE = True
except Exception:
    HAS_YFINANCE = False

# Optional ARCH package for GARCH features
try:
    from arch import arch_model
    HAS_ARCH = True
except Exception:
    HAS_ARCH = False

# No CSV fallback: this script now requires yfinance and downloads directly from Yahoo
OUT_DIR = Path("outputs/SVR_OUTPUT")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUT_DIR / "svr_run_log.txt"

def load_and_prep(source) -> pd.DataFrame:
    """
    Load and prepare data. `source` can be a Path to a CSV or a pandas DataFrame already
    containing at least a Date/Price column. Returns a DataFrame indexed by Date with
    Price, LogReturn, SqReturn and optional India_VIX columns.
    """
    if isinstance(source, pd.DataFrame):
        df_full = source.copy()
    else:
        raise ValueError("load_and_prep expects a pandas DataFrame (downloaded from yfinance).")
    df = df_full.copy()

    # Expect a date column and an underlying price column (as in prior work)
    # Try common names
    date_col = None
    price_col = None
    for c in df.columns:
        if c.lower() in ("date", "trade_date", "timestamp"):
            date_col = c
        if c.lower() in ("underlying value", "underlying_value", "price", "close"):
            price_col = c

    if date_col is None:
        # fallback: first column as date
        date_col = df.columns[0]
    if price_col is None:
        # fallback: second column
        price_col = df.columns[1]

    # Prepare DataFrame with Price and preserve extra columns (e.g., VIX) if available
    cols_to_keep = [date_col, price_col]
    # detect a VIX-like column
    vix_col = None
    for c in df_full.columns:
        if 'vix' in c.lower() or 'india_vix' in c.lower() or 'india vix' in c.lower():
            vix_col = c
            cols_to_keep.append(c)
            break

    df = df_full[cols_to_keep].rename(columns={date_col: "Date", price_col: "Price"}).copy()
    if vix_col:
        df = df.rename(columns={vix_col: 'India_VIX'})
    df["Date"] = pd.to_datetime(df["Date"])  
    df = df.drop_duplicates(subset=["Date"])  
    df = df.sort_values("Date").set_index("Date")

    # Ensure numeric price
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Price"])  

    # Log returns
    df["LogReturn"] = np.log(df["Price"]).diff()

    # Squared returns
    df["SqReturn"] = df["LogReturn"] ** 2

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # Target: next day's variance proxy = (LogReturn^2) shifted -1 (i.e., target is next day's squared return)
    df = df.copy()
    # Create percent-scaled squared returns to match GARCH (returns scaled by 100)
    df["SqReturn_pct"] = (df["LogReturn"] * 100) ** 2
    df["Target_Variance"] = df["SqReturn_pct"].shift(-1)

    # Create 5 lagged squared returns: Sq_Return_Lag_1 ... Sq_Return_Lag_5
    for i in range(1, 6):
        col = f"Sq_Return_Lag_{i}"
        df[col] = df["SqReturn_pct"].shift(i)

    # Create 5 lagged log returns: Return_Lag_1 ... Return_Lag_5
    for i in range(1, 6):
        col = f"Return_Lag_{i}"
        df[col] = df["LogReturn"].shift(i)

    # Drop rows with NaNs introduced by shifting
    df_final = df.dropna(subset=["Target_Variance"] + [f"Sq_Return_Lag_{i}" for i in range(1,6)] + [f"Return_Lag_{i}" for i in range(1,6)])

    # If India_VIX present, include 1-lag VIX as a feature
    if 'India_VIX' in df_final.columns:
        df_final['India_VIX_Lag_1'] = df_final['India_VIX'].shift(1)
        df_final = df_final.dropna(subset=['India_VIX_Lag_1'])
        feature_cols = [f"Sq_Return_Lag_{i}" for i in range(1,6)] + [f"Return_Lag_{i}" for i in range(1,6)] + ['India_VIX_Lag_1']
    else:
        feature_cols = [f"Sq_Return_Lag_{i}" for i in range(1,6)] + [f"Return_Lag_{i}" for i in range(1,6)]

    # Define feature matrix X and target y
    X = df_final[feature_cols].copy()
    y = df_final["Target_Variance"].copy()

    # For convenience, return a DataFrame with features + target and Date index
    out_df = pd.concat([X, y], axis=1)
    return out_df


def main():
    # Prefer downloading from Yahoo Finance directly using the same defaults as Phase 1.
    default_ticker = '^NSEI'
    default_start = '2022-01-01'
    default_end = '2025-08-20'

    df = None
    with open(LOG_PATH, "a") as logf:
        logf.write("\n=== SVR Feature Engineering run ===\n")

    used_source = None
    if HAS_YFINANCE:
        try:
            with open(LOG_PATH, "a") as logf:
                logf.write(f"Attempting to download {default_ticker} and ^INDIAVIX from Yahoo Finance ({default_start} to {default_end})...\n")

            nifty_raw = yf.download(default_ticker, start=default_start, end=default_end, progress=False, auto_adjust=False)
            vix_raw = None
            try:
                vix_raw = yf.download('^INDIAVIX', start=default_start, end=default_end, progress=False, auto_adjust=False)
            except Exception:
                vix_raw = None


            # Extract price series (prefer Adj Close then Close) as a 1-D Series
            if nifty_raw is None or nifty_raw.empty:
                raise RuntimeError("Nifty download returned no data")
            price_series = None
            for col in ['Adj Close', 'Close']:
                if col in nifty_raw.columns:
                    price_series = nifty_raw[col].copy()
                    break
            if price_series is None:
                raise RuntimeError('No price column found in Nifty download')
            price_series.index = pd.to_datetime(price_series.index)
            price_series.name = 'Price'

            # Extract VIX as 1-D Series if present
            vix_series = None
            if vix_raw is not None and not vix_raw.empty:
                for col in ['Adj Close', 'Close']:
                    if col in vix_raw.columns:
                        vix_series = vix_raw[col].copy()
                        break
                if vix_series is not None:
                    vix_series.index = pd.to_datetime(vix_series.index)
                    vix_series.name = 'India_VIX'

            # Build full business-day index and reindex each series to that index
            idx = pd.bdate_range(start=default_start, end=default_end)
            merged = pd.DataFrame(index=idx)
            merged['Price'] = price_series.reindex(idx)
            if vix_series is not None:
                merged['India_VIX'] = vix_series.reindex(idx)

            # Forward-fill to cover non-trading days
            merged['Price'] = merged['Price'].ffill()
            if 'India_VIX' in merged.columns:
                merged['India_VIX'] = merged['India_VIX'].ffill()

            # Verify we have valid price values
            if merged['Price'].isna().all():
                raise RuntimeError('Price series empty after reindex/ffill')

            with open(LOG_PATH, "a") as logf:
                logf.write(f"Downloaded Nifty rows: {len(price_series)}; merged range after reindex: {merged.index.min()} to {merged.index.max()} (rows={len(merged)})\n")

            df_input = merged.reset_index().rename(columns={'index': 'Date'})
            df = load_and_prep(df_input)
            used_source = 'yfinance'
            with open(LOG_PATH, "a") as logf:
                logf.write("Downloaded data and prepared DataFrame from yfinance.\n")
        except Exception as e:
            with open(LOG_PATH, "a") as logf:
                logf.write(f"yfinance download/prep failed: {e}\n")

    # If we couldn't prepare df from yfinance, fail explicitly (no CSV fallback)
    if df is None:
        if not HAS_YFINANCE:
            with open(LOG_PATH, "a") as logf:
                logf.write("yfinance not available; this script requires yfinance and direct download from Yahoo Finance.\n")
            raise RuntimeError("yfinance not available; install yfinance or run the original CSV-based SVR script.")
        else:
            with open(LOG_PATH, "a") as logf:
                logf.write("yfinance download succeeded but data preparation failed; aborting (no CSV fallback).\n")
            raise RuntimeError("Failed to prepare data from yfinance; aborting.")

    # Log final dataframe date range for debugging
    try:
        with open(LOG_PATH, 'a') as logf:
            logf.write(f"Final prepared df index range: {df.index.min()} to {df.index.max()} (rows={len(df)}) using source={used_source}\n")
        print(f"Data prepared: {df.index.min()} to {df.index.max()} (rows={len(df)}) source={used_source}")
    except Exception:
        pass

    # (CSV fallback removed) data must be provided via yfinance download

    # Run stationarity tests and log results
    try:
        lr = df['LogReturn'].dropna()
        sr = df['SqReturn'].dropna()
        adf_lr = adfuller(lr)
        adf_sr = adfuller(sr)
        with open(LOG_PATH, 'a') as logf:
            logf.write(f"ADF LogReturn: stat={adf_lr[0]:.4f}, p={adf_lr[1]:.4g}\n")
            logf.write(f"ADF SqReturn: stat={adf_sr[0]:.4f}, p={adf_sr[1]:.4g}\n")
    except Exception as e:
        with open(LOG_PATH, 'a') as logf:
            logf.write(f"ADF tests failed: {e}\n")

    features_df = build_features(df)

    out_csv = OUT_DIR / "svr_feature_dataset.csv"
    features_df.to_csv(out_csv)

    # --- Step 2: Train/Test split and scaling (use exact split index from GARCH backtest) ---
    # Prepare X and y
    X = features_df.drop(columns=["Target_Variance"]).copy()
    y = features_df["Target_Variance"].copy()

    # Define train/test split proportionally (use 80% train / 20% test)
    split_index = int(0.8 * len(X))

    if split_index <= 0 or split_index >= len(X):
        # fallback to 80% split if the explicit index is invalid for this dataset
        split_index = int(0.8 * len(X))

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()
    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

    # Feature scaling: fit StandardScaler on training data only
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Do NOT scale the target variable y. SVR is sensitive to feature scaling but
    # scaling the target (especially refitting a scaler inside the rolling loop)
    # causes the flatline behavior. We'll use raw y for tuning and training.
    y_train_unscaled = y_train.values.ravel()

    # Save scaled datasets and scaler for downstream modeling
    X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

    X_train_scaled_csv = OUT_DIR / "X_train_scaled.csv"
    X_test_scaled_csv = OUT_DIR / "X_test_scaled.csv"
    y_train_csv = OUT_DIR / "y_train.csv"
    y_test_csv = OUT_DIR / "y_test.csv"
    scaler_pkl = OUT_DIR / "scaler.pkl"

    X_train_scaled_df.to_csv(X_train_scaled_csv)
    X_test_scaled_df.to_csv(X_test_scaled_csv)
    y_train.to_csv(y_train_csv)
    y_test.to_csv(y_test_csv)

    with open(scaler_pkl, "wb") as f:
        pickle.dump(scaler, f)

    summary = []
    summary.append(f"Rows after preprocessing (with LogReturn): {len(df)}\n")
    summary.append(f"Rows after feature engineering (X/y): {len(features_df)}\n")
    summary.append(f"Feature columns: {list(features_df.columns.drop('Target_Variance'))}\n")
    summary.append(f"Saved feature dataset to: {out_csv}\n")
    summary.append(f"Train/Test split index used: {split_index} (train rows: {len(X_train)}, test rows: {len(X_test)})\n")
    summary.append(f"Saved X_train_scaled: {X_train_scaled_csv}\n")
    summary.append(f"Saved X_test_scaled: {X_test_scaled_csv}\n")
    summary.append(f"Saved y_train: {y_train_csv}\n")
    summary.append(f"Saved y_test: {y_test_csv}\n")
    summary.append(f"Saved scaler pickle to: {scaler_pkl}\n")

    with open(LOG_PATH, "a") as logf:
        logf.writelines(summary)

    # Also print key info to stdout for convenience
    print("SVR feature engineering completed.")
    print("Saved:", out_csv)
    print("Rows after feature engineering:", len(features_df))
    print("Train rows:", len(X_train), "Test rows:", len(X_test))

    # --- Step 3: Hyperparameter tuning with TimeSeriesSplit (RandomizedSearchCV) ---
    try:
        # Define parameter grid (as requested)
        param_grid = {
            'kernel': ['rbf'],
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 1],
            'epsilon': [0.01, 0.05, 0.1]
        }

        tscv = TimeSeriesSplit(n_splits=5)
        svr = SVR()

        rand_search = RandomizedSearchCV(
            estimator=svr,
            param_distributions=param_grid,
            n_iter=20,
            scoring='neg_root_mean_squared_error',
            cv=tscv,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        # Fit on training data only (use unscaled y)
        rand_search.fit(X_train_scaled, y_train_unscaled)

        best_params = rand_search.best_params_
        best_score = rand_search.best_score_

        # Save best params and model
        best_params_path = OUT_DIR / "svr_best_params.json"
        with open(best_params_path, "w") as f:
            json.dump({"best_params": best_params, "best_score": best_score}, f, indent=2)

        best_model_pkl = OUT_DIR / "svr_best_model.pkl"
        with open(best_model_pkl, "wb") as f:
            pickle.dump(rand_search.best_estimator_, f)

        # Save cv results
        cv_results_df = pd.DataFrame(rand_search.cv_results_)
        cv_results_csv = OUT_DIR / "svr_cv_results.csv"
        cv_results_df.to_csv(cv_results_csv, index=False)

        tune_summary = []
        tune_summary.append(f"Saved RandomizedSearchCV best params to: {best_params_path}\n")
        tune_summary.append(f"Saved best model pickle to: {best_model_pkl}\n")
        tune_summary.append(f"Saved CV results to: {cv_results_csv}\n")

        with open(LOG_PATH, "a") as logf:
            logf.writelines(tune_summary)

        print("Hyperparameter tuning completed. Best score (neg RMSE):", best_score)
        print("Best params:", best_params)
    except Exception as e:
        msg = f"Hyperparameter tuning failed: {e}\n"
        with open(LOG_PATH, "a") as logf:
            logf.write(msg)
        print(msg)

    # --- Step 4: Rolling one-step-ahead forecasts (refit each day) ---
    # Initialize best_params from file if available, else fallback to defaults
    try:
        if 'best_params' not in locals():
            # try load from saved json
            best_params_path = OUT_DIR / "svr_best_params.json"
            if best_params_path.exists():
                with open(best_params_path, "r") as f:
                    bp = json.load(f)
                    best_params = bp.get("best_params", None)
            else:
                best_params = None
    except Exception:
        best_params = None

    if best_params is None:
        # sensible defaults
        best_params = {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'epsilon': 0.1}

    # Prepare for rolling forecasts
    forecasts = []
    forecast_index = []

    # We'll iterate over test set positions: for i from 0 to len(X_test)-1
    for i in range(len(X_test)):
        # global index for the row to predict
        train_end = split_index + i  # training uses rows up to train_end-1; we will use [:train_end]

        # current training features and target
        X_train_curr = X.iloc[:train_end].copy()
        y_train_curr = y.iloc[:train_end].copy()

        # feature row for the day to predict (the row at position train_end)
        x_pred_row = X.iloc[train_end:train_end+1].copy()

        # If there are not enough training rows, skip
        if len(X_train_curr) < 5:
            forecasts.append(np.nan)
            forecast_index.append(X.iloc[train_end:train_end+1].index[0])
            continue

        # Refit scaler on current training features only
        scaler_curr = StandardScaler()
        scaler_curr.fit(X_train_curr)

        X_train_curr_scaled = scaler_curr.transform(X_train_curr)
        x_pred_scaled = scaler_curr.transform(x_pred_row)

        # Use raw (unscaled) y for training the SVR. Features are scaled.
        y_train_curr_raw = y_train_curr.values.ravel()

        # Initialize and fit SVR with best params
        model_curr = SVR(**best_params)
        try:
            model_curr.fit(X_train_curr_scaled, y_train_curr_raw)
            # prediction is already in original variance units
            pred = float(model_curr.predict(x_pred_scaled)[0])
        except Exception as e:
            # If fitting fails, record NaN and continue
            pred = np.nan
            with open(LOG_PATH, "a") as logf:
                logf.write(f"SVR refit failed at step {i} (train_end={train_end}): {e}\n")

        forecasts.append(pred)
        forecast_index.append(x_pred_row.index[0])

    # Consolidate results into DataFrame aligned with test index
    svr_forecasts_df = pd.DataFrame({"SVR_Forecast_Variance": forecasts}, index=forecast_index)

    svr_forecasts_csv = OUT_DIR / "svr_rolling_forecasts.csv"
    svr_forecasts_df.to_csv(svr_forecasts_csv)

    with open(LOG_PATH, "a") as logf:
        logf.write(f"Saved rolling SVR forecasts to: {svr_forecasts_csv}\n")

    # --- Visualization: compare SVR forecasts to actual realized variance (time-series only) ---
    try:
        import matplotlib.pyplot as plt

        # Build results frame: align forecasts with actual target (y_test)
        results = svr_forecasts_df.join(y.rename('Actual_Variance'), how='left')

        # Compute annualized volatility for plotting (convert percent^2 -> vol)
        def var_to_annual_vol(var_series):
            vol_daily = np.sqrt(var_series.clip(lower=0)) / 100.0
            return vol_daily * np.sqrt(252)

        results['Actual_Vol_Annual'] = var_to_annual_vol(results['Actual_Variance'])
        results['SVR_Forecast_Vol_Annual'] = var_to_annual_vol(results['SVR_Forecast_Variance'])

        # Time series plot: only this plot is kept
        plt.figure(figsize=(12, 5))
        plt.plot(results.index, results['Actual_Vol_Annual'], label='Actual (Annualized)', color='tab:blue')
        plt.plot(results.index, results['SVR_Forecast_Vol_Annual'], label='SVR Forecast (Annualized)', color='tab:orange')
        plt.title('SVR: Actual vs Forecast (Annualized Volatility)')
        plt.xlabel('Date')
        plt.ylabel('Annualized Volatility')
        plt.legend()
        plt.tight_layout()
        ts_png = OUT_DIR / 'svr_actual_vs_forecast.png'
        plt.savefig(ts_png, dpi=150)
        plt.close()

        with open(LOG_PATH, "a") as logf:
            logf.write(f"Saved SVR time-series plot to: {ts_png}\n")
    except Exception as e:
        with open(LOG_PATH, "a") as logf:
            logf.write(f"Could not create SVR time-series plot: {e}\n")

    # Diagnostic: detect flatline forecasts
    try:
        unique_preds = np.unique(np.array(forecasts)[~np.isnan(forecasts)])
        if len(unique_preds) == 1:
            with open(LOG_PATH, 'a') as logf:
                logf.write(f"Warning: SVR produced a flatline forecast with single unique value: {unique_preds[0]}\n")
    except Exception:
        pass

    print("Saved rolling SVR forecasts to:", svr_forecasts_csv)

    # --- Additional Diagnostics: scatter, correlation, decile RMSE ---
    try:
        from scipy.stats import pearsonr, spearmanr
        # Align forecasts with actual target (y_test)
        results = svr_forecasts_df.join(y.rename('Actual_Variance'), how='left')

        # Drop NaNs for diagnostics
        diag = results.dropna(subset=['Actual_Variance', 'SVR_Forecast_Variance']).copy()

        if len(diag) > 0:
            # Overall RMSE and correlation
            from sklearn.metrics import mean_squared_error
            rmse_overall = float(np.sqrt(mean_squared_error(diag['Actual_Variance'], diag['SVR_Forecast_Variance'])))
            try:
                pr, pp = pearsonr(diag['Actual_Variance'], diag['SVR_Forecast_Variance'])
            except Exception:
                pr, pp = (np.nan, np.nan)

            # Decile RMSE: compute RMSE of forecast in bins of actual variance
            diag['actual_decile'] = pd.qcut(diag['Actual_Variance'].rank(method='first'), 10, labels=False) + 1
            decile_stats = []
            for d in sorted(diag['actual_decile'].unique()):
                sub = diag[diag['actual_decile'] == d]
                if len(sub) > 0:
                    rmse_d = float(np.sqrt(mean_squared_error(sub['Actual_Variance'], sub['SVR_Forecast_Variance'])))
                else:
                    rmse_d = np.nan
                decile_stats.append({'decile': int(d), 'rmse': rmse_d, 'n': len(sub)})
            decile_df = pd.DataFrame(decile_stats)

            # Scatter plot (variance scale)
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 6))
            plt.scatter(diag['Actual_Variance'], diag['SVR_Forecast_Variance'], alpha=0.6, s=20)
            mn = min(diag['Actual_Variance'].min(), diag['SVR_Forecast_Variance'].min())
            mx = max(diag['Actual_Variance'].max(), diag['SVR_Forecast_Variance'].max())
            plt.plot([mn, mx], [mn, mx], color='red', linestyle='--', label='45°')
            plt.xlabel('Actual Variance (percent^2)')
            plt.ylabel('SVR Forecast Variance (percent^2)')
            plt.title(f'SVR: Actual vs Forecast (variance) — RMSE={rmse_overall:.4f}, Pearson={pr:.3f}')
            plt.legend()
            scatter_png = OUT_DIR / 'svr_scatter_actual_vs_forecast_variance.png'
            plt.tight_layout()
            plt.savefig(scatter_png, dpi=150)
            plt.close()

            # Save decile RMSE
            decile_csv = OUT_DIR / 'svr_decile_rmse.csv'
            decile_df.to_csv(decile_csv, index=False)

            # Log diagnostics
            with open(LOG_PATH, 'a') as logf:
                logf.write(f"SVR diagnostics: overall RMSE={rmse_overall:.6f}, Pearson_r={pr:.4f} (p={pp:.4g})\n")
                logf.write(f"Saved scatter plot to: {scatter_png}\n")
                logf.write(f"Saved decile RMSE to: {decile_csv}\n")

            print('SVR diagnostics saved:', scatter_png, decile_csv)
    except Exception as e:
        with open(LOG_PATH, 'a') as logf:
            logf.write(f"SVR diagnostics failed: {e}\n")

    # --- Retrain SVR to predict sqrt(variance) (vol) and produce squared forecasts ---
    try:
        # Prepare arrays and rolling forecasting same as before but on sqrt target
        forecasts_vol = []
        forecast_index_vol = []

        # We'll reuse best_params determined earlier
        for i in range(len(X_test)):
            train_end = split_index + i
            X_train_curr = X.iloc[:train_end].copy()
            y_train_curr = y.iloc[:train_end].copy()
            x_pred_row = X.iloc[train_end:train_end+1].copy()

            if len(X_train_curr) < 5:
                forecasts_vol.append(np.nan)
                forecast_index_vol.append(X.iloc[train_end:train_end+1].index[0])
                continue

            scaler_curr = StandardScaler()
            scaler_curr.fit(X_train_curr)
            X_train_curr_scaled = scaler_curr.transform(X_train_curr)
            x_pred_scaled = scaler_curr.transform(x_pred_row)

            # Transform target: vol_percent = sqrt(variance)
            y_train_curr_vol = np.sqrt(np.maximum(y_train_curr.values.ravel(), 0.0))

            model_curr = SVR(**best_params)
            try:
                model_curr.fit(X_train_curr_scaled, y_train_curr_vol)
                pred_vol = float(model_curr.predict(x_pred_scaled)[0])
                # ensure non-negative vol
                if np.isnan(pred_vol):
                    pred_var = np.nan
                else:
                    pred_vol_clipped = max(pred_vol, 0.0)
                    pred_var = pred_vol_clipped ** 2
            except Exception as e:
                pred_var = np.nan
                with open(LOG_PATH, 'a') as logf:
                    logf.write(f"SVR(sqrt-target) refit failed at step {i}: {e}\n")

            forecasts_vol.append(pred_var)
            forecast_index_vol.append(x_pred_row.index[0])

        svr_forecasts_sqrt_df = pd.DataFrame({'SVR_Forecast_Variance_sqrttrained': forecasts_vol}, index=forecast_index_vol)
        svr_forecasts_sqrt_csv = OUT_DIR / 'svr_rolling_forecasts_sqrttrained.csv'
        svr_forecasts_sqrt_df.to_csv(svr_forecasts_sqrt_csv)
        with open(LOG_PATH, 'a') as logf:
            logf.write(f"Saved SVR(sqrt-target) rolling forecasts to: {svr_forecasts_sqrt_csv}\n")
        print('Saved SVR(sqrt-target) rolling forecasts to:', svr_forecasts_sqrt_csv)

        # Quick evaluation: align with actuals and save scatter
        eval_df = svr_forecasts_sqrt_df.join(y.rename('Actual_Variance'), how='left').dropna()
        if len(eval_df) > 0:
            from sklearn.metrics import mean_squared_error
            rmse_sqrt = float(np.sqrt(mean_squared_error(eval_df['Actual_Variance'], eval_df['SVR_Forecast_Variance_sqrttrained'])))
            # scatter (variance scale)
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6,6))
            plt.scatter(eval_df['Actual_Variance'], eval_df['SVR_Forecast_Variance_sqrttrained'], alpha=0.6, s=20)
            mn = min(eval_df['Actual_Variance'].min(), eval_df['SVR_Forecast_Variance_sqrttrained'].min())
            mx = max(eval_df['Actual_Variance'].max(), eval_df['SVR_Forecast_Variance_sqrttrained'].max())
            plt.plot([mn,mx],[mn,mx], color='red', linestyle='--')
            plt.xlabel('Actual Variance (percent^2)')
            plt.ylabel('SVR(sqrt) Forecast Variance (percent^2)')
            plt.title(f'SVR(sqrt) Actual vs Forecast — RMSE={rmse_sqrt:.4f}')
            scatter2 = OUT_DIR / 'svr_sqrt_scatter_actual_vs_forecast_variance.png'
            plt.tight_layout(); plt.savefig(scatter2, dpi=150); plt.close()

            with open(LOG_PATH, 'a') as logf:
                logf.write(f"SVR(sqrt-target) RMSE (variance scale) = {rmse_sqrt:.6f}\n")
                logf.write(f"Saved SVR(sqrt) scatter to: {scatter2}\n")
    except Exception as e:
        with open(LOG_PATH, 'a') as logf:
            logf.write(f"SVR sqrt-target retrain/forecast failed: {e}\n")
        print('SVR sqrt-target retrain failed:', e)


if __name__ == "__main__":
    main()
