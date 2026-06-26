"""
Merge GARCH and SVR forecasts for direct comparison for S&P 500.
Saves merged DataFrame to S&P500/output/COMPARISON_OUTPUT/comparison_final.csv
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


GARCH_CSV = Path("S&P500/output/garch/final_volatility_forecasts.csv")
SVR_CSV = Path("S&P500/output/SVR_OUTPUT/svr_rolling_forecasts.csv")
OUT_DIR = Path("S&P500/output/COMPARISON_OUTPUT")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "compare_run_log.txt"


def read_with_date_index(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    try:
        df = pd.read_csv(path, parse_dates=True, index_col=0)
    except Exception:
        df = pd.read_csv(path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
    # Ensure index is datetime and sorted
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def diebold_mariano(e1, e2, h=1, alternative='two-sided'):
    """
    Simple Diebold-Mariano test for equal predictive accuracy.
    e1/e2 are loss series (e.g., squared errors) from model1 and model2 on same index.
    Returns DM statistic and two-sided p-value.
    """
    d = e1 - e2
    T = len(d)
    if T < 2:
        return np.nan, np.nan
    mean_d = np.mean(d)
    # Newey-West estimator for variance with lag = h-1
    def autocov(x, lag):
        return np.sum((x[:T-lag] - mean_d) * (x[lag:] - mean_d)) / T

    var_d = autocov(d, 0)
    for lag in range(1, h):
        w = 1 - lag / h
        var_d += 2 * w * autocov(d, lag)

    dm_stat = mean_d / np.sqrt(var_d / T) if var_d > 0 else np.nan
    if np.isnan(dm_stat):
        return np.nan, np.nan
    if alternative == 'two-sided':
        p = 2 * (1 - stats.t.cdf(abs(dm_stat), df=T-1))
    elif alternative == 'less':
        p = stats.t.cdf(dm_stat, df=T-1)
    else:
        p = 1 - stats.t.cdf(dm_stat, df=T-1)
    return float(dm_stat), float(p)


def main():
    with open(LOG_PATH, 'a') as logf:
        logf.write('\n=== Comparison run ===\n')
    # Read files
    garch_df = read_with_date_index(GARCH_CSV)
    svr_df = read_with_date_index(SVR_CSV)

    # Identify columns
    # For GARCH file we expect Actual_Variance and Forecast_Variance (or similar)
    garch_cols = {c.lower(): c for c in garch_df.columns}
    if 'actual_variance' in garch_cols:
        actual_col = garch_cols['actual_variance']
    elif 'realizedvar' in garch_cols:
        actual_col = garch_cols['realizedvar']
    elif 'actual variance' in garch_cols:
        actual_col = garch_cols['actual variance']
    else:
        # fallback: try common names
        candidates = ['Actual_Variance', 'RealizedVar', 'Realized_Var', 'ActualVariance', 'RealizedVariance', 'Realized Var']
        actual_col = None
        for cand in candidates:
            if cand in garch_df.columns:
                actual_col = cand
                break
    # Forecast column
    if 'forecast_variance' in garch_cols:
        garch_fore_col = garch_cols['forecast_variance']
    else:
        candidates = ['Forecast_Variance', 'Forecast', 'Forecast_Variance']
        garch_fore_col = None
        for cand in candidates:
            if cand in garch_df.columns:
                garch_fore_col = cand
                break

    if actual_col is None or garch_fore_col is None:
        raise ValueError(f"Could not identify Actual or Forecast columns in GARCH CSV. Found columns: {garch_df.columns}")

    # SVR forecast column
    svr_cols = {c.lower(): c for c in svr_df.columns}
    # Prefer the explicit SVR forecast variance column name, then common variants
    if 'svr_forecast_variance' in svr_cols:
        svr_col = svr_cols['svr_forecast_variance']
    elif 'svr_forecast' in svr_cols:
        svr_col = svr_cols['svr_forecast']
    else:
        # fallback to first numeric column
        svr_col = None
        for c in svr_df.columns:
            if pd.api.types.is_numeric_dtype(svr_df[c]):
                svr_col = c
                break
    if svr_col is None:
        raise ValueError(f"Could not identify SVR forecast column in {SVR_CSV}")

    # Build unified DataFrame
    combined = pd.DataFrame(index=garch_df.index)
    combined['Actual_Variance'] = garch_df[actual_col]
    combined['GARCH_Forecast_Variance'] = garch_df[garch_fore_col]

    # Join SVR forecasts (align by index)
    # If there are overlapping indexes missing in svr_df, join will put NaN
    combined = combined.join(svr_df[[svr_col]].rename(columns={svr_col: 'SVR_Forecast_Variance'}), how='left')

    out_csv = OUT_DIR / 'comparison_final.csv'
    combined.to_csv(out_csv)

    with open(LOG_PATH, 'a') as logf:
        logf.write(f"Saved comparison CSV to: {out_csv}\n")

    print('Saved comparison CSV to:', out_csv)

    # --- Step 2: Compute RMSE and MAE for GARCH and SVR forecasts ---
    # Align and drop NA rows for fair comparison
    comp = combined.dropna(subset=['Actual_Variance', 'GARCH_Forecast_Variance', 'SVR_Forecast_Variance']).copy()
    results = []
    if len(comp) == 0:
        with open(LOG_PATH, 'a') as logf:
            logf.write('No overlapping forecast rows available to compute metrics.\n')
        print('No overlapping rows to compute metrics.')
        return

    # GARCH metrics
    rmse_garch = np.sqrt(mean_squared_error(comp['Actual_Variance'], comp['GARCH_Forecast_Variance']))
    mae_garch = mean_absolute_error(comp['Actual_Variance'], comp['GARCH_Forecast_Variance'])

    # SVR metrics
    rmse_svr = np.sqrt(mean_squared_error(comp['Actual_Variance'], comp['SVR_Forecast_Variance']))
    mae_svr = mean_absolute_error(comp['Actual_Variance'], comp['SVR_Forecast_Variance'])

    metrics_df = pd.DataFrame({
        'Model': ['GARCH (GJR)', 'SVR'],
        'RMSE_variance': [rmse_garch, rmse_svr],
        'MAE_variance': [mae_garch, mae_svr]
    })

    metrics_csv = OUT_DIR / 'metrics_summary.csv'
    metrics_df.to_csv(metrics_csv, index=False)

    # Append summary to log
    with open(LOG_PATH, 'a') as logf:
        logf.write(f"Metrics saved to: {metrics_csv}\n")
        logf.write(metrics_df.to_string(index=False) + "\n")

    print('\nModel performance on overlapping test period:')
    print(metrics_df.to_string(index=False))

    # --- Diebold-Mariano test on squared errors (GARCH vs SVR) ---
    try:
        se_garch = (comp['Actual_Variance'] - comp['GARCH_Forecast_Variance']) ** 2
        se_svr = (comp['Actual_Variance'] - comp['SVR_Forecast_Variance']) ** 2
        dm_stat, dm_p = diebold_mariano(se_garch.values, se_svr.values, h=1, alternative='two-sided')
        dm_out = {'DM_stat': dm_stat, 'DM_pvalue': dm_p, 'loss': 'squared_error', 'horizon': 1}
        dm_df = pd.DataFrame([dm_out])
        dm_csv = OUT_DIR / 'dm_test.csv'
        dm_df.to_csv(dm_csv, index=False)
        with open(LOG_PATH, 'a') as logf:
            logf.write(f"Diebold-Mariano test (squared errors) saved to: {dm_csv}\n")
            logf.write(f"DM stat={dm_stat:.6f}, p-value={dm_p:.6g}\n")
        print(f"\nDiebold-Mariano test (squared errors): DM stat={dm_stat:.6f}, p-value={dm_p:.6g}")
    except Exception as e:
        with open(LOG_PATH, 'a') as logf:
            logf.write(f"Diebold-Mariano test failed: {e}\n")
        print('Diebold-Mariano test failed:', e)

    # --- Step 3: Mincer-Zarnowitz regressions for both models ---
    mz_log = []
    mz_out_path = OUT_DIR / 'mz_regression_summaries.txt'

    def run_mz(actual, forecast, name):
        X = sm.add_constant(forecast)
        model = sm.OLS(actual, X)
        res = model.fit()

        # t-tests: intercept = 0, slope = 1
        # Extract intercept t and p (robust to index types)
        try:
            intercept_t = float(res.tvalues.get('const', res.tvalues.iloc[0]))
            intercept_p = float(res.pvalues.get('const', res.pvalues.iloc[0]))
        except Exception:
            intercept_t = float(res.tvalues.iloc[0])
            intercept_p = float(res.pvalues.iloc[0])

        # Test slope == 1
        # Find the slope variable name (non-const column)
        slope_vars = [c for c in X.columns if c != 'const']
        if len(slope_vars) == 0:
            # fallback: use positional second column
            varname = X.columns[1] if len(X.columns) > 1 else X.columns[0]
        else:
            varname = slope_vars[0]

        try:
            t_test = res.t_test(f"{varname} = 1")
            # t_test.tvalue/pvalue may be arrays; extract scalar
            slope_t = float(t_test.tvalue.flatten()[0])
            slope_p = float(t_test.pvalue.flatten()[0])
        except Exception:
            # fallback compute manually
            slope = float(res.params[varname])
            se = float(res.bse[varname])
            df_resid = int(res.df_resid)
            from scipy import stats
            slope_t = (slope - 1.0) / se
            slope_p = 2 * (1 - stats.t.cdf(abs(slope_t), df_resid))

        mz_log.append(f"MZ regression for {name}:\n")
        mz_log.append(res.summary().as_text() + "\n")
        mz_log.append(f"Test intercept=0: t={intercept_t:.4f}, p={intercept_p:.4g}\n")
        mz_log.append(f"Test slope=1: t={slope_t:.4f}, p={slope_p:.4g}\n\n")

        return res
        return res

    # Run for GARCH
    res_garch = run_mz(comp['Actual_Variance'], comp['GARCH_Forecast_Variance'], 'GARCH (GJR)')
    # Run for SVR
    res_svr = run_mz(comp['Actual_Variance'], comp['SVR_Forecast_Variance'], 'SVR')

    # Save MZ summaries to file
    with open(mz_out_path, 'w') as f:
        f.writelines(mz_log)

    with open(LOG_PATH, 'a') as logf:
        logf.write(f"Saved Mincer-Zarnowitz summaries to: {mz_out_path}\n")

    print('\nSaved Mincer-Zarnowitz regression summaries to:', mz_out_path)

    # Recalibration removed to avoid in-sample data leakage; comparisons use raw forecasts only

    # --- Step 4: Final visual comparison (annualized volatility) ---
    try:
        # Use overlapping period where all three series are present
        vis_df = combined.dropna(subset=['Actual_Variance', 'GARCH_Forecast_Variance', 'SVR_Forecast_Variance']).copy()
        if len(vis_df) == 0:
            with open(LOG_PATH, 'a') as logf:
                logf.write('No overlapping rows for plotting annualized volatility.\n')
            print('No overlapping rows for plotting.')
        else:
            # Helper: clip negative variances before sqrt to avoid runtime warnings
            def safe_annualize(var_series, series_name):
                # ensure float dtype
                arr = var_series.astype(float).to_numpy(copy=True)
                neg_count = int((arr < 0).sum())
                if neg_count > 0:
                    with open(LOG_PATH, 'a') as logf:
                        logf.write(f"Found and clipped {neg_count} negative values in {series_name} before sqrt.\n")
                # clip to zero to avoid invalid sqrt inputs
                arr_clipped = np.maximum(arr, 0.0)
                # annualize
                ann = np.sqrt(arr_clipped) * np.sqrt(252)
                return ann, neg_count

            vis_df['Actual_AnnVol'], n1 = safe_annualize(vis_df['Actual_Variance'], 'Actual_Variance')
            vis_df['GARCH_AnnVol'], n2 = safe_annualize(vis_df['GARCH_Forecast_Variance'], 'GARCH_Forecast_Variance')
            vis_df['SVR_AnnVol'], n3 = safe_annualize(vis_df['SVR_Forecast_Variance'], 'SVR_Forecast_Variance')

            # Log total clipped counts if any
            total_clipped = n1 + n2 + n3
            if total_clipped > 0:
                with open(LOG_PATH, 'a') as logf:
                    logf.write(f"Total negative variance values clipped before plotting: {total_clipped}\n")

            plt.figure(figsize=(10, 5))
            plt.plot(vis_df.index, vis_df['Actual_AnnVol'], label='Actual Annualized Vol', color='black', linewidth=2)
            plt.plot(vis_df.index, vis_df['GARCH_AnnVol'], label='GJR-GARCH Forecast (Annualized)', color='tab:blue')
            plt.plot(vis_df.index, vis_df['SVR_AnnVol'], label='SVR Forecast (Annualized)', color='tab:orange')
            plt.legend()
            plt.title('Out-of-Sample Volatility Forecast Comparison: GJR-GARCH vs. SVR')
            plt.xlabel('Date')
            plt.ylabel('Annualized Volatility (sqrt(variance) * sqrt(252))')
            plt.grid(alpha=0.3)

            plot_path = OUT_DIR / 'comparison_volatility_plot.png'
            plt.tight_layout()
            plt.savefig(plot_path, dpi=200)
            plt.close()

            with open(LOG_PATH, 'a') as logf:
                logf.write(f"Saved annualized volatility comparison plot to: {plot_path}\n")

            print('Saved annualized volatility comparison plot to:', plot_path)
    except Exception as e:
        with open(LOG_PATH, 'a') as logf:
            logf.write(f"Plotting failed: {e}\n")
        print('Plotting failed:', e)


if __name__ == '__main__':
    main()
