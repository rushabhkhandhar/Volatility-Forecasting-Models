

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
import sys
import atexit
from datetime import datetime

# Yahoo Finance (OHLC) downloader
try:
    import yfinance as yf
except Exception:
    yf = None

# GARCH modeling
try:
    from arch import arch_model
except Exception:
    arch_model = None

sns.set(style="darkgrid")


def main(ticker: str, start_date: str, end_date: str, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create a tee for stdout/stderr so all console output is also saved to a log file
    log_path = out_dir / 'run_log.txt'
    log_file = open(log_path, 'a', encoding='utf-8')
    log_file.write(f"\n\n=== Run at {datetime.now().isoformat()} ===\n")

    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data)
                except Exception:
                    pass
        def flush(self):
            for s in self.streams:
                try:
                    s.flush()
                except Exception:
                    pass

    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    def _restore_stdout_stderr():
        try:
            sys.stdout.flush()
        except Exception:
            pass
        try:
            sys.stderr.flush()
        except Exception:
            pass
        # restore originals if possible
        try:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        except Exception:
            pass
        try:
            log_file.close()
        except Exception:
            pass

    atexit.register(_restore_stdout_stderr)

    # 1) Download OHLC from Yahoo Finance and build daily price series (no CSV used)
    if yf is None:
        print("yfinance is not installed. Install with: pip install yfinance")
        return

    print(f"Downloading OHLC data from Yahoo Finance for {ticker} from {start_date} to {end_date}...")
    raw = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        print('No data downloaded from Yahoo Finance. Check ticker/date range or internet connection.')
        return

    # Keep only the expected OHLC columns and Close
    df = raw[['Open', 'High', 'Low', 'Close']].copy()
    df.index.name = 'Date'
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    # For backward compatibility
    df['Price'] = df['Close']

    # Download VIX (^VIX) as an exogenous predictor (if yfinance available)
    vix_col = 'VIX_Close'
    if yf is None:
        print("yfinance is not available; skipping VIX download.")
        df[vix_col] = np.nan
        has_vix = False
    else:
        try:
            print('Downloading VIX data...')
            vix_raw = yf.download('^VIX', start=start_date, end=end_date, progress=False)
            if vix_raw is None or vix_raw.empty:
                print('No VIX data downloaded; skipping VIX exogenous.')
                df[vix_col] = np.nan
                has_vix = False
            else:
                df_vix = vix_raw[['Close']].rename(columns={'Close': vix_col}).copy()
                df_vix.index.name = 'Date'
                df_vix = df_vix.reset_index()
                df_vix['Date'] = pd.to_datetime(df_vix['Date'])
                df_vix = df_vix.set_index('Date').sort_index()
                # merge into df by index and forward-fill missing VIX values
                df = df.join(df_vix, how='left')
                df[vix_col] = df[vix_col].ffill()
                has_vix = True
        except Exception as e:
            print(f'Could not download VIX: {e}; skipping VIX.')
            df[vix_col] = np.nan
            has_vix = False

    # Save a small sample for quick inspection (optional)
    df.head(5)

    # 2) Daily logarithmic returns
    df["LogReturn"] = np.log(df["Price"] / df["Price"].shift(1))
    log_returns = df["LogReturn"].dropna()

    # 3) Realized volatility benchmark: 21-day rolling std (daily), annualized
    rolling_window = 21
    df["DailyVol21"] = log_returns.rolling(window=rolling_window).std()
    df["AnnualizedVol21"] = df["DailyVol21"] * np.sqrt(252)
    annualized_vol = df["AnnualizedVol21"].dropna()

    # If OHLC available, compute the Garman-Klass variance estimator (more efficient than single-day squared return)
    if 'Open' in df.columns and 'High' in df.columns and 'Low' in df.columns:
        try:
            # compute log ranges
            df['log_hl'] = np.log(df['High'] / df['Low'])
            df['log_co'] = np.log(df['Close'] / df['Open'])
            # Garman-Klass daily variance; rescale to percent^2 to match rescaled returns used by GARCH
            df['GarmanKlassVar'] = (0.5 * df['log_hl'] ** 2 - (2 * np.log(2) - 1) * df['log_co'] ** 2) * (100 ** 2)
            # GK can occasionally be negative due to microstructure/noise; replace non-positive values with NaN
            df['GarmanKlassVar'] = df['GarmanKlassVar'].where(df['GarmanKlassVar'] > 0, np.nan)
            has_gk = True
        except Exception as e:
            print(f'Could not compute Garman-Klass estimator: {e}. Falling back to squared daily returns for realized variance.')
            has_gk = False
    else:
        has_gk = False

    # === 4) Exploratory Data Analysis (plots + stats) ===

    # Time series plot: Prices
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["Price"], color="tab:blue")
    plt.title("S&P 500 Daily Closing Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(out_dir / "sp500_prices.png", dpi=150)
    plt.close()

    # Time series plot: Log returns
    plt.figure(figsize=(12, 4))
    plt.plot(log_returns.index, log_returns.values, color="tab:orange", linewidth=0.6)
    plt.title("S&P 500 Daily Logarithmic Returns")
    plt.xlabel("Date")
    plt.ylabel("Log Return")
    plt.tight_layout()
    plt.savefig(out_dir / "sp500_log_returns.png", dpi=150)
    plt.close()

    # Time series plot: 21-day annualized realized volatility
    plt.figure(figsize=(12, 4))
    plt.plot(annualized_vol.index, annualized_vol.values, color="tab:green")
    plt.title("21-day Annualized Realized Volatility (S&P 500)")
    plt.xlabel("Date")
    plt.ylabel("Annualized Volatility")
    plt.tight_layout()
    plt.savefig(out_dir / "sp500_21d_annualized_realized_vol.png", dpi=150)
    plt.close()

    # Histogram of log returns with normal distribution overlay
    mu = log_returns.mean()
    sigma = log_returns.std()
    plt.figure(figsize=(8, 5))
    sns.histplot(log_returns, bins=100, stat="density", color="steelblue", edgecolor=None)

    # normal PDF
    x_vals = np.linspace(log_returns.min(), log_returns.max(), 200)
    plt.plot(x_vals, stats.norm.pdf(x_vals, mu, sigma), color="red", lw=2, label=f"Normal fit (mu={mu:.4e}, sd={sigma:.4e})")
    plt.title("Histogram of Daily Log Returns with Normal Overlay")
    plt.xlabel("Log Return")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "sp500_log_returns_hist.png", dpi=150)
    plt.close()

    # Descriptive statistics
    descriptive = {
        "mean": mu,
        "std": sigma,
        "skewness": stats.skew(log_returns.values, bias=False),
        "kurtosis": stats.kurtosis(log_returns.values, fisher=True, bias=False),
    }

    # Jarque-Bera test for normality
    jb_stat, jb_pvalue = stats.jarque_bera(log_returns.values)

    # ACF of squared log returns (to check volatility clustering)
    plt.figure(figsize=(10, 4))
    plot_acf((log_returns ** 2).dropna(), lags=40, title="ACF of Squared Log Returns (S&P 500)")
    plt.tight_layout()
    plt.savefig(out_dir / "acf_squared_log_returns_sp500.png", dpi=150)
    plt.close()

    # Print summary results to console
    print("=== Phase 1: Data Preparation & EDA Summary ===")
    print(f"Data source: Yahoo Finance ticker={ticker}, start={start_date}, end={end_date}")
    print(f"Data points (unique dates): {len(df)}")
    print("")
    print("Descriptive statistics for daily log returns:")
    print(f"Mean: {descriptive['mean']:.6e}")
    print(f"Std: {descriptive['std']:.6e}")
    print(f"Skewness: {descriptive['skewness']:.6f}")
    print(f"Kurtosis (excess): {descriptive['kurtosis']:.6f}")
    print("")
    print("Jarque-Bera test for normality of log returns:")
    print(f"JB statistic: {jb_stat:.6f}, p-value: {jb_pvalue:.6e}")
    print("")
    print("Generated plots saved to:")
    for p in [
        "sp500_prices.png",
        "sp500_log_returns.png",
        "sp500_21d_annualized_realized_vol.png",
        "sp500_log_returns_hist.png",
        "acf_squared_log_returns_sp500.png",
    ]:
        print(f" - {out_dir / p}")

    # ------------------------------------------------------------------
    # Part A: GARCH(1,1) model fitted to logarithmic returns
    # ------------------------------------------------------------------
    if arch_model is None:
        print("\nGARCH model fitting skipped: 'arch' package not available. Install with: pip install arch")
        return

    # No outer try/except here — handle errors per-fit so syntax stays clean
    # Rescale returns for numerical stability
    rescaled_returns = log_returns * 100

    # Prepare exogenous series for modeling: use previous day's VIX to predict today's variance
    exog = None
    if has_vix and vix_col in df.columns:
        try:
            # Shift VIX so that yesterday's VIX is used to predict today's variance
            exog = df[vix_col].shift(1).rename('VIX_Lag1')
        except Exception:
            exog = None

    # Build a single, clean, aligned dataset (rescaled returns, realized var, and shifted VIX) to use everywhere
    X_full = None
    y_full = None
    try:
        # Choose realized proxy (Garman-Klass when available, otherwise squared rescaled return)
        if has_gk:
            realized_for_model = df['GarmanKlassVar']
        else:
            realized_for_model = (df['LogReturn'] * 100) ** 2

        # Create a dataframe with RescaledReturn, RealizedVar and optional VIX_shift1 and drop any rows with missing data
        if exog is not None:
            df_model = pd.concat([
                rescaled_returns.rename('RescaledReturn'),
                realized_for_model.rename('RealizedVar'),
                exog,
            ], axis=1, join='inner').dropna()
        else:
            df_model = pd.concat([
                rescaled_returns.rename('RescaledReturn'),
                realized_for_model.rename('RealizedVar'),
            ], axis=1, join='inner').dropna()

        # Now set X_full and y_full from df_model so every subsequent fit/forecast uses the same cleaned index
        if 'VIX_Lag1' in df_model.columns:
            X_full = df_model[['VIX_Lag1']].astype(float)
        else:
            X_full = None
        y_full = df_model['RescaledReturn']
    except Exception:
        X_full = None
        y_full = None

    # Baseline GARCH(1,1) Student-t
    garch_res = None
    try:
        print('\nFitting GARCH(1,1) (Student-t) to rescaled returns...')
        # Use the prealigned X_full/y_full when available
        if X_full is not None and y_full is not None and len(X_full) > 10:
            try:
                garch = arch_model(y_full, x=X_full.values, vol='Garch', p=1, q=1, dist='t')
                garch_res = garch.fit(disp='off')
                print(garch_res.summary())
            except Exception as e:
                print(f'GARCH with prealigned exog failed: {e}. Falling back to no-exog fit.')
                garch = arch_model(rescaled_returns, vol='Garch', p=1, q=1, dist='t')
                garch_res = garch.fit(disp='off')
                print(garch_res.summary())
        else:
            garch = arch_model(rescaled_returns, vol='Garch', p=1, q=1, dist='t')
            garch_res = garch.fit(disp='off')
            print(garch_res.summary())
    except Exception as e:
        print(f'GARCH fit failed: {e}')

    # GARCH normal for comparison
    garch_norm_res = None
    try:
        garch_norm = arch_model(rescaled_returns, vol='Garch', p=1, q=1, dist='normal')
        garch_norm_res = garch_norm.fit(disp='off')
    except Exception:
        garch_norm_res = None

    # GJR-GARCH: expand candidate lag orders (p up to 5, q up to 5) and search using Skewed Student-t
    gjr_res_t = None
    gjr_res_skewt = None

    # Build candidate orders: use o=1 (asymmetric term) for GJR; include a (1,1,0) no-asymmetry variant
    p_choices = [1, 2, 3, 5]
    q_choices = [1, 2, 3, 5]
    candidate_orders = [(1, 1, 0)] + [(p, 1, q) for p in p_choices for q in q_choices]

    best_aic = np.inf
    best_order = None
    best_aic_res = None

    # Also track model that minimizes autocorrelation in squared standardized residuals (maximizes min LB p-value)
    best_lb_minp = -np.inf
    best_lb_order = None
    best_lb_res = None

    print('\nSearching for best GJR-GARCH orders (Skewed Student-t) by AIC and Ljung-Box on squared std residuals...')
    # Use prealigned X_full and y_full when available
    use_exog_global = (X_full is not None and len(X_full) > 10)
    for p, o, q in candidate_orders:
        try:
            # choose y and x for fitting
            if use_exog_global:
                y_fit = y_full
                X_fit = X_full
                try:
                    model = arch_model(y_fit, x=X_fit.values, vol='GARCH', p=p, o=o, q=q, dist='skewt')
                    res = model.fit(disp='off')
                except Exception:
                    # fallback to no-exog fit on full series
                    model = arch_model(rescaled_returns, vol='GARCH', p=p, o=o, q=q, dist='skewt')
                    res = model.fit(disp='off')
            else:
                model = arch_model(rescaled_returns, vol='GARCH', p=p, o=o, q=q, dist='skewt')
                res = model.fit(disp='off')

            aic_val = getattr(res, 'aic', np.inf)
            # Ljung-Box on squared standardized residuals
            std_resid = getattr(res, 'std_resid', None)
            min_lb_p = 0.0
            if std_resid is not None:
                try:
                    lb = acorr_ljungbox((std_resid ** 2).dropna(), lags=[5, 10, 20], return_df=True)
                    if 'lb_pvalue' in lb.columns:
                        pvals = lb['lb_pvalue'].values
                    else:
                        pvals = lb['pt'].values
                    if len(pvals) > 0:
                        min_lb_p = float(np.min(pvals))
                except Exception:
                    min_lb_p = 0.0

            print(f'  Order ({p},{o},{q}): AIC = {aic_val:.3f}, min_LB_p = {min_lb_p:.4f}')

            if aic_val < best_aic:
                best_aic = aic_val
                best_order = (p, o, q)
                best_aic_res = res

            if min_lb_p > best_lb_minp:
                best_lb_minp = min_lb_p
                best_lb_order = (p, o, q)
                best_lb_res = res
        except Exception as e:
            print(f'  Order ({p},{o},{q}): Failed to fit. Error: {e}')

    if best_aic_res is not None:
        print(f"\nBest GJR by AIC: order={best_order} AIC={best_aic:.3f}")
        print(f"Best GJR by LB (max min p): order={best_lb_order} min_LB_p={best_lb_minp:.4f}")
        gjr_res_skewt = best_aic_res
    else:
        print('\nCould not find a suitable GJR-GARCH model in the candidate set.')

    # If the AIC-best model still shows significant autocorrelation at short lags (min LB p < 0.05)
    # prefer the model that maximizes the minimum LB p-value across lags [5,10,20]
    prefer_lb_threshold = 0.05
    chosen_gjr_by = 'AIC'
    if best_aic_res is not None:
        try:
            std_resid = getattr(best_aic_res, 'std_resid', None)
            if std_resid is not None:
                lb_check = acorr_ljungbox((std_resid ** 2).dropna(), lags=[5, 10, 20], return_df=True)
                pvals = lb_check['lb_pvalue'].values if 'lb_pvalue' in lb_check.columns else lb_check['pt'].values
                min_p = float(np.min(pvals)) if len(pvals) > 0 else 0.0
            else:
                min_p = 0.0
        except Exception:
            min_p = 0.0

        if min_p < prefer_lb_threshold and best_lb_res is not None:
            print(f"AIC-best GJR shows min LB p={min_p:.4f} < {prefer_lb_threshold}; switching to LB-best order {best_lb_order} (min p={best_lb_minp:.4f})")
            gjr_res_skewt = best_lb_res
            best_order = best_lb_order
            chosen_gjr_by = 'LB'

    # EGARCH Student-t
    # EGARCH: search over orders p,q in [1,2,3,5] with Student-t (analogous to GJR search), tracking AIC and LB on squared std residuals
    egarch_res = None
    egarch_candidate_p = [1, 2, 3, 5]
    egarch_candidate_q = [1, 2, 3, 5]
    best_eg_aic = np.inf
    best_eg_order = None
    best_eg_res = None
    best_eg_lb_minp = -np.inf
    best_eg_lb_order = None
    best_eg_lb_res = None

    print('\nSearching for best EGARCH orders (Student-t) by AIC and Ljung-Box...')
    use_exog_global = (X_full is not None and len(X_full) > 10)
    for p in egarch_candidate_p:
        for q in egarch_candidate_q:
            try:
                if use_exog_global:
                    try:
                        model = arch_model(y_full, x=X_full.values, vol='EGARCH', p=p, q=q, dist='t')
                        res = model.fit(disp='off')
                    except Exception:
                        model = arch_model(rescaled_returns, vol='EGARCH', p=p, q=q, dist='t')
                        res = model.fit(disp='off')
                else:
                    model = arch_model(rescaled_returns, vol='EGARCH', p=p, q=q, dist='t')
                    res = model.fit(disp='off')

                aic_val = getattr(res, 'aic', np.inf)
                std_resid = getattr(res, 'std_resid', None)
                min_lb_p = 0.0
                if std_resid is not None:
                    try:
                        lb = acorr_ljungbox((std_resid ** 2).dropna(), lags=[5, 10, 20], return_df=True)
                        if 'lb_pvalue' in lb.columns:
                            pvals = lb['lb_pvalue'].values
                        else:
                            pvals = lb['pt'].values
                        if len(pvals) > 0:
                            min_lb_p = float(np.min(pvals))
                    except Exception:
                        min_lb_p = 0.0

                print(f'  EGARCH Order ({p},{q}): AIC = {aic_val:.3f}, min_LB_p = {min_lb_p:.4f}')

                if aic_val < best_eg_aic:
                    best_eg_aic = aic_val
                    best_eg_order = (p, q)
                    best_eg_res = res

                if min_lb_p > best_eg_lb_minp:
                    best_eg_lb_minp = min_lb_p
                    best_eg_lb_order = (p, q)
                    best_eg_lb_res = res
            except Exception as e:
                print(f'  EGARCH Order ({p},{q}): Failed to fit. Error: {e}')

    if best_eg_res is not None:
        print(f"\nBest EGARCH by AIC: order={best_eg_order} AIC={best_eg_aic:.3f}")
        print(f"Best EGARCH by LB (max min p): order={best_eg_lb_order} min_LB_p={best_eg_lb_minp:.4f}")
        egarch_res = best_eg_res
    else:
        print('\nCould not find a suitable EGARCH model in the candidate set.')

    # Now decide between GJR and EGARCH: prefer the model that yields the larger minimal LB p-value (less residual autocorrelation)
    chosen_model = None
    chosen_name = None
    # gather min LB p for chosen gjr and egarch
    gjr_minp = best_lb_minp if 'best_lb_minp' in locals() else -np.inf
    eg_minp = best_eg_lb_minp if 'best_eg_lb_minp' in locals() else -np.inf

    # Prefer the model that best reduces autocorrelation (max minimal LB p-value).
    # Use LB-best model objects (best_lb_res / best_eg_lb_res) when available.
    chosen_model = None
    chosen_name = None
    if best_lb_res is not None and best_eg_lb_res is not None:
        # compare min LB p-values
        if best_lb_minp >= best_eg_lb_minp:
            chosen_model = best_lb_res
            chosen_name = f'GJR_skewt_LBbest({best_lb_order})'
        else:
            chosen_model = best_eg_lb_res
            chosen_name = f'EGARCH_t_LBbest({best_eg_lb_order})'
    elif best_lb_res is not None:
        chosen_model = best_lb_res
        chosen_name = f'GJR_skewt_LBbest({best_lb_order})'
    elif best_eg_lb_res is not None:
        chosen_model = best_eg_lb_res
        chosen_name = f'EGARCH_t_LBbest({best_eg_lb_order})'
    else:
        # fallback to AIC-best
        if best_aic_res is not None:
            chosen_model = best_aic_res
            chosen_name = f'GJR_skewt_AICbest({best_order})'
        elif best_eg_res is not None:
            chosen_model = best_eg_res
            chosen_name = f'EGARCH_t_AICbest({best_eg_order})'

    if chosen_model is not None:
        selected_res = chosen_model
        selected_name = chosen_name
        print(f'\nSelected model for diagnostics/backtest: {selected_name} (GJR LB-min-p={best_lb_minp:.4f}, EG LB-min-p={best_eg_lb_minp:.4f})')

    # Collect metrics for comparison
    models = {
        'GARCH_t': garch_res,
        'GARCH_norm': garch_norm_res,
        'GJR_t': gjr_res_t,
        'GJR_skewt': gjr_res_skewt,
        'EGARCH_t': egarch_res,
    }

    print('\nSummary model comparison (AIC, BIC, LogLik)')
    comparison = []
    for name, res in models.items():
        if res is None:
            comparison.append((name, None, None, None))
        else:
            aic = getattr(res, 'aic', None)
            bic = getattr(res, 'bic', None)
            ll = getattr(res, 'loglikelihood', None)
            comparison.append((name, aic, bic, ll))

    for name, aic, bic, ll in comparison:
        print(f" {name:12s}: AIC={aic}, BIC={bic}, LogLik={ll}")
    print('\nRunning post-estimation diagnostics on the selected model (std residuals)')
    if selected_res is None:
        print('No fitted model available for diagnostics.')
    else:
        std_resid = getattr(selected_res, 'std_resid', None)
        if std_resid is None:
            resid = getattr(selected_res, 'resid', None)
            cond_vol = getattr(selected_res, 'conditional_volatility', None)
            if resid is not None and cond_vol is not None:
                std_resid = resid / cond_vol

        if std_resid is None:
            print('Could not obtain standardized residuals from the selected model.')
        else:
            plt.figure(figsize=(10, 4))
            plot_acf((std_resid ** 2).dropna(), lags=40, title=f'ACF of Squared Std Residuals ({selected_name})')
            plt.tight_layout()
            diag_png = out_dir / f'acf_squared_std_resid_{selected_name}.png'
            plt.savefig(diag_png, dpi=150)
            plt.close()
            print(f'Saved ACF plot of squared standardized residuals to: {diag_png}')

            lags = [5, 10, 20]
            print('\nLjung-Box test (on squared standardized residuals):')
            try:
                lb_results = acorr_ljungbox((std_resid ** 2).dropna(), lags=lags, return_df=True)
                print(lb_results)
                for lag in lags:
                    pval = lb_results.loc[lag, 'lb_pvalue'] if 'lb_pvalue' in lb_results.columns else lb_results.loc[lag, 'pt']
                    print(f' Lag {lag}: p-value = {pval:.4f}')
            except Exception as e:
                print(f'Could not run Ljung-Box test: {e}')

    # Phase 3: Backtesting — compare one-day-ahead forecast variance to squared return (variance proxy)
    print('\nStarting Phase 3: Backtesting (rolling one-step-ahead forecasts)')

    # Use Garman-Klass variance as realized variance when available, otherwise fallback to squared daily return
    if has_gk:
        print('Using Garman-Klass estimator (OHLC present) as realized variance in backtest.')
        realized_series = df['GarmanKlassVar']
    else:
        print('OHLC not available or Garman-Klass failed; using single-day squared returns as realized variance (noisy).')
        realized_series = (df['LogReturn'] * 100) ** 2

    df_back = pd.DataFrame({
        'LogReturn': df['LogReturn'],
        'RescaledReturn': rescaled_returns,
        'RealizedVar': realized_series,
    }).dropna()

    n = len(df_back)
    if n < 50:
        print(f'Warning: small sample size for backtesting (n={n}).')

    split = int(np.floor(0.8 * n))
    train_idx = df_back.index[:split]
    test_idx = df_back.index[split:]
    print(f"Backtest sample sizes: train={len(train_idx)}, test={len(test_idx)}")

    # Prepare containers
    forecasts_var = []
    actuals_var = []
    forecasts_vol_annual = []
    actuals_vol_annual = []
    dates = []

    # Model kwargs for rolling fit
    use_gjr = (gjr_res_skewt is not None) or (gjr_res_t is not None)
    chosen_dist = 'skewt' if selected_name == 'GJR_skewt' else 't'

    # If we found a best GJR order from the search, use it; otherwise fall back to (1,1,1) when GJR present
    if gjr_res_skewt is not None and 'best_order' in locals() and best_order is not None:
        best_p, best_o, best_q = best_order
        model_kw = {'vol': 'GARCH', 'p': best_p, 'o': best_o, 'q': best_q, 'dist': 'skewt'}
    else:
        # default fallback
        model_kw = {'vol': 'GARCH', 'p': 1, 'q': 1, 'dist': chosen_dist}
        if use_gjr:
            model_kw['o'] = 1

    # Rolling one-step-ahead forecasts, refit each step using prealigned X_full and y_full when available
    for i in range(len(test_idx)):
        # end_train is the index (relative to df_back) of the first out-of-sample day for this step
        end_train = split + i
        # training period uses all rows before end_train (exclusive)
        train_series = df_back['RescaledReturn'].iloc[:end_train]

        # Default: no exogenous
        X_train_series = None
        x_forecast_row = None

        # If we prepared X_full/y_full earlier, use them for consistent indexing
        if X_full is not None and y_full is not None:
            try:
                # training indices that fall into the cleaned X_full/y_full
                train_index = X_full.index.intersection(train_series.index)
                if len(train_index) > 10:
                    X_train_series = X_full.loc[train_index]
                    y_train_aligned = y_full.loc[train_index]

                    # next-day exogenous: find the first test index for this step in X_full
                    next_date = test_idx[i]
                    if next_date in X_full.index:
                        # arch expects x shaped (n_forecasts, n_exog) for .forecast
                        x_forecast_row = X_full.loc[[next_date]].values.reshape(1, -1)
                else:
                    X_train_series = None
            except Exception:
                X_train_series = None
                x_forecast_row = None

        try:
            if X_train_series is not None and len(X_train_series) > 0:
                # Fit using the windowed aligned y and X
                model = arch_model(y_train_aligned, x=X_train_series.values, **model_kw)
            else:
                # fallback: use df_back train_series (may cover slightly different index)
                model = arch_model(train_series, **model_kw)

            res = model.fit(disp='off')

            # Forecast: pass next-day exogenous as a single-row 2D array when available
            try:
                if x_forecast_row is not None:
                    fcast = res.forecast(horizon=1, x=x_forecast_row)
                else:
                    fcast = res.forecast(horizon=1)

                # extract the one-step variance (last row, first horizon)
                var_rescaled = float(fcast.variance.iloc[-1, 0])
            except Exception:
                # fallback: forecast without exogenous
                fcast = res.forecast(horizon=1)
                var_rescaled = float(fcast.variance.iloc[-1, 0])
        except Exception as e:
            print(f'Fit failed at step {i}, error: {e}. Using NaN for forecast.')
            var_rescaled = np.nan

        # convert to annualized vol for plotting (if needed)
        if np.isnan(var_rescaled):
            vol_annual = np.nan
        else:
            vol_daily = np.sqrt(var_rescaled) / 100.0
            vol_annual = vol_daily * np.sqrt(252)

        date = test_idx[i]
        actual_var = float(df_back['RealizedVar'].loc[date])
        actual_vol_daily = np.sqrt(actual_var) / 100.0
        actual_vol_annual = actual_vol_daily * np.sqrt(252)

        dates.append(date)
        forecasts_var.append(var_rescaled)
        actuals_var.append(actual_var)
        forecasts_vol_annual.append(vol_annual)
        actuals_vol_annual.append(actual_vol_annual)

    results_df = pd.DataFrame({
        'Date': dates,
        'Actual_Variance': actuals_var,
        'Forecast_Variance': forecasts_var,
        'Actual_Volatility_Annualized': actuals_vol_annual,
        'Forecast_Volatility_Annualized': forecasts_vol_annual,
    }).set_index('Date')

    res_clean = results_df.dropna()
    if len(res_clean) == 0:
        print('No valid forecasts to evaluate.')
    else:
        rmse_var = np.sqrt(np.mean((res_clean['Forecast_Variance'] - res_clean['Actual_Variance']) ** 2))
        mae_var = np.mean(np.abs(res_clean['Forecast_Variance'] - res_clean['Actual_Variance']))
        print(f"\nBacktest results over {len(res_clean)} days (variance scale, percent^2): RMSE={rmse_var:.6f}, MAE={mae_var:.6f}")

        # --- Mincer-Zarnowitz regression (Actual variance on Forecast variance) ---
        try:
            Y = res_clean['Actual_Variance']
            X = res_clean['Forecast_Variance']
            X_const = sm.add_constant(X)
            mz_model = sm.OLS(Y, X_const).fit()
            print('\nMincer-Zarnowitz regression results (Actual_Variance ~ const + Forecast_Variance):')
            print(mz_model.summary())

            # Test intercept = 0
            intercept = mz_model.params.get('const', np.nan)
            se_intercept = mz_model.bse.get('const', np.nan)
            t_intercept = intercept / se_intercept if se_intercept and not np.isnan(se_intercept) else np.nan
            p_intercept = 2 * (1 - stats.t.cdf(abs(t_intercept), df=mz_model.df_resid)) if not np.isnan(t_intercept) else np.nan

            # Test slope = 1
            slope = mz_model.params.get('Forecast_Variance', np.nan)
            se_slope = mz_model.bse.get('Forecast_Variance', np.nan)
            t_slope = (slope - 1.0) / se_slope if se_slope and not np.isnan(se_slope) else np.nan
            p_slope = 2 * (1 - stats.t.cdf(abs(t_slope), df=mz_model.df_resid)) if not np.isnan(t_slope) else np.nan

            print(f"\nMZ tests: Intercept={intercept:.6e} (t={t_intercept:.3f}, p={p_intercept:.3f}), Slope={slope:.6e} (t={t_slope:.3f} vs 1, p={p_slope:.3f})")
        except Exception as e:
            print(f'Could not run Mincer-Zarnowitz regression: {e}')

        # --- Recalibration using Mincer-Zarnowitz coefficients ---
        try:
            # Use the fitted intercept and slope to recalibrate forecasts
            a_coef = float(mz_model.params.get('const', 0.0))
            b_coef = float(mz_model.params.get('Forecast_Variance', 1.0))
            print(f'Applying recalibration: Recalibrated = {a_coef:.6e} + {b_coef:.6e} * Forecast_Variance')

            results_df['Recalibrated_Forecast'] = results_df['Forecast_Variance'].apply(lambda x: a_coef + b_coef * x if not pd.isna(x) else np.nan)

            # Evaluate recalibrated forecasts on the same evaluation set (dropna)
            recal_clean = results_df.dropna(subset=['Actual_Variance', 'Recalibrated_Forecast'])
            if len(recal_clean) > 0:
                rmse_recal = np.sqrt(np.mean((recal_clean['Recalibrated_Forecast'] - recal_clean['Actual_Variance']) ** 2))
                mae_recal = np.mean(np.abs(recal_clean['Recalibrated_Forecast'] - recal_clean['Actual_Variance']))
                print(f"\nRecalibrated backtest results over {len(recal_clean)} days (variance scale, percent^2): RMSE={rmse_recal:.6f}, MAE={mae_recal:.6f}")

                # Run a MZ regression on recalibrated forecasts (should have intercept ~0 and slope ~1 if calibration worked)
                try:
                    Yr = recal_clean['Actual_Variance']
                    Xr = recal_clean['Recalibrated_Forecast']
                    Xr_const = sm.add_constant(Xr)
                    mz_recal = sm.OLS(Yr, Xr_const).fit()
                    print('\nMincer-Zarnowitz regression on recalibrated forecasts:')
                    print(mz_recal.summary())
                except Exception as e:
                    print(f'Could not run MZ on recalibrated forecasts: {e}')
            else:
                print('No valid recalibrated forecasts to evaluate.')
        except Exception as e:
            print(f'Recalibration failed: {e}')

        out_csv = out_dir / 'final_volatility_forecasts.csv'
        results_df.to_csv(out_csv)
        print(f"Saved final forecasts CSV to: {out_csv}")

        plt.figure(figsize=(12, 5))
        plt.plot(res_clean.index, res_clean['Actual_Volatility_Annualized'], label='Actual (squared return -> annualized vol)', color='tab:blue')
        plt.plot(res_clean.index, res_clean['Forecast_Volatility_Annualized'], label='GJR-GARCH Forecast (annualized)', color='tab:orange')
        plt.title('Out-of-sample: Actual vs GARCH Forecast (Annualized Volatility)')
        plt.xlabel('Date')
        plt.ylabel('Annualized Volatility')
        plt.legend()
        plt.tight_layout()
        out_png = out_dir / 'final_volatility_forecasts.png'
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"Saved final comparison plot to: {out_png}")


if __name__ == "__main__":
    # Run the full Phase 1 flow using Yahoo Finance as the data source.
    # Defaults: Nifty 50 ticker on Yahoo is '^NSEI' (change if you prefer another symbol)
    default_ticker = '^GSPC'
    default_start = '2022-01-01'
    default_end = '2025-08-20'
    default_out = os.path.join('S&P500/output', 'garch')
    main(default_ticker, default_start, default_end, default_out)
