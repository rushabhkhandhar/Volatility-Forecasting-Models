# Volatility Forecasting Model: SVR vs. GARCH

This project implements and compares two approaches for forecasting financial volatility: a traditional econometric model (GARCH) and a machine learning model (Support Vector Regression - SVR).

## Project Overview

The goal is to predict the realized variance of an asset (e.g., S&P 500) using historical price data. The project compares the predictive performance of the SVR model against the GARCH benchmark.

### Key Features
- **Data Acquisition:** Automated downloading of OHLC data from Yahoo Finance.
- **Feature Engineering:** Calculation of Log Returns and Squared Returns.
- **Modeling:**
  - **GARCH:** GJR-GARCH(1,1) model using the `arch` library.
  - **SVR:** Support Vector Regression with RBF kernel using `scikit-learn`.
- **Evaluation:** Comparison using RMSE, MAE, Diebold-Mariano test, and Mincer-Zarnowitz regression.

## Installation & Usage

### Prerequisites
- Python 3.8+
- Required packages: `numpy`, `pandas`, `scikit-learn`, `statsmodels`, `arch`, `yfinance`, `matplotlib`, `seaborn`

### Running the Project
1.  **Data Prep & EDA:**
    ```bash
    python phase1_data_prep_eda.py
    ```
2.  **Train SVR Model:**
    ```bash
    python phase1_svr_model.py
    ```
3.  **Run GARCH Model:**
    (Ensure you run the GARCH script located in `S&P500/garch.py` or equivalent if separate)
4.  **Compare Results:**
    ```bash
    python phase1_compare_svr_garch.py
    ```

## Project Structure
- `phase1_data_prep_eda.py`: Data fetching and exploratory analysis.
- `phase1_svr_model.py`: SVR model training and forecasting.
- `phase1_compare_svr_garch.py`: Statistical comparison of model outputs.
- `outputs/`: Directory containing generated CSVs, logs, and plots.

---

 Objective
- **Goal:** Forecast the daily variance (volatility) of financial assets.
- **Problem:** Volatility is latent (not directly observable) and clusters over time.
- **Approach:** Compare a data-driven Machine Learning approach (SVR) against a standard industry benchmark (GARCH).

Methodology
- **Data Source:** Yahoo Finance (Daily OHLC data).
- **Target Variable:** Realized Variance (Proxied by Squared Log Returns).
- **Data Processing:**
  - Calculation of Log Returns.
  - Feature scaling (StandardScaler).
  - Time-series splitting for validation.

 Models Implemented
- **Baseline Model: GARCH (GJR-GARCH)**
  - Captures leverage effects (asymmetry in volatility response to shocks).
  - Standard econometric tool for volatility modeling.
- **Challenger Model: Support Vector Regression (SVR)**
  - Kernel-based learning algorithm.
  - Maps inputs to high-dimensional feature space.
  - Robust to outliers via the epsilon-insensitive loss function.

 SVR Model Configuration
- **Kernel:** Radial Basis Function (RBF) - captures non-linear relationships.
- **Hyperparameters (Optimized):**
  - `C` (Regularization): 1
  - `epsilon` (Margin of tolerance): 0.01
  - `gamma`: Scale
- **Features Used:** Lagged Log Returns, Lagged Squared Returns.

 Performance Comparison (Key Results)
*Lower values indicate better performance.*

| Metric | GARCH (Benchmark) | SVR (Proposed) | Improvement |
| :--- | :--- | :--- | :--- |
| **RMSE** (Root Mean Sq Error) | 0.590 | **0.484** | **~18%** |
| **MAE** (Mean Absolute Error) | 0.439 | **0.294** | **~33%** |

**Observation:** The SVR model consistently achieves lower error rates compared to the GARCH model.

 Statistical Significance
- **Test Used:** Diebold-Mariano (DM) Test.
- **Hypothesis:** Tests if the difference in predictive accuracy is significant.
- **Result:**
  - **DM Statistic:** 2.33
  - **p-value:** 0.021
- **Conclusion:** Since p < 0.05, we reject the null hypothesis. The SVR model's superior performance is **statistically significant**.

 Conclusion
- **Summary:** SVR outperforms GARCH in forecasting daily variance for the tested period.
- **Key Findings:**
  - Machine Learning models can capture complex volatility dynamics better than rigid parametric models.
  - SVR reduced the Mean Absolute Error by approximately 33%.
- **Future Work:** Explore LSTM/GRU (Deep Learning) models and incorporate implied volatility (VIX) as a feature.
