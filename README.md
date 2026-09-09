# Regime-Aware Portfolio Optimization Using Correlation Eigenstructure and Machine Learning-Based Market Regime Forecasting

A regime-aware portfolio optimization framework combining **rolling correlation eigenstructure**, **machine-learning-based market regime classification**, and **regime-dependent portfolio optimization**.

The project develops two complementary pipelines:

- **Nowcast** — identifies the current market regime.
- **Forecast** — predicts the market regime for the next trading day.

The predicted regime is then used to control portfolio exposure to the dominant eigenvector of the rolling correlation structure.

---

## Project Overview

The framework follows the pipeline:

**Market Prices → Log Returns → Rolling Correlation → Eigenstructure → Market Features → ML Regime Classification → Portfolio Optimization**

The core idea is that market correlations can change significantly across market conditions. During stressed periods, assets may become more strongly aligned with a common systemic direction, weakening diversification.

This project therefore uses the eigenstructure of the rolling correlation matrix to identify this dominant direction and incorporates it directly into portfolio optimization.

---

## Main Components

### 1. Rolling Correlation Eigenstructure

A **60-day rolling correlation matrix** is constructed from the selected stock universe.

For each rolling window, the framework extracts:

- Leading eigenvalue
- Second eigenvalue
- Third eigenvalue
- Leading/second eigenvalue ratio
- Leading eigenvector

The leading eigenvector represents the dominant correlation direction across the portfolio universe.

---

### 2. Market Features

The model combines eigenstructure information with market-level and cross-sectional features.

The final classifier uses **8 features**:

1. `market_realized_vol_20d`
2. `market_cross_sectional_dispersion`
3. `leading_eigenvalue`
4. `second_eigenvalue`
5. `eigenvalue_ratio`
6. `eigenvalue_ratio_diff5`
7. `vol_diff5`
8. `dispersion_diff5`

A larger engineered feature set was also tested, but the compact 8-feature specification performed better in the reported experiments.

---

## Machine Learning Regime Classification

The market is classified into three regimes:

| Regime | Meaning |
|---|---|
| `calm` | Lower market stress |
| `transitioning` | Intermediate or changing conditions |
| `stressed` | Elevated market stress |

The classification system uses a **hard-voting ensemble** consisting of:

- Random Forest
- XGBoost
- LightGBM

The models are evaluated using chronological train/test construction and time-series cross-validation.

---

## Nowcast Model

The nowcast pipeline estimates the regime for the **current day**.

Conceptually:

```text
X_t → Regime_t
