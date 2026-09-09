# Regime-Aware Portfolio Optimization Using Correlation Eigenstructure and Machine Learning

> An end-to-end quantitative finance framework that combines rolling correlation eigenstructure, extensive feature engineering, machine-learning-based market regime classification, and regime-conditioned portfolio optimization.

---

## Overview

Financial markets are not stationary.

The relationships between assets can change substantially across market conditions. During relatively calm periods, assets may provide meaningful diversification benefits. During periods of stress, correlations can increase and a common market component can become dominant across the cross-section.

This project develops a **regime-aware portfolio construction framework** designed to explicitly model this changing dependence structure.

Rather than treating regime detection and portfolio construction as separate problems, the framework connects them:

```text
Prices
   ↓
Log Returns
   ↓
Rolling Correlation Matrix
   ↓
Eigenvalues / Eigenvectors
   ↓
Market & Correlation Features
   ↓
Extensive Feature Engineering
   ↓
Machine Learning
   ↓
Regime Classification
   ↓
Portfolio Optimization
   ↓
Risk & Eigenstructure Constraints
   ↓
Portfolio Weight Generation
```

The central idea is:

> **Use the structure of asset correlations to identify changing market regimes, then allow the predicted regime to influence the portfolio's feasible risk exposure.**

---

# Key Features

* Rolling correlation analysis
* Correlation-matrix eigenvalue decomposition
* Leading eigenvector analysis
* Market volatility and cross-sectional dispersion measures
* Extensive dynamic feature engineering
* Feature-set experimentation and ablation
* Machine-learning regime classification
* Random Forest + XGBoost + LightGBM ensemble
* Same-day regime **nowcasting**
* One-step-ahead regime **forecasting**
* Long-only, fully invested portfolio optimization
* Eigenvector exposure constraint
* Comparison with unconstrained minimum-variance portfolios
* Time-series cross-validation
* Leakage and failure-mode analysis
* Nowcast vs forecast portfolio comparison

---

# 1. Research Motivation

Modern portfolio theory relies heavily on diversification.

If assets are not perfectly correlated, combining them can reduce portfolio variance.

However, diversification is **state-dependent**.

During periods of market stress, correlations can rise and several assets can become driven by the same underlying market factor. A portfolio that appears diversified from an individual-asset perspective may therefore have substantial exposure to a common systemic direction.

This project investigates whether the changing dependence structure of a stock universe can be used to:

1. characterize market conditions,
2. classify market regimes,
3. identify potentially adverse regimes, and
4. modify portfolio construction accordingly.

The project is inspired by research on correlation eigenstructure and regime shifts, particularly the work of **Maksym A. Girnyk**, but the implementation extends the regime-detection mechanism into a multi-feature machine-learning framework and introduces both nowcast and next-day forecast pipelines.

---

# 2. Dataset and Universe

The final stock universe consists of six equities:

| Ticker | Company           |
| ------ | ----------------- |
| AAPL   | Apple             |
| JPM    | JPMorgan Chase    |
| JNJ    | Johnson & Johnson |
| XOM    | Exxon Mobil       |
| WMT    | Walmart           |
| CAT    | Caterpillar       |

Approximately ten years of historical price data are used for the primary dataset.

The main rolling windows are:

| Component           |          Window |
| ------------------- | --------------: |
| Correlation matrix  | 60 trading days |
| Realized volatility | 20 trading days |
| Dynamic changes     |  5 trading days |
| Time-series CV      |         5 folds |

The model uses chronological train/test construction rather than randomly shuffling financial observations.

---

# 3. Mathematical Pipeline

## 3.1 Log Returns

For stock `i` at time `t`, the project calculates log returns:

```text
rᵢ,t = ln(Pᵢ,t / Pᵢ,t-1)
```

Log returns provide the basic input for the rolling correlation and volatility calculations.

---

## 3.2 Market Return

A cross-sectional market return is constructed as the equal-weighted average return across the universe:

```text
r̄ₜ = (1/N) Σ rᵢ,t
```

This provides a simple representation of the common movement of the six-stock universe.

---

## 3.3 Realized Volatility

The framework calculates annualized 20-day realized volatility:

```text
σ₂₀,t = Std(r̄ₜ₋₁₉:ₜ) × √252
```

This captures the recent level of market volatility.

---

## 3.4 Cross-Sectional Dispersion

Cross-sectional dispersion measures how differently the individual stocks are behaving on a given day:

```text
Dₜ = Stdᵢ(rᵢ,t)
```

High dispersion indicates that individual assets are behaving more differently from each other.

---

# 4. Correlation Eigenstructure

One of the central components of the project is the rolling correlation matrix.

Using a 60-day rolling window:

```text
Cₜ = Corr(rₜ₋₅₉, ..., rₜ)
```

The matrix is decomposed as:

```text
Cₜ = Vₜ Λₜ Vₜᵀ
```

where:

* `Λ` contains the eigenvalues
* `V` contains the eigenvectors
* `λ₁ ≥ λ₂ ≥ ... ≥ λₙ`

The project retains the leading eigenvalues and calculates the leading-to-second eigenvalue ratio:

```text
Rₜ = λ₁,t / λ₂,t
```

A larger ratio indicates that the first common direction is relatively dominant compared with the second eigenmode.

---

# 5. Why Eigenvectors Matter

Eigenvalues describe the **strength of common modes**.

Eigenvectors describe the **direction of those modes across the assets**.

The leading eigenvector `v₁` therefore represents the dominant direction of co-movement in the six-dimensional asset space.

This becomes particularly important in the portfolio layer.

Instead of only asking:

> "How much total variance does my portfolio have?"

the framework also asks:

> "How exposed is my portfolio to the dominant systemic direction?"

This creates a direct connection between correlation structure and portfolio risk management.

---

# 6. Feature Engineering

A substantial part of the project is the feature-engineering layer.

The model does not rely on a single volatility indicator or a single eigenvalue threshold.

The candidate feature space includes:

### Market Features

* 20-day realized volatility
* Cross-sectional dispersion
* Rolling skewness
* Downside volatility
* Drawdown
* Return autocorrelation

### Eigenstructure Features

* Leading eigenvalue
* Second eigenvalue
* Third eigenvalue
* Leading/second eigenvalue ratio
* Five-day change in eigenvalue ratio
* Eigenvalue gaps

### Dynamic Features

For a signal `xₜ`, the project calculates:

```text
Δ₅xₜ = xₜ - xₜ₋₅
```

It also calculates rolling slopes and rolling z-scores.

Additional dynamic variables include:

* Volatility-of-volatility
* Volatility/dispersion relationships
* Eigenvalue-ratio dynamics
* Volatility changes
* Dispersion changes

---

# 7. Feature Selection

An important experiment compared different feature-set sizes.

The project tested:

* 8 features
* 15 features
* 19 features

Interestingly, adding more engineered features did **not** automatically improve the model.

The final eight-feature specification generalized better than the larger alternatives, which showed signs of overfitting.

The final feature set consists of:

1. Market realized volatility
2. Market cross-sectional dispersion
3. Leading eigenvalue
4. Second eigenvalue
5. Eigenvalue ratio
6. Five-day eigenvalue-ratio change
7. Five-day volatility change
8. Five-day dispersion change

This was an empirical feature-selection result rather than an assumption that more features are always better.

---

# 8. Market Regime Classification

The framework divides market conditions into three regimes:

```text
CALM
TRANSITIONING
STRESSED
```

The stress proxy is divided using empirical quantiles:

```text
≤ 33rd percentile  → Calm
33rd–66th percentile → Transitioning
> 66th percentile → Stressed
```

The regime classification is then learned using the engineered market and eigenstructure features.

---

# 9. Two ML Pipelines

The project contains two different regime-classification approaches.

## 9.1 Nowcast

The nowcast answers:

> **What regime are we currently in?**

Conceptually:

```text
Xₜ → Regimeₜ
```

The model uses information available for the current day to classify the current regime.

---

## 9.2 Forecast

The forecast answers:

> **What regime are we likely to be in tomorrow?**

Conceptually:

```text
Xₜ → Regimeₜ₊₁
```

The historical target is shifted so that today's features are associated with tomorrow's regime.

The future regime is therefore used as a **training target**, not as an input feature.

This creates a genuine one-step-ahead forecasting setup.

---

# 10. Machine Learning Architecture

The final classifier is a hard-voting ensemble consisting of:

```text
Random Forest
       +
    XGBoost
       +
    LightGBM
       ↓
 Hard Voting Ensemble
       ↓
 Regime Classification
```

Each model produces a regime prediction and the ensemble selects the majority class.

The model is evaluated using:

* Cross-validation
* Held-out test data
* Accuracy
* Precision
* Recall
* F1 score
* Time-series cross-validation

Five expanding time-series folds are used as an additional robustness check.

---

# 11. Classification Results

The reported cross-validated results are approximately:

| Model    | Task            | Cross-Validated Accuracy |
| -------- | --------------- | -----------------------: |
| Nowcast  | Same-day regime |                     ~90% |
| Forecast | Next-day regime |                     ~90% |

The two models answer different questions despite sharing the same underlying feature and classifier architecture.

---

# 12. Portfolio Optimization

Once a regime has been predicted, the framework moves into the portfolio layer.

The baseline portfolio solves a long-only, fully invested minimum-variance problem:

```text
minimize    wᵀΣw

subject to

1ᵀw = 1
wᵢ ≥ 0
```

where:

* `w` = portfolio weight vector
* `Σ` = covariance matrix

This produces the unconstrained minimum-variance portfolio.

---

# 13. Regime-Aware Risk Constraint

The key portfolio innovation is the additional eigenvector exposure constraint.

Let:

```text
v₁ = leading eigenvector
```

Portfolio exposure to the dominant eigenmode is:

```text
E(w) = wᵀv₁
```

During transitioning or stressed regimes, the framework imposes:

```text
|wᵀv₁| ≤ 0.15
```

or equivalently:

```text
-0.15 ≤ wᵀv₁ ≤ 0.15
```

Because this constraint is linear in `w`, the resulting optimization problem remains a convex quadratic program.

---

# 14. Why the Constraint Changes Portfolio Weights

Suppose the unconstrained portfolio has:

```text
|wᵀv₁| > 0.15
```

That portfolio becomes infeasible once the regime-aware constraint is activated.

The optimizer must therefore redistribute the portfolio weights.

It can reduce weights in assets contributing strongly to the dominant eigenvector exposure and increase weights in other assets while still satisfying:

```text
Σwᵢ = 1
```

This means the model is not merely changing portfolio weights incidentally.

It is explicitly controlling exposure to the dominant systemic direction.

---

# 15. Portfolio Optimization Results

In the reported experiments:

| Metric               |     Baseline |  Constrained |
| -------------------- | -----------: | -----------: |
| Eigenvector exposure |      ~−0.396 |       −0.150 |
| Portfolio variance   | ~9.62 × 10⁻⁵ | ~1.14 × 10⁻⁴ |

The constraint successfully pushed the dominant eigenvector exposure toward the prescribed limit.

Example weight changes included:

| Stock | Baseline | Constrained |
| ----- | -------: | ----------: |
| AAPL  |    0.065 |       0.027 |
| JPM   |    0.055 |       0.160 |
| JNJ   |    0.446 |       0.329 |
| XOM   |    0.112 |       0.010 |
| WMT   |    0.285 |       0.242 |
| CAT   |    0.036 |       0.232 |

The increase in portfolio variance is expected: the constrained feasible set is smaller than the unconstrained feasible set.

The framework therefore makes an explicit trade-off:

```text
Lower systemic eigenmode exposure
              ↕
Higher minimum achievable variance
```

---

# 16. September 3 Nowcast vs Forecast Experiment

One of the most interesting experiments compares the two pipelines on the same date.

### Forecast

The forecast model used **September 2 information** to predict the regime for September 3.

Prediction:

```text
TRANSITIONING
```

### Nowcast

The nowcast classified September 3 using current information.

Prediction:

```text
TRANSITIONING
```

Both models therefore produced the same regime classification.

---

# 17. September 3 Portfolio Comparison

The resulting portfolios were extremely similar.

### Nowcast Portfolio

```text
[
0.02691662,
0.15962744,
0.32908587,
0.00976785,
0.24215061,
0.23245160
]
```

### Forecast Portfolio

```text
[
0.02590968,
0.15896371,
0.32977993,
0.00950186,
0.24275456,
0.23309025
]
```

The reported similarity between the two six-dimensional weight vectors was:

```text
R² = 0.9999654945
```

This should be interpreted as **portfolio-weight similarity**, not investment performance or predictive accuracy.

---

# 18. Benchmark: Single Eigenvalue-Ratio Rule

The project also implements a simpler baseline based on a single eigenvalue-ratio threshold.

This provides a useful comparison against the multi-feature ML classifier.

The single-signal rule agreed with the ML-derived regime assignment on approximately:

```text
51% of test observations
```

For a three-class classification problem, random assignment would have an expected agreement of approximately:

```text
33.3%
```

This suggests that the eigenvalue ratio contains useful regime information, but using it alone does not reproduce the richer multi-feature classifier reliably.

---

# 19. Leakage & Failure-Mode Analysis

One of the most important parts of the project was identifying problems in the initial implementation.

## 19.1 VIX Label Leakage

An initial implementation achieved approximately:

```text
99.7% accuracy
```

This appeared extremely strong.

However, the model had access to VIX information that was also involved in constructing the regime labels.

This created **label leakage**.

The result was therefore not considered valid evidence of model performance.

The direct VIX information was removed from the classifier.

---

## 19.2 Universe–Index Mismatch

Another methodological issue was identified.

VIX is derived from a broad S&P 500 options market, while this project operates on a six-stock universe.

Using broad-market VIX directly as the ground truth for a small six-stock universe creates a mismatch between:

```text
Target Market
     vs.
Modeled Universe
```

The project therefore moved toward an in-universe volatility proxy for constructing the regime target.

---

## 19.3 Feature-Set Overfitting

The project also tested larger engineered feature sets.

The larger specifications showed signs of overfitting:

```text
Training accuracy ↑
Test accuracy ↓
```

Reducing the model to eight carefully selected features resulted in better generalization.

---

# 20. Mathematical Validation

Several theoretical relationships were also examined.

The project reports:

### Eigenvalue Relationship

A Pearson correlation of approximately:

```text
−0.62
```

between the first and second eigenvalues.

### Eigenportfolio Validation

The leading eigenportfolio was compared with the empirical market return.

The reported regression produced approximately:

```text
R² ≈ 0.994
β ≈ 1.04
```

This supports the interpretation of the leading eigenvector as a strong representation of the common market direction within the universe.

---

# 21. What the Model Actually Learns

The classifier is not simply learning a single volatility threshold.

It receives information about:

### Market state

* Realized volatility
* Cross-sectional dispersion

### Correlation structure

* Leading eigenvalue
* Second eigenvalue
* Eigenvalue ratio

### Short-term dynamics

* Eigenvalue-ratio changes
* Volatility changes
* Dispersion changes

The model therefore combines:

```text
LEVEL
+
CROSS-SECTIONAL STRUCTURE
+
SHORT-HORIZON DYNAMICS
```

This is the central reason the feature-engineering layer is important to the framework.

---

# 22. Complete Decision Architecture

The complete decision process can be summarized as:

```text
                    PRICE DATA
                        │
                        ▼
                  LOG RETURNS
                        │
                        ▼
             ROLLING CORRELATION
                        │
                        ▼
            EIGENVALUE DECOMPOSITION
                   │           │
                   ▼           ▼
              EIGENVALUES   EIGENVECTORS
                   │           │
                   └─────┬─────┘
                         ▼
              MARKET + STRUCTURE
                    FEATURES
                         │
                         ▼
              EXTENSIVE FEATURE
                  ENGINEERING
                         │
                         ▼
               ML ENSEMBLE MODEL
                         │
                         ▼
                REGIME CLASS
             ┌───────────┼───────────┐
             ▼           ▼           ▼
           CALM    TRANSITIONING   STRESSED
             │           │           │
             │           └─────┬─────┘
             │                 ▼
             │       EIGENVECTOR CONSTRAINT
             │                 │
             └────────┬────────┘
                      ▼
             PORTFOLIO OPTIMIZATION
                      │
                      ▼
              RISK-CONSTRAINED
                PORTFOLIO
                      │
                      ▼
               FINAL WEIGHTS
```

---

# 23. Important Interpretation

The framework is intentionally different from simply building a return-prediction model.

The objective is not:

```text
Features → Predict Returns
```

Instead:

```text
Features
   ↓
Understand Market Regime
   ↓
Determine Appropriate Risk Constraint
   ↓
Change Feasible Portfolio Space
   ↓
Optimize Portfolio
```

The ML component therefore acts as a **regime-detection layer** inside a broader portfolio construction system.

---

# 24. What This Project Does Not Claim

This project does not promote any investement advice.

---

# 25. Future Implementations and Fixes

* The primary universe contains only six stocks.
* The project has also been tested on a 10-stock universe.
* Broader cross-sectional testing is still required.
* The nowcast and forecast implementations currently exist as separate notebooks.
* Portfolio performance requires more rigorous walk-forward validation.
* The framework should be tested across substantially larger universes.

---

# 26. Future Work

Planned improvements include:

### Larger Universe

Expand from six stocks toward a universe of at least 50 stocks.

### Unified System

Combine the nowcast and forecast implementations into a single framework.

### Robust Walk-Forward Testing

Develop a more rigorous calibration procedure that prevents threshold choices from dominating the results.

### Broader Validation

Test the framework across:

* Different market environments
* Different stock universes
* Different time periods
* Different correlation windows
* Different regime definitions

### Portfolio Extensions

Investigate additional regime-aware constraints and alternative portfolio objectives.

---

# 27. Repository Structure

A suggested repository structure is:

```text
Regime-Aware-Portfolio-Optimization/
│
├── README.md
│
├── notebooks/
│   ├── nowcast/
│   └── forecast/
│
├── src/
│   ├── preprocessing/
│   ├── features/
│   ├── regime/
│   └── portfolio/
│
├── results/
│   ├── classification/
│   ├── portfolio/
│   └── experiments/
│
├── reports/
│
└── requirements.txt
```

The principal implementations correspond to the **nowcast** and **forecast** pipelines.

---

# 28. Reproducibility

The project is designed to make the main methodology reproducible.

The repository contains the notebooks and implementation covering:

* Data preprocessing
* Return construction
* Rolling correlation analysis
* Eigenvalue/eigenvector calculations
* Feature engineering
* Regime classification
* Machine-learning models
* Portfolio optimization
* Eigenvector constraints
* Portfolio weight generation
* Main experiments

---

# 29. Technology Stack

The project uses Python and a quantitative finance / machine learning stack including:

* Python
* NumPy
* pandas
* scikit-learn
* XGBoost
* LightGBM
* CVXPY
* Matplotlib
* Financial market data APIs

---

# 30. Key Results at a Glance

| Component                             | Result                  |
| -------------------------------------- | ----------------------- |
| Universe                              | 6 stocks                |
| Historical data                       | ~10 years               |
| Correlation window                    | 60 days                 |
| Final features                        | 8                       |
| Classifier                            | RF + XGBoost + LightGBM |
| Nowcast CV accuracy                   | ~90%                    |
| Forecast CV accuracy                  | ~90%                    |
| Baseline eigenvector exposure         | ~−0.396                 |
| Constrained exposure                  | −0.150                  |
| Sep. 3 regime                         | Transitioning           |
| Nowcast vs forecast weight similarity | R² ≈ 0.999965           |
| Single eigenvalue-ratio agreement     | ~51%                    |

---

# 31. Key Takeaway

The central contribution of this project is the connection between **market structure, machine learning, and portfolio construction**.

The framework treats the correlation matrix not merely as a source of pairwise correlations, but as a structured object whose eigenvalues and eigenvectors can provide information about dominant market modes.

That information is transformed through extensive feature engineering and machine learning into a regime signal.

The regime signal then changes the portfolio optimization problem itself by activating an explicit constraint on exposure to the dominant eigenvector.

In short:

```text
MARKET DATA
     ↓
CORRELATION STRUCTURE
     ↓
EIGENSTRUCTURE
     ↓
FEATURE ENGINEERING
     ↓
MACHINE LEARNING
     ↓
REGIME DETECTION
     ↓
RISK CONSTRAINTS
     ↓
OPTIMIZATION
     ↓
PORTFOLIO WEIGHTS
```

The result is an end-to-end framework in which **regime detection is not the final output — it becomes an input to portfolio risk management.**

---

# Citation

This project was inspired by:

> Girnyk, M. A. (2026). *Correlation Structures and Regime Shifts in Nordic Stock Markets*. Applied Mathematical Finance, 33(1), 46–71.

The project explicitly extends the eigenstructure-based regime mechanism into a multi-feature machine-learning framework and adds a one-step-ahead forecasting pipeline.



---

# Author

**Kriss Khanna**

Quantitative Finance | Machine Learning | Portfolio Optimization

GitHub: [https://github.com/k-means-kriss/](https://github.com/k-means-kriss/)

---

## Project Status

**Research / Experimental**

The core framework has been implemented and evaluated. Future work focuses on larger universes, unified deployment, and more robust walk-forward validation.
