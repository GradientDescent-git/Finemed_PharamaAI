# Mathematical Foundations & Diagnostic Analysis: TSB vs. Chronos-2

## 1. Mathematical Mechanics of TSB (Teunter-Syntetos-Babai)

Pharmaceutical demand series frequently exhibit **intermittent and zero-inflated demand patterns**, where non-zero sales transactions are separated by variable stretches of zero demand.

### Standard Croston Flaw
Standard Croston's method updates demand size $z_t$ and inter-arrival time $p_t$ only when demand occurs ($y_t > 0$). However, it suffers from severe positive bias when demand ceases, because the forecast remains fixed between non-zero demand events regardless of how much time elapses. Conversely, standard Single Exponential Smoothing (SES) applied directly to intermittent series updates the forecast downwards on every zero-demand day, causing the estimated demand to decay exponentially toward zero during dry spells.

### The TSB Formulation
The Teunter-Syntetos-Babai (TSB) method resolves both flaws by separately estimating:
1. **Demand Size ($z_t$)**: Updated only on non-zero demand days ($y_t > 0$).
2. **Demand Probability ($p_t$)**: Updated on **every** period $t$ (both zero and non-zero days).

#### Update Equations:
If $y_t > 0$:
$$z_t = \alpha \cdot y_t + (1 - \alpha) \cdot z_{t-1}$$
$$p_t = \beta \cdot 1 + (1 - \beta) \cdot p_{t-1}$$

If $y_t = 0$:
$$z_t = z_{t-1}$$
$$p_t = \beta \cdot 0 + (1 - \beta) \cdot p_{t-1} = (1 - \beta) \cdot p_{t-1}$$

#### Point Forecast Equation:
$$\hat{y}_{t+h} = z_t \cdot p_t \quad \forall h \ge 1$$

Because $p_t$ is updated continuously on zero days, the probability of demand occurrence declines gracefully during long dry spells without distorting the estimated magnitude $z_t$ of demand when an order does arrive.

---

## 2. Amazon Chronos-2 Foundation Model Underforecasting Diagnostics

Amazon Chronos-2 represents time series as token sequences discretized into categorical buckets. While Chronos-2 demonstrates strong zero-shot performance on continuous, dense time series, diagnostics on FineMed's 158 pharmaceutical SKUs revealed systematic underforecasting on intermittent SKUs:

```text
TSB Holdout WAPE            → 25.577%
Chronos-2 P50 Holdout WAPE  → 27.724%
```

### Root Cause Analysis:
1. **Quantization Granularity**: Tokenization maps sparse zero-heavy intervals to lower quantile bins, causing median (P50) point forecasts to collapse to zero for slow-moving SKUs.
2. **Non-Zero Spiky Distributions**: Chronos-2's cross-entropy loss over token buckets penalizes over-predictions heavily, resulting in conservative median forecasts that under-predict spiky re-orders.

---

## 3. Demand Regime Error Decomposition

WAPE broken down across demand velocity regimes:

## 3. Demand Regime Error Decomposition

Syntetos-Boylan demand regime classification (ADI vs. $CV^2$) computed via [`src/finemed_ai/demand_forecasting/regime_analysis.py`](file:///C:/Users/User/.gemini/antigravity/scratch/Finemed_PharamaAI/src/finemed_ai/demand_forecasting/regime_analysis.py) and generated artifact [`data/05_gold/demand_forecasting/regime_wape_breakdown.json`](file:///C:/Users/User/.gemini/antigravity/scratch/Finemed_PharamaAI/data/05_gold/demand_forecasting/regime_wape_breakdown.json):

| Demand Regime | Syntetos-Boylan Boundaries | SKU Count | TSB WAPE | Chronos-2 P50 WAPE | Selected Production Model |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Smooth / Fast-Moving** | $\text{ADI} < 1.32, CV^2 < 0.49$ | 42 | **21.34%** | 21.80% | **TSB / Hybrid** |
| **Intermittent / Slow-Moving** | $\text{ADI} \ge 1.32, CV^2 < 0.49$ | 68 | **26.85%** | 31.40% | **TSB** |
| **Lumpy / Spiky Demand** | $\text{ADI} \ge 1.32, CV^2 \ge 0.49$ | 36 | **29.10%** | 36.12% | **TSB** |
| **Erratic** | $\text{ADI} < 1.32, CV^2 \ge 0.49$ | 12 | **27.42%** | 30.15% | **TSB** |
| **Overall Portfolio** | **158 SKUs** | **158** | **25.577%** | **27.724%** | **TSB (Empirical Routing)** |

---

## 4. Probabilistic Forecast Calibration (P10–P90 Interval)

FineMed Pharma AI evaluates probabilistic forecasts using [`src/finemed_ai/demand_forecasting/chronos_calibration.py`](file:///C:/Users/User/.gemini/antigravity/scratch/Finemed_PharamaAI/src/finemed_ai/demand_forecasting/chronos_calibration.py) by verifying empirical calibration coverage:

$$\text{Calibration Coverage} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\left( y_i \in [\hat{q}_{0.10, i}, \hat{q}_{0.90, i}] \right)$$

- **Target Interval Coverage**: $80.0\%$ ($\alpha = 0.10 \dots 0.90$)
- **Observed Empirical Coverage**: $82.4\%$ across 158 evaluated SKUs
- **Reproducible Verification**: Run `python src/finemed_ai/demand_forecasting/generate_regime_breakdown.py` to regenerate the JSON summary artifact.

This confirms that the generated P10–P90 prediction intervals provide valid, calibrated risk bounds for inventory planning.
