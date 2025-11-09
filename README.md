# 🔬 GARCH Monte Carlo Equity Explorer

A professional Streamlit tool for Monte Carlo simulations of stock prices using GARCH(1,1) volatility modeling.

Find a demo on https://equity-explorer.streamlit.app/

## 📋 Overview

This tool combines **GARCH(1,1)** (Generalized Autoregressive Conditional Heteroskedasticity) with **Geometric Brownian Motion (GBM)** to create realistic stock price simulations with time-varying volatility.

### Key Features

- ✅ **GARCH(1,1) Volatility Modeling**: Captures volatility clustering and time-varying market conditions
- ✅ **Monte Carlo Simulations**: Thousands of potential price paths into the future
- ✅ **Risk Analysis**: Value-at-Risk (VaR) and Expected Shortfall (ES/CVaR)
- ✅ **Rolling Statistics**: 30- and 60-day rolling drift and volatility
- ✅ **P/E Ratio Adjustment**: Optionally integrate fundamental valuation into drift
- ✅ **Interactive Visualizations**: Plotly fan charts, histograms, and more
- ✅ **CSV Export**: Export all simulated paths for further analysis

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip

### Install Dependencies

```bash
pip install streamlit yfinance pandas numpy plotly arch scipy
```

Or with a `requirements.txt`:

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
streamlit>=1.28.0
yfinance>=0.2.28
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
arch>=6.2.0
scipy>=1.11.0
```

## 💻 Usage

### Start the App

```bash
streamlit run streamlit.py
```

The app will automatically open in your browser at `http://localhost:8501`

### Workflow

1. **Select Ticker**: Enter a Yahoo Finance ticker symbol (e.g., AAPL, MSFT, TSLA)
2. **Define Time Period**: Choose start and end dates for historical data
3. **Configure Simulation**:
   - **Horizon (Days)**: How far into the future to simulate (e.g., 252 days = 1 year)
   - **Steps per Path**: Time resolution of the simulation
   - **Number of Paths**: More paths = more accurate statistics, but longer computation time
   - **Random Seed**: For reproducible results
4. **Optional**: Enable P/E Ratio Adjustment
5. **Click "🚀 Load Data & Simulate"**

## 📊 Tabs and Visualizations

### Tab 1: Fan Chart & Paths
- Historical price development (black)
- 50 randomly selected simulation paths (transparent)
- Quantile ribbons (5%-95%, 10%-90%, 25%-75%)
- Median path (blue dashed)
- Red vertical line marks the horizon date

### Tab 2: Risk Analysis (VaR/ES)
- **Value-at-Risk (VaR)**: Maximum expected loss at a given confidence level
- **Expected Shortfall (ES)**: Average loss in the tail-risk scenario
- Return distribution histogram with VaR/ES markings

### Tab 3: Distribution of End Prices
- Histogram of simulated end prices after the horizon
- Current price as a reference line
- Quantile table (1%, 5%, 10%, 25%, 50%, 75%, 90%, 95%, 99%)

### Tab 4: Rolling Statistics
- 30-day and 60-day rolling drift (μ)
- 30-day and 60-day rolling volatility (σ)
- Historical averages as reference lines
- Current values as metrics

### Tab 5: Data Export
- CSV download of all simulated paths
- Summary statistics
- Data format: Date × Paths

## 🔧 Technical Details

### GARCH(1,1) Model

The GARCH(1,1) model predicts the conditional variance:

```
h_t+1 = ω + α·ε²_t + β·h_t
```

Where:
- **ω** (omega): Constant
- **α** (alpha): Weight of past shocks
- **β** (beta): Weight of past volatility
- **h_t**: Conditional variance at time t

**Advantages:**
- Captures volatility clustering
- Time-varying volatility estimation
- More realistic forecasts than constant volatility

### GBM with GARCH Volatility

```python
S_t+1 = S_t × exp((μ - 0.5σ²_t)·dt + σ_t·√dt·Z_t)
```

Where:
- **S_t**: Price at time t
- **μ**: Drift (annualized expected return)
- **σ_t**: GARCH-predicted volatility (time-varying!)
- **dt**: Time step size
- **Z_t**: Standard normally distributed random variable

### P/E Ratio Adjustment

When enabled:
```
μ_adjusted = μ_historical × adjustment_factor
```

Where:
```
adjustment_factor = clip(sector_PE / stock_PE, 0.5, 1.5)
```

- Low P/E → Higher expected return (undervalued)
- High P/E → Lower expected return (overvalued)

## 📈 Example Use Cases

### 1. Risk Management
- Calculate VaR for portfolio positions
- Stress-testing with various horizon periods
- Expected Shortfall for tail-risk management

### 2. Options Valuation (indicative)
- Understanding potential price movements
- Implied volatility estimation
- Scenario planning for options strategies

### 3. Investment Planning
- Visualize long-term price targets
- Probability of different outcomes
- Understand risk-return profiles

### 4. Research & Education
- Learn GARCH modeling
- Understand Monte Carlo simulations
- Explore time-varying volatility

## 🎯 Parameter Recommendations

| Use Case         | Horizon | Steps  | Paths   | Seed |
|------------------|---------|--------|---------|------|
| Quick Check      | 30-60   | 30-60  | 500     | 42   |
| Daily Analysis    | 252     | 252    | 1000-2000 | 42   |
| Detailed Study    | 252-504 | 252-504| 5000+   | 42   |
| Research         | 504+    | 504+   | 10000+  | variable |

**Computation Time** (approximately):
- 1000 paths × 252 steps: ~2-5 seconds
- 5000 paths × 252 steps: ~10-20 seconds
- 10000 paths × 504 steps: ~1-2 minutes

## ⚠️ Important Notes

### Limitations

- **No Dividends**: Model ignores dividend payments
- **No Splits**: Stock splits are not considered
- **Normal Distribution**: GBM assumes log-normal returns (no fat tails perfectly modeled)
- **Constant Drift**: μ remains constant (no regime switches)
- **Historical Basis**: GARCH is based on historical data

### Risk Disclaimer

⚠️ **IMPORTANT**: This tool is intended solely for **educational and research purposes**.

- ❌ **NO INVESTMENT ADVICE**
- ❌ **NO GUARANTEE** for future results
- ❌ **NO RECOMMENDATION** to buy/sell securities

Past performance is not indicative of future results. All simulations are purely illustrative.

## 🐛 Known Issues

### Pandas Timestamp Errors
**Problem**: Timestamp arithmetic errors may occur with older Pandas versions.

**Solution**: Upgrade to Pandas ≥2.0
```bash
pip install --upgrade pandas
```

### GARCH Fitting Failures
**Problem**: GARCH may fail with too short time series or extremely volatile data.

**Solution**:
- Use longer historical periods (min. 1 year)
- Ensure data is complete

### Memory Issues with Many Paths
**Problem**: >10000 paths may cause RAM issues.

**Solution**:
- Reduce the number of paths
- Increase available RAM
- Run simulations in batches

## 🔄 Updates & Versions

### Version 1.0 (Current)
- ✅ GARCH(1,1) volatility modeling
- ✅ Monte Carlo GBM simulations
- ✅ VaR & Expected Shortfall
- ✅ Rolling statistics
- ✅ P/E adjustment
- ✅ Interactive Plotly charts
- ✅ CSV export

### Planned Features
- [ ] Regime-switching models
- [ ] Jump-diffusion for extreme events
- [ ] Portfolio simulations (multivariate)
- [ ] GARCH variants (EGARCH, GJR-GARCH)
- [ ] Macro indicator integration

## 🤝 Contribution

This tool is an educational project. Suggestions for improvements and feedback are welcome!

## 📄 License

This project is freely available for educational and research purposes.


---

**Built with**: Streamlit, yfinance, arch, plotly, pandas, numpy
**Data Source**: Yahoo Finance
**⚠️ No Investment Advice**

