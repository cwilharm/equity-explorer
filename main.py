# streamlit_gbm_montecarlo_advanced.py
# Advanced Streamlit app with GARCH, Historical Bootstrap, VaR, Scenario Analysis, and more

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')
from pandas.tseries.offsets import BDay

st.set_page_config(page_title="GARCH Monte Carlo – Equity Explorer", layout="wide")

st.title("🔬 Equity Explorer — GARCH(1,1) Monte‑Carlo Simulationen")
st.markdown(
    """
    Professional tool with **GARCH(1,1)** volatility modeling for time-varying volatility,
    **value-at-risk**, and **expected shortfall** analyses.
    """
)

# ----- Sidebar inputs -----
st.sidebar.header("📊 Inputs")

with st.sidebar:
    ticker = st.text_input("Ticker (Yahoo)", value="AAPL")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=pd.to_datetime("2018-01-01"))
    with col2:
        end_date = st.date_input("End", value=pd.to_datetime("today"))

    st.markdown("### Simulation Parameters")
    horizon_days = st.number_input("Horizon (Days)", min_value=1, value=252)
    n_steps = st.number_input("Steps per Path", min_value=2, value=252)
    n_paths = st.number_input("Number of Paths", min_value=100, value=2000)
    seed = st.number_input("Random Seed (0 = random)", value=42)

    st.markdown("### P/E Drift Adjustment")
    use_pe_adjustment = st.checkbox("Use P/E Ratio for Drift", value=False)

    st.markdown("### VaR Confidence Level")
    var_confidence = st.slider("VaR Confidence", 0.90, 0.99, 0.95, 0.01)

    submit = st.button("🚀 Load Data & Simulate", use_container_width=True)

if not submit:
    st.info("👈 Please set parameters in the sidebar and click 'Load Data & Simulate'.")
    st.stop()

# ----- Fetch data -----
with st.spinner("Loading data from Yahoo Finance..."):
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        ticker_info = yf.Ticker(ticker).info
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

if df.empty:
    st.error("No price data found. Please check ticker or date.")
    st.stop()

prices = df["Close"].squeeze().dropna()
last_price = float(prices.iloc[-1])

# ----- Compute returns, mu, sigma -----
log_returns = np.log(prices / prices.shift(1)).dropna()
simple_returns = prices.pct_change().dropna()

trading_days_per_year = 252
mu_daily = float(log_returns.mean())
sigma_daily = float(log_returns.std())

mu_annual = mu_daily * trading_days_per_year
sigma_annual = sigma_daily * np.sqrt(trading_days_per_year)

# ----- GARCH(1,1) Model -----
def fit_garch_model(returns_values):
    """Fit GARCH(1,1) model to returns"""
    try:
        # Ensure an explicit numeric index (RangeIndex) and convert returns to float array
        arr = np.asarray(returns_values, dtype=float)
        # Scale returns to percentage for better convergence and prevent Timestamp arithmetic inside arch
        returns_series = pd.Series(arr * 100, index=pd.RangeIndex(start=0, stop=len(arr), step=1))
        model = arch_model(returns_series, vol='Garch', p=1, q=1, rescale=False)
        # keep fit call minimal to avoid unexpected kwargs across arch versions
        res = model.fit(disp='off')
        return res
    except Exception as e:
        st.warning(f"GARCH fitting failed: {e}")
        return None

# Fit GARCH model (no caching to avoid timestamp serialization issues)
with st.spinner("Fitting GARCH(1,1) model..."):
    garch_result = fit_garch_model(log_returns.values)

if garch_result is not None:
    try:
        # Extract GARCH parameters and create manual forecast
        # This avoids timestamp arithmetic issues in arch library
        omega = garch_result.params['omega']
        alpha = garch_result.params['alpha[1]']
        beta = garch_result.params['beta[1]']

        # Get last conditional variance
        last_variance = float(garch_result.conditional_volatility.iloc[-1]) ** 2

        # Manual GARCH(1,1) forecast: h_t+1 = omega + alpha * epsilon_t^2 + beta * h_t
        # For multi-step ahead, iterate the forecast to get daily varying volatility
        garch_variance = np.zeros(horizon_days)
        h_t = last_variance
        epsilon_t_sq = float(log_returns.iloc[-1]) ** 2 * 10000  # last squared return in %

        for i in range(horizon_days):
            h_t = omega + alpha * epsilon_t_sq + beta * h_t
            garch_variance[i] = h_t
            epsilon_t_sq = h_t  # For simplicity, use unconditional expectation

        # For display: show average GARCH volatility
        garch_sigma_daily = np.sqrt(np.mean(garch_variance)) / 100  # Convert back from percentage
        garch_sigma_annual = garch_sigma_daily * np.sqrt(trading_days_per_year)

    except Exception as e:
        st.warning(f"GARCH forecast issue: {e}. Using historical volatility.")
        garch_sigma_daily = sigma_daily
        garch_sigma_annual = sigma_annual
        garch_variance = None
else:
    garch_sigma_daily = sigma_daily
    garch_sigma_annual = sigma_annual
    garch_variance = None

# ----- Rolling Statistics -----
rolling_30d_mu = log_returns.rolling(30).mean() * trading_days_per_year
rolling_30d_sigma = log_returns.rolling(30).std() * np.sqrt(trading_days_per_year)
rolling_60d_mu = log_returns.rolling(60).mean() * trading_days_per_year
rolling_60d_sigma = log_returns.rolling(60).std() * np.sqrt(trading_days_per_year)

# ----- P/E Adjustment -----
pe_ratio = ticker_info.get('forwardPE', ticker_info.get('trailingPE', None))
sector_pe = 20  # Typical market P/E
pe_adjustment = 1.0

if use_pe_adjustment and pe_ratio is not None and pe_ratio > 0:
    # If P/E is below average, increase drift (undervalued)
    # If P/E is above average, decrease drift (overvalued)
    pe_adjustment = max(0.5, min(1.5, sector_pe / pe_ratio))

# ----- Display KPIs -----
st.markdown("### 📈 Market Data & Statistics")
col1, col2, col3, col4, col5, col6 = st.columns(6)
last_date_str = pd.Timestamp(prices.index[-1].date()).strftime('%Y-%m-%d')
col1.metric("Last Price", f"${last_price:.2f}", help=f"Date: {last_date_str}")
col2.metric("Annualized Drift (μ)", f"{mu_annual:.2%}")
col3.metric("Historical Vol (σ)", f"{sigma_annual:.2%}")
col4.metric("GARCH Vol", f"{garch_sigma_annual:.2%}")
if pe_ratio:
    col5.metric("P/E Ratio", f"{pe_ratio:.1f}", help=f"Adjustment: {pe_adjustment:.2f}x")
else:
    col5.metric("P/E Ratio", "N/A")
col6.metric("Horizon", f"{horizon_days} Days", help="Simulation period into the future")

st.markdown("---")

# ----- Simulation Function -----
def simulate_garch_gbm(S0, mu, garch_vol_forecast, n_steps, n_paths):
    """GBM with GARCH(1,1) forecasted volatility"""
    T = n_steps / trading_days_per_year
    dt = T / n_steps

    sim_matrix = np.zeros((n_steps + 1, n_paths))
    sim_matrix[0] = S0

    # Use forecasted volatility (cycle if needed)
    vol_forecast = np.tile(garch_vol_forecast, (n_paths, 1)).T

    for t in range(1, n_steps + 1):
        z = np.random.standard_normal(n_paths)
        # Use GARCH volatility
        sigma_t = np.sqrt(vol_forecast[min(t-1, len(vol_forecast)-1), :]) / 100
        sim_matrix[t] = sim_matrix[t - 1] * np.exp((mu - 0.5 * sigma_t ** 2) * dt + sigma_t * np.sqrt(dt) * z)

    return sim_matrix

# ----- Run Simulations -----
T = horizon_days / trading_days_per_year
mu_baseline = mu_annual * pe_adjustment

results = {}

# Set random seed once before all simulations
if seed != 0:
    np.random.seed(int(seed))

# Check if GARCH model fitted successfully
if garch_result is None or garch_variance is None:
    st.error("GARCH(1,1) model could not be fitted. Please check the data or try a different period.")
    st.stop()

with st.spinner("Running GARCH(1,1) + GBM Monte Carlo simulations..."):
    # Run single simulation with GARCH volatility
    sim_baseline = simulate_garch_gbm(last_price, mu_baseline, garch_variance, n_steps, n_paths)
    results['GARCH(1,1) + GBM'] = sim_baseline

# Ensure last_trading_date is a Timestamp at midnight
last_trading_date = pd.to_datetime(prices.index[-1]).normalize()

# Create simulation index using BusinessDay frequency
sim_index = pd.date_range(
    start=last_trading_date,
    periods=int(n_steps) + 1,
    freq=BDay()
)

# Extract key dates
horizon_end_date = sim_index[-1]
today_date = last_trading_date
today_date_dt = today_date.to_pydatetime()

st.info(
    f"📅 **Simulation Period**: {today_date.strftime('%Y-%m-%d')} "
    f"→ **{horizon_end_date.strftime('%Y-%m-%d')}** ({horizon_days} Trading Days)"
)

# ----- VaR and Expected Shortfall -----
def calculate_risk_metrics(final_prices, S0, confidence=0.95):
    """Calculate VaR and Expected Shortfall"""
    returns = (final_prices - S0) / S0
    var_threshold = np.percentile(returns, (1 - confidence) * 100)
    var_price = S0 * (1 + var_threshold)

    # Expected Shortfall (CVaR): average of returns below VaR
    tail_returns = returns[returns <= var_threshold]
    es_threshold = tail_returns.mean() if len(tail_returns) > 0 else var_threshold
    es_price = S0 * (1 + es_threshold)

    return var_price, es_price, var_threshold, es_threshold

# ----- Visualization Tabs -----
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Fan Chart & Szenarios",
    "📉 Risk Analyses (VaR/ES)",
    "📈 Distribution Prices",
    "🔄 Rolling Statistics",
    "📋 Data Export"
])

with tab1:
    st.subheader("Historical Prices & Monte Carlo Fan Chart")

    for method_name, sim_matrix in results.items():
        st.markdown(f"#### {method_name}")
        sim_df = pd.DataFrame(sim_matrix, index=sim_index)

        fig = go.Figure()

        # Historical prices
        fig.add_trace(go.Scatter(
            x=prices.index,
            y=prices.values,
            mode='lines',
            name='Historical',
            line=dict(color='black', width=2)
        ))

        # Sample paths
        n_show = min(50, n_paths)
        for i in range(n_show):
            fig.add_trace(go.Scatter(
                x=sim_df.index,
                y=sim_df.iloc[:, i],
                mode='lines',
                line=dict(width=0.5),
                opacity=0.3,
                showlegend=False,
                hoverinfo='skip'
            ))

        # Quantile ribbons
        q05 = sim_df.quantile(0.05, axis=1)
        q10 = sim_df.quantile(0.10, axis=1)
        q25 = sim_df.quantile(0.25, axis=1)
        q50 = sim_df.quantile(0.50, axis=1)
        q75 = sim_df.quantile(0.75, axis=1)
        q90 = sim_df.quantile(0.90, axis=1)
        q95 = sim_df.quantile(0.95, axis=1)

        # Median
        fig.add_trace(go.Scatter(
            x=sim_df.index,
            y=q50,
            mode='lines',
            name='Median (P50)',
            line=dict(color='blue', width=2, dash='dash')
        ))

        # Ribbons
        fig.add_trace(go.Scatter(
            x=sim_df.index.tolist() + sim_df.index[::-1].tolist(),
            y=q95.tolist() + q05[::-1].tolist(),
            fill='toself',
            fillcolor='rgba(0,100,200,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=True,
            name='5%-95% Quantile'
        ))

        fig.add_trace(go.Scatter(
            x=sim_df.index.tolist() + sim_df.index[::-1].tolist(),
            y=q90.tolist() + q10[::-1].tolist(),
            fill='toself',
            fillcolor='rgba(0,100,200,0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=True,
            name='10%-90% Quantile'
        ))

        fig.add_trace(go.Scatter(
            x=sim_df.index.tolist() + sim_df.index[::-1].tolist(),
            y=q75.tolist() + q25[::-1].tolist(),
            fill='toself',
            fillcolor='rgba(0,100,200,0.25)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=True,
            name='25%-75% Quantile'
        ))

        fig.update_layout(
            height=500,
            xaxis_title='Datum',
            yaxis_title='Preis ($)',
            template='plotly_white',
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader(f"📉 Value-at-Risk (VaR) & Expected Shortfall @ {var_confidence:.0%} Confidence")

    for method_name, sim_matrix in results.items():
        sim_df = pd.DataFrame(sim_matrix, index=sim_index)
        final_prices = sim_df.iloc[-1]

        var_price, es_price, var_return, es_return = calculate_risk_metrics(
            final_prices.values, last_price, var_confidence
        )

        st.markdown(f"#### {method_name}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("VaR Price", f"${var_price:.2f}", f"{var_return:.2%}")
        col2.metric("Expected Shortfall", f"${es_price:.2f}", f"{es_return:.2%}")
        col3.metric("Median Price", f"${final_prices.median():.2f}")
        col4.metric("Mean Price", f"${final_prices.mean():.2f}")

        # Risk distribution plot
        fig = go.Figure()

        returns = (final_prices - last_price) / last_price * 100

        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=100,
            name='Return distribution',
            marker_color='lightblue'
        ))

        # Add VaR and ES lines
        fig.add_vline(
            x=var_return * 100,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"VaR ({var_confidence:.0%})",
            annotation_position="top"
        )

        fig.add_vline(
            x=es_return * 100,
            line_dash="dash",
            line_color="red",
            annotation_text="Expected Shortfall",
            annotation_position="top"
        )

        fig.update_layout(
            title=f"Return Distribution after {horizon_days} Days",
            xaxis_title="Return (%)",
            yaxis_title="Frequency",
            template='plotly_white',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("📈 Distribution of simulated final prices")

    # Comparison of all methods
    fig = go.Figure()

    colors = ['blue', 'green', 'red', 'purple', 'orange']

    for idx, (method_name, sim_matrix) in enumerate(results.items()):
        sim_df = pd.DataFrame(sim_matrix, index=sim_index)
        final_prices = sim_df.iloc[-1]

        fig.add_trace(go.Histogram(
            x=final_prices,
            nbinsx=80,
            name=method_name,
            opacity=0.6,
            marker_color=colors[idx % len(colors)]
        ))

    # Add current price line
    fig.add_vline(
        x=last_price,
        line_dash="solid",
        line_color="black",
        line_width=2,
        annotation_text=f"Current: ${last_price:.2f}",
        annotation_position="top"
    )

    fig.update_layout(
        title=f"Distribution of Final Prices after {horizon_days} Days",
        xaxis_title="Price ($)",
        yaxis_title="Frequency",
        barmode='overlay',
        template='plotly_white',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Quantile table
    st.markdown("### Quantiles of Simulated Final Prices")

    quantile_data = {}
    quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

    for method_name, sim_matrix in results.items():
        sim_df = pd.DataFrame(sim_matrix, index=sim_index)
        final_prices = sim_df.iloc[-1]
        quantile_data[method_name] = [final_prices.quantile(q) for q in quantiles]

    quantile_df = pd.DataFrame(
        quantile_data,
        index=[f"{int(q*100)}%" for q in quantiles]
    )

    st.dataframe(quantile_df.style.format("${:.2f}"), use_container_width=True)

with tab4:
    st.subheader("🔄 Rolling Statistics (30 & 60 Days)")

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Rolling Drift (μ)", "Rolling Volatility (σ)"),
        vertical_spacing=0.15
    )

    # Rolling Drift
    fig.add_trace(
        go.Scatter(x=rolling_30d_mu.index, y=rolling_30d_mu * 100, name='30-Day μ', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=rolling_60d_mu.index, y=rolling_60d_mu * 100, name='60-Day μ', line=dict(color='lightblue')),
        row=1, col=1
    )
    fig.add_hline(y=mu_annual * 100, line_dash="dash", line_color="gray", row=1, col=1)

    # Rolling Volatility
    fig.add_trace(
        go.Scatter(x=rolling_30d_sigma.index, y=rolling_30d_sigma * 100, name='30-Day σ', line=dict(color='red')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=rolling_60d_sigma.index, y=rolling_60d_sigma * 100, name='60-Day σ', line=dict(color='pink')),
        row=2, col=1
    )
    fig.add_hline(y=sigma_annual * 100, line_dash="dash", line_color="gray", row=2, col=1)

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Annualized (%)", row=1, col=1)
    fig.update_yaxes(title_text="Annualized (%)", row=2, col=1)

    fig.update_layout(
        height=700,
        template='plotly_white',
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # Latest values
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("30d μ (current)", f"{rolling_30d_mu.iloc[-1]:.2%}")
    col2.metric("60d μ (current)", f"{rolling_60d_mu.iloc[-1]:.2%}")
    col3.metric("30d σ (current)", f"{rolling_30d_sigma.iloc[-1]:.2%}")
    col4.metric("60d σ (current)", f"{rolling_60d_sigma.iloc[-1]:.2%}")

with tab5:
    st.subheader("📋 Data Export")

    # Select which simulation to export
    export_method = st.selectbox("Method for Export", list(results.keys()))

    sim_matrix = results[export_method]
    sim_df = pd.DataFrame(sim_matrix, index=sim_index)

    # Generate CSV without caching to avoid timestamp serialization issues
    buffer = StringIO()
    sim_df.to_csv(buffer)
    csv = buffer.getvalue()

    st.download_button(
        label=f"📥 Download {export_method} CSV",
        data=csv,
        file_name=f"simulated_paths_{ticker}_{export_method.replace(' ', '_')}.csv",
        mime='text/csv'
    )

    st.caption(f"Shape: {sim_df.shape[0]} Time Steps × {sim_df.shape[1]} Paths")

    # Export statistics
    st.markdown("### Export Statistics")
    final_prices = sim_df.iloc[-1]

    stats_data = {
        "Metric": ["Mean", "Median", "Std. Deviation", "Min", "Max", "5% Quantile", "95% Quantile"],
        "Value": [
            f"${final_prices.mean():.2f}",
            f"${final_prices.median():.2f}",
            f"${final_prices.std():.2f}",
            f"${final_prices.min():.2f}",
            f"${final_prices.max():.2f}",
            f"${final_prices.quantile(0.05):.2f}",
            f"${final_prices.quantile(0.95):.2f}"
        ]
    }

    st.table(pd.DataFrame(stats_data))

# ----- Notes & Assumptions -----
st.markdown("---")
with st.expander("ℹ️ Methodology & Notes", expanded=False):
   st.markdown("""
   ### GARCH(1,1) + GBM Method

   **GARCH(1,1) - Generalized Autoregressive Conditional Heteroskedasticity**
   - Models time-varying volatility
   - Captures volatility clustering (volatile periods follow volatile periods)
   - Uses historical data to estimate future volatility 
   - Forecasts volatility for the entire simulation horizon

   **GBM - Geometric Brownian Motion**
   - Price paths follow a log-normal distribution
   - Uses GARCH-forecasted volatility (instead of constant σ)
   - Drift (μ) based on historical logarithmic returns

   ### P/E Ratio Adjustment

   - Uses Forward P/E or Trailing P/E as fundamental indicator
   - Low P/E → Increase drift (potentially undervalued)
   - High P/E → Decrease drift (potentially overvalued) 
   - Adjustment range: 0.5x to 1.5x

   ### Risk Metrics

   - **Value-at-Risk (VaR)**: Maximum expected loss at given confidence level
   - **Expected Shortfall (ES/CVaR)**: Average loss in worst-case scenario
   - **Rolling Statistics**: 30 and 60-day windows for μ and σ

   ### Important Notes

   - All simulations are **purely illustrative** and not investment advice
   - Past performance is not indicative of future results
   - GARCH models are based on historical patterns and cannot predict regime changes
   - Extreme events (Black Swans) may be underestimated
   - GARCH estimation may become unstable in highly volatile markets

   ### Advantages of GARCH(1,1)

   ✅ Captures volatility clustering
   ✅ More realistic volatility estimation than constant σ
   ✅ Accounts for time-varying market conditions
   ✅ Widely used in the financial industry
   """)

st.markdown("---")
st.caption("Built with Streamlit, yfinance and arch | Data: Yahoo Finance | No Investment Advice")
