import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Stock Performance Tracker")

STATE_FILE = Path(__file__).with_name("pgStocks_state.json")
TRADING_DAYS_PER_YEAR = 252
ROLLING_WINDOW = 30
DEFAULT_STATE = {
    "tickers_input": "^dji, ^rut, ^ixic, \nvtsax, fcntx, ponax, \nORCL, MSFT,",
    "market_input": "^GSPC",
    "selected_period": "5 Years",
    "show_beta": True,
    "show_sharpe": False,
    "risk_free_rate": 0.0,
}


def load_saved_state():
    if not STATE_FILE.exists():
        return DEFAULT_STATE.copy()

    try:
        saved_state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return DEFAULT_STATE.copy()

    state = DEFAULT_STATE.copy()
    state.update({key: saved_state[key] for key in DEFAULT_STATE if key in saved_state})
    return state


def save_state(state):
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError as exc:
        st.sidebar.warning(f"Could not save local settings: {exc}")


def annualized_sharpe_ratio(returns, annual_risk_free_rate):
    std_dev = returns.std()
    if pd.isna(std_dev) or std_dev == 0:
        return None

    daily_risk_free_rate = annual_risk_free_rate / TRADING_DAYS_PER_YEAR
    excess_returns = returns - daily_risk_free_rate
    sharpe_ratio = excess_returns.mean() / std_dev * (TRADING_DAYS_PER_YEAR ** 0.5)
    return sharpe_ratio if pd.notna(sharpe_ratio) else None


st.title("Stock Performance Tracker")

saved_state = load_saved_state()

# Sidebar for ticker input
st.sidebar.header("Configuration")
tickers_input = st.sidebar.text_area(
    "Edit Tickers (comma-separated)",
    value=saved_state["tickers_input"],
    height=100,
    help="""Enter stock tickers separated by commas. Some common indexes
            include the DJIA (^DJI), S&P 500 (^GSPC), Russell 2000 (^RUT),
            and NASDAQ (^IXIC). E.g. ^dji, ^rut, ^ixic, vtsax, fcntx,
            ponax, ORCL, MSFT,""",
)

# Parse tickers from input
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if not tickers:
    st.error("Please enter at least one ticker symbol.")
    st.stop()

# Define market index for beta calculation
market_input = st.sidebar.text_input(
    "Baseline Index",
    value=saved_state["market_input"],
    help="""Enter the ticker for the baseline index.
            Common ones: ^GSPC (S&P 500), ^DJI (Dow Jones),
            ^IXIC (NASDAQ), ^RUT (Russell 2000)""",
)
market = market_input.upper()

# Time period selection
st.sidebar.header("Time Period")
period_options = {
    "5 Years": pd.DateOffset(years=5),
    "3 Years": pd.DateOffset(years=3),
    "1 Year": pd.DateOffset(years=1),
    "6 Months": pd.DateOffset(months=6),
    "3 Months": pd.DateOffset(months=3),
    "1 Month": pd.DateOffset(months=1),
}
period_labels = list(period_options.keys())
selected_period = st.sidebar.selectbox(
    "Select period:",
    period_labels,
    index=period_labels.index(saved_state["selected_period"])
    if saved_state["selected_period"] in period_labels
    else 0,
)

# Options
st.sidebar.header("Options")
show_beta = st.sidebar.checkbox(
    "Show Beta",
    value=saved_state["show_beta"],
    help="Plots the 30-day rolling beta relative to the selected baseline index.",
)
show_sharpe = st.sidebar.checkbox(
    "Show Sharpe Ratio",
    value=saved_state["show_sharpe"],
    help="Plots the 30-day rolling Sharpe ratio on the secondary axis.",
)
risk_free_rate_pct = st.sidebar.number_input(
    "Annual Risk-Free Rate (%)",
    min_value=-10.0,
    max_value=20.0,
    value=float(saved_state["risk_free_rate"]) * 100,
    step=0.25,
    help="Used for Sharpe ratio calculations in the table and optional plot.",
)
risk_free_rate = risk_free_rate_pct / 100

save_state(
    {
        "tickers_input": tickers_input,
        "market_input": market,
        "selected_period": selected_period,
        "show_beta": show_beta,
        "show_sharpe": show_sharpe,
        "risk_free_rate": risk_free_rate,
    }
)

end = pd.Timestamp.today()
start = end - period_options[selected_period]

# Download data
st.sidebar.info(f'Downloading data for: {", ".join(tickers)}')
try:
    data = yf.download(tickers + [market], start=start, end=end, progress=False)
    prices = data["Close"].copy()

    required_symbols = tickers + [market]
    missing_symbols = [symbol for symbol in required_symbols if symbol not in prices.columns]
    if missing_symbols:
        st.error(f"Missing price data for: {', '.join(missing_symbols)}")
        st.stop()

    prices = prices[required_symbols].dropna(how="all")
    if prices.empty:
        st.error("No price data was returned for the selected symbols and period.")
        st.stop()

    prices_norm = prices.div(prices.ffill().bfill().iloc[0]).mul(100)

    # Display chart
    fig, ax = plt.subplots(figsize=(12, 6))
    ticker_colors = {}
    for ticker in tickers:
        series = prices_norm[ticker].dropna()
        if not series.empty:
            price_line = ax.plot(series.index, series, label=ticker, linewidth=2)[0]
            ticker_colors[ticker] = price_line.get_color()

    ax.set_title(f"Stock Performance - {selected_period}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Index (Start = 100)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.grid(True, alpha=0.2)

    market_prices = prices[market]
    market_returns = market_prices.pct_change()
    plot_metric_lines = []

    # Add secondary axis for rolling beta and/or Sharpe ratio if enabled
    if show_beta or show_sharpe:
        ax2 = ax.twinx()

        for ticker in tickers:
            stock_returns = prices[ticker].pct_change()

            if show_beta:
                beta_series = (
                    stock_returns.rolling(window=ROLLING_WINDOW).cov(market_returns)
                    / market_returns.rolling(window=ROLLING_WINDOW).var()
                )
                beta_line = ax2.plot(
                    beta_series.index,
                    beta_series,
                    label="_nolegend_",
                    linestyle="dotted",
                    alpha=0.7,
                    color=ticker_colors.get(ticker),
                )[0]
                plot_metric_lines.append(beta_line)

            if show_sharpe:
                rolling_sharpe = stock_returns.rolling(window=ROLLING_WINDOW).apply(
                    lambda values: annualized_sharpe_ratio(
                        pd.Series(values), risk_free_rate
                    )
                    if len(values) == ROLLING_WINDOW
                    else None,
                    raw=False,
                )
                sharpe_line = ax2.plot(
                    rolling_sharpe.index,
                    rolling_sharpe,
                    label="_nolegend_",
                    linestyle="dashdot",
                    alpha=0.8,
                    color=ticker_colors.get(ticker),
                )[0]
                plot_metric_lines.append(sharpe_line)

        ax2.set_ylabel("Rolling Risk Metrics (30-day)", fontsize=12)

    primary_lines, primary_labels = ax.get_legend_handles_labels()
    ax.legend(primary_lines, primary_labels, loc="upper left")

    plt.tight_layout()
    st.pyplot(fig)

    # Calculate and display beta, volatility, and Sharpe metrics
    st.subheader(f"Risk Metrics for {selected_period} (relative to Baseline Index)")
    market_returns = market_prices.pct_change().dropna()
    metrics_data = []
    for ticker in tickers:
        stock_returns = prices[ticker].pct_change().dropna()
        common_index = stock_returns.index.intersection(market_returns.index)
        if len(common_index) > 1:
            aligned_returns = stock_returns.loc[common_index]
            aligned_market_returns = market_returns.loc[common_index]
            cov = aligned_returns.cov(aligned_market_returns)
            var = aligned_market_returns.var()
            beta = cov / var if var != 0 else 0
            std_dev = aligned_returns.std()
            sharpe_ratio = annualized_sharpe_ratio(aligned_returns, risk_free_rate)
            metrics_data.append(
                {
                    "Ticker": ticker,
                    "Beta": round(beta, 2),
                    "Std Dev of Returns": round(std_dev, 4),
                    "Sharpe Ratio": round(sharpe_ratio, 2) if sharpe_ratio is not None else "N/A",
                }
            )
        else:
            metrics_data.append(
                {
                    "Ticker": ticker,
                    "Beta": "N/A",
                    "Std Dev of Returns": "N/A",
                    "Sharpe Ratio": "N/A",
                }
            )
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, width="stretch")

    # Display data table
    st.subheader("Normalized Price Data")
    st.dataframe(prices_norm[tickers], width="stretch")

except Exception as exc:
    st.error(f"Error downloading data: {exc}")
