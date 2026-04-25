import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.title('Stock Performance Tracker')
st.set_page_config(page_title="Stock Performance Tracker")

# Sidebar for ticker input
st.sidebar.header('Configuration')
tickers_input = st.sidebar.text_area(
    'Edit Tickers (comma-separated)',
    value='^dji, ^rut, ^ixic, vtsax, fcntx, ponax, ORCL, MSFT,',
    height=100,
    help='Enter stock tickers separated by commas. Some common indexes include the DJIA (^DJI), S&P 500 (^GSPC), Russell 2000 (^RUT), and NASDAQ (^IXIC). E.g. ^dji, ^rut, ^ixic, vtsax, fcntx, ponax, ORCL, MSFT,'
)

# Parse tickers from input
tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

if not tickers:
    st.error('Please enter at least one ticker symbol.')
    st.stop()

# Define market index for beta calculation
market_input = st.sidebar.text_input('Baseline Index', value='^GSPC', help='Enter the ticker for the baseline index. Common ones: ^GSPC (S&P 500), ^DJI (Dow Jones), ^IXIC (NASDAQ), ^RUT (Russell 2000)')
market = market_input.upper()

# Time period selection
st.sidebar.header('Time Period')
period_options = {'5 Years': pd.DateOffset(years=5), '3 Years': pd.DateOffset(years=3),
                  '1 Year': pd.DateOffset(years=1), '6 Months': pd.DateOffset(months=6), 
                  '3 Months': pd.DateOffset(months=3), '1 Month': pd.DateOffset(months=1)}
selected_period = st.sidebar.selectbox('Select period:', list(period_options.keys()), index=0)

# Options
st.sidebar.header('Options')
show_beta = st.sidebar.checkbox('Show Beta', value=True)

end = pd.Timestamp.today()
start = end - period_options[selected_period]

# Download data
st.sidebar.info(f'Downloading data for: {", ".join(tickers)}')
try:
    data = yf.download(tickers + [market], start=start, end=end, progress=False)
    # Extract adjusted close prices (yfinance now returns 'Close')
    prices = data['Close']
    prices_norm = prices / prices.iloc[0] * 100

    # Display chart
    fig, ax = plt.subplots(figsize=(12, 6))
    for t in tickers:
        ax.plot(prices_norm.index, prices_norm[t], label=t, linewidth=2)

    ax.set_title(f'Stock Performance – {selected_period}', fontsize=14, fontweight='bold')
    ax.set_ylabel('Index (Start = 100)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.2)

    # Add secondary axis for beta if enabled
    if show_beta:
        ax2 = ax.twinx()
        market_prices = prices[market]
        market_returns = market_prices.pct_change()
        for t in tickers:
            stock_returns = prices[t].pct_change()
            beta_series = stock_returns.rolling(window=30).cov(market_returns) / market_returns.rolling(window=30).var()
            ax2.plot(prices_norm.index, beta_series, label=f'{t} Beta', linestyle='dotted', alpha=0.7)
        ax2.set_ylabel('Beta (30-day rolling)', fontsize=12)
        ax2.legend(loc='upper right')

    plt.tight_layout()
    st.pyplot(fig)

    # Calculate and display beta and volatility metrics
    st.subheader(f'Risk Metrics for {selected_period} (relative to Baseline Index)')
    market_prices = prices[market]
    market_returns = market_prices.pct_change().dropna()
    metrics_data = []
    for t in tickers:
        stock_returns = prices[t].pct_change().dropna()
        common_index = stock_returns.index.intersection(market_returns.index)
        if len(common_index) > 1:
            cov = stock_returns.loc[common_index].cov(market_returns.loc[common_index])
            var = market_returns.loc[common_index].var()
            beta = cov / var if var != 0 else 0
            std_dev = stock_returns.loc[common_index].std()
            metrics_data.append({'Ticker': t, 'Beta': round(beta, 2), 'Std Dev of Returns': round(std_dev, 4)})
        else:
            metrics_data.append({'Ticker': t, 'Beta': 'N/A', 'Std Dev of Returns': 'N/A'})
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, width='stretch')

    # Display data table
    st.subheader('Normalized Price Data')
    st.dataframe(prices_norm[tickers], width='stretch')

except Exception as e:
    st.error(f'Error downloading data: {str(e)}')
