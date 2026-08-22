# Stock Performance Tracker

A Streamlit app for tracking stock, ETF, and index performance with per-user saved configuration storage.

This app lets you:
- compare normalized performance for multiple tickers over selected time periods, including the maximum available Yahoo Finance history
- measure rolling beta and Sharpe ratio relative to a baseline index
- discover peer funds or stocks automatically from a peer source ticker
- save named dashboard configurations per user
- restore the last-used user configuration automatically on login

## Features

- **User authentication**: sign in or register from the sidebar.
- **Saved configurations**: save and load named dashboard setups for each user.
- **Database storage**: user accounts and saved configurations are stored in a local SQLite database.
- **Auto peers**: discover related securities for ETFs, mutual funds, and stocks.
- **Shared ticker caching**: downloaded ticker metadata and adjusted-close price history are stored in a centralized SQLite cache and reused across users.
- **Snapshot caching**: data snapshots are also saved with each configuration for fast dashboard restores.

## Quick start

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Create your local environment file from the example:

   ```bash
   cp .env.example .env
   ```

3. Run the application:

   ```bash
   streamlit run pgStocks.py
   ```

4. Open the URL shown by Streamlit in your browser.

## How to use it

1. Use the sidebar to enter:
   - a list of tickers to compare
   - a baseline index for risk metrics
   - a peer source ticker for related-security discovery
   - manual peer overrides and peer count
   - the analysis period, including `Max` for all available Yahoo Finance history, beta/sharpe options, and risk-free rate
2. Save your current configuration using a memorable name.
3. Sign in or register in the sidebar to keep your personal configurations.
4. When signed in, the app restores your last used saved configuration automatically.

## Data Storage

The app stores user data in your platform-specific config directory. On Linux, the default files are:

- `~/.config/pgStocks/pgStocks_users.db` for users and saved configurations
- `~/.config/pgStocks/pgStocks_ticker_cache.db` for shared ticker metadata and price history

The ticker cache is shared by all users of the local app. It keeps the longest history it has seen for each ticker, backfills older missing ranges when a user asks for a longer period, and appends newer dates as needed so Yahoo Finance is only called for missing or refreshed market data.

## Notes

- If you are not signed in, the app still works as a guest using the local JSON state file.
- The app uses `yfinance` for market prices, security metadata, and peer screening.
- Existing saved configurations are kept per user and are not shared across accounts.
