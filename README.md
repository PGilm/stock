# Stock Performance Tracker

A Streamlit app for tracking stock, ETF, and index performance with per-user saved configuration storage.

This app lets you:
- compare normalized performance for multiple tickers over selected time periods
- measure rolling beta and Sharpe ratio relative to a baseline index
- discover peer funds or stocks automatically from a peer source ticker
- save named dashboard configurations per user
- restore the last-used user configuration automatically on login

## Features

- **User authentication**: sign in or register from the sidebar.
- **Saved configurations**: save and load named dashboard setups for each user.
- **Database storage**: user accounts and saved configurations are stored in a local SQLite database.
- **Auto peers**: discover related securities for ETFs, mutual funds, and stocks.
- **Snapshot caching**: data snapshots are saved with each configuration, reducing repeated downloads.

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
   - the analysis period, beta/sharpe options, and risk-free rate
2. Save your current configuration using a memorable name.
3. Sign in or register in the sidebar to keep your personal configurations.
4. When signed in, the app restores your last used saved configuration automatically.

## User data location

The app stores user data in your platform-specific config directory under:

- `~/.config/pgStocks/pgStocks_users.db` on Linux

Saved configurations and user accounts are persisted there.

## Notes

- If you are not signed in, the app still works as a guest using the local JSON state file.
- The app uses `yfinance` for market and peer metadata.
- Existing saved configurations are kept per user and are not shared across accounts.
