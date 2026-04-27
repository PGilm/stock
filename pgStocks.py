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
MAX_AUTO_PEERS = 10
DEFAULT_STATE = {
    "tickers_input": "^dji, ^rut, ^ixic, \nvtsax, fcntx, ponax, \nORCL, MSFT,",
    "market_input": "^GSPC",
    "selected_period": "5 Years",
    "show_beta": True,
    "show_sharpe": False,
    "risk_free_rate": 0.0,
    "peer_source_input": "VTSAX",
    "manual_peer_input": "",
    "enable_auto_peers": True,
    "include_peers_in_chart": True,
    "peer_count": 5,
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


def parse_ticker_input(raw_value):
    tickers = []
    seen = set()
    for value in raw_value.split(","):
        ticker = value.strip().upper()
        if ticker and ticker not in seen:
            seen.add(ticker)
            tickers.append(ticker)
    return tickers


def dedupe_tickers(values):
    deduped = []
    seen = set()
    for value in values:
        ticker = value.strip().upper()
        if ticker and ticker not in seen:
            seen.add(ticker)
            deduped.append(ticker)
    return deduped


def _first_present(mapping, keys):
    for key in keys:
        value = mapping.get(key)
        if value not in (None, "", []):
            return value
    return None


def _normalize_provider_name(value):
    if not value:
        return None
    return " ".join(str(value).strip().upper().split())


def _coerce_assets(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_assets(value):
    if value is None or pd.isna(value):
        return "N/A"
    absolute_value = abs(float(value))
    if absolute_value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"
    if absolute_value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if absolute_value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    return f"${value:,.0f}"


def _ensure_price_frame(close_data):
    if isinstance(close_data, pd.Series):
        return close_data.to_frame()
    return close_data.copy()


def _append_ranked_peers(
    selected_peers,
    candidates,
    selected_tickers,
    used_providers,
    used_strategies,
    peer_limit,
    *,
    allow_used_provider,
    allow_source_provider,
    allow_used_strategy,
    source_provider,
):
    added_count = 0
    for candidate in candidates:
        if len(selected_peers) >= peer_limit:
            break
        if candidate["Ticker"] in selected_tickers:
            continue

        provider = candidate["Provider"]
        strategy_key = candidate["StrategyKey"]

        if not allow_used_provider and provider != "UNKNOWN" and provider in used_providers:
            continue
        if not allow_source_provider and source_provider and provider == source_provider:
            continue
        if not allow_used_strategy and strategy_key and strategy_key in used_strategies:
            continue

        selected_peers.append(candidate)
        selected_tickers.add(candidate["Ticker"])
        if provider != "UNKNOWN":
            used_providers.add(provider)
        if strategy_key:
            used_strategies.add(strategy_key)
        added_count += 1

    return added_count


def _normalize_strategy_name(value):
    if not value:
        return None
    normalized = str(value).upper()
    replacements = {
        "VANGUARD": " ",
        "FIDELITY INVESTMENTS": " ",
        "FIDELITY": " ",
        "AMERICAN FUNDS": " ",
        "CAPITAL GROUP": " ",
        "SCHWAB": " ",
        "BLACKROCK": " ",
        "ISHARES": " ",
        "STATE STREET": " ",
        "SPDR": " ",
        "INDEX TRUST": " ",
        "TRUST": " ",
        "FUND": " ",
        "FD": " ",
        "ADMIRAL": " ",
        "INVESTOR": " ",
        "INSTITUTIONAL": " ",
        "INST": " ",
        "SERVICE": " ",
        "CLASS": " ",
        "CL": " ",
    }
    for old_value, new_value in replacements.items():
        normalized = normalized.replace(old_value, new_value)
    return " ".join(normalized.split())


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_security_metadata(ticker):
    metadata = {
        "ticker": ticker,
        "name": ticker,
        "quote_type": None,
        "category": None,
        "family": None,
        "exchange": None,
        "error": None,
    }

    try:
        instrument = yf.Ticker(ticker)
        info = instrument.info or {}
    except Exception as exc:
        metadata["error"] = str(exc)
        return metadata

    metadata["name"] = _first_present(
        info, ("longName", "shortName", "displayName", "name")
    ) or ticker
    metadata["quote_type"] = (
        _first_present(info, ("quoteType", "quote_type")) or ""
    ).upper() or None
    metadata["category"] = _first_present(info, ("category", "categoryName"))
    metadata["family"] = _first_present(info, ("fundFamily", "family"))
    metadata["exchange"] = _first_present(info, ("exchange", "fullExchangeName"))

    if metadata["category"] and metadata["family"]:
        return metadata

    try:
        fund_overview = instrument.funds_data.fund_overview or {}
    except Exception:
        fund_overview = {}

    metadata["category"] = metadata["category"] or _first_present(
        fund_overview, ("categoryName", "category")
    )
    metadata["family"] = metadata["family"] or _first_present(
        fund_overview, ("family", "fundFamily")
    )
    return metadata


def _make_screen_query(query_class, category, exchange):
    clauses = [query_class("eq", ["categoryname", category])]
    if exchange:
        clauses.append(query_class("eq", ["exchange", exchange]))
    if len(clauses) == 1:
        return clauses[0]
    return query_class("and", clauses)


@st.cache_data(show_spinner=False, ttl=3600)
def discover_peer_funds(
    ticker, category, quote_type, exchange, source_family, peer_limit
):
    if not category:
        return [], "No category was found for the selected peer source fund.", None

    query_class = None
    normalized_quote_type = (quote_type or "").upper()
    if normalized_quote_type and normalized_quote_type not in {"ETF", "MUTUALFUND"}:
        return [], "Peer screening currently supports mutual funds and ETFs.", None
    if normalized_quote_type == "ETF" and hasattr(yf, "ETFQuery"):
        query_class = yf.ETFQuery
    elif hasattr(yf, "FundQuery"):
        query_class = yf.FundQuery

    if query_class is None or not hasattr(yf, "screen"):
        return [], "This yfinance version does not expose fund screening.", None

    request_size = min(max(peer_limit * 12, 25), 100)
    try:
        response = yf.screen(
            _make_screen_query(query_class, category, exchange),
            size=request_size,
            sortField="fundnetassets",
            sortAsc=False,
        )
    except Exception as exc:
        return [], str(exc), None

    quotes = response.get("quotes", []) if isinstance(response, dict) else []
    candidates = []
    for quote in quotes:
        symbol = (quote.get("symbol") or quote.get("ticker") or "").upper()
        if not symbol or symbol == ticker.upper():
            continue
        metadata = fetch_security_metadata(symbol)
        family = (
            metadata["family"]
            or quote.get("fundFamily")
            or quote.get("family")
            or ""
        )
        provider = _normalize_provider_name(family)
        display_name = (
            metadata["name"]
            or quote.get("longName")
            or quote.get("shortName")
            or symbol
        )
        assets = _coerce_assets(
            _first_present(
                quote,
                (
                    "fundNetAssets",
                    "fundnetassets",
                    "totalAssets",
                    "netAssets",
                    "aum",
                ),
            )
        )
        candidates.append(
            {
                "Ticker": symbol,
                "Name": display_name,
                "Category": metadata["category"] or quote.get("categoryName") or category,
                "Family": family,
                "Provider": provider or "UNKNOWN",
                "Assets": assets,
                "StrategyKey": _normalize_strategy_name(display_name),
                "SameFamilyAsSource": (
                    provider is not None
                    and provider == _normalize_provider_name(source_family)
                ),
            }
        )

    candidates.sort(
        key=lambda peer: (
            peer["SameFamilyAsSource"],
            -(peer["Assets"] if peer["Assets"] is not None else -1),
            peer["Ticker"],
        )
    )

    peers = []
    selection_note = None
    used_providers = set()
    used_strategies = set()
    selected_tickers = set()
    source_provider = _normalize_provider_name(source_family)

    phase_counts = []
    phase_counts.append(
        _append_ranked_peers(
            peers,
            candidates,
            selected_tickers,
            used_providers,
            used_strategies,
            peer_limit,
            allow_used_provider=False,
            allow_source_provider=False,
            allow_used_strategy=False,
            source_provider=source_provider,
        )
    )
    phase_counts.append(
        _append_ranked_peers(
            peers,
            candidates,
            selected_tickers,
            used_providers,
            used_strategies,
            peer_limit,
            allow_used_provider=True,
            allow_source_provider=False,
            allow_used_strategy=False,
            source_provider=source_provider,
        )
    )
    phase_counts.append(
        _append_ranked_peers(
            peers,
            candidates,
            selected_tickers,
            used_providers,
            used_strategies,
            peer_limit,
            allow_used_provider=True,
            allow_source_provider=True,
            allow_used_strategy=False,
            source_provider=source_provider,
        )
    )
    phase_counts.append(
        _append_ranked_peers(
            peers,
            candidates,
            selected_tickers,
            used_providers,
            used_strategies,
            peer_limit,
            allow_used_provider=True,
            allow_source_provider=True,
            allow_used_strategy=True,
            source_provider=source_provider,
        )
    )

    if len(peers) < peer_limit:
        selection_note = (
            f"Only {len(peers)} eligible peers were available after relaxing provider and strategy matching."
        )
    elif phase_counts[1] > 0 or phase_counts[2] > 0 or phase_counts[3] > 0:
        selection_note = (
            "Provider diversity was relaxed to fill the requested peer count."
        )

    return peers, None, selection_note


def build_peer_table(manual_peers, auto_peers, peer_source):
    rows = []
    for ticker in manual_peers:
        metadata = fetch_security_metadata(ticker)
        rows.append(
            {
                "Ticker": ticker,
                "Name": metadata["name"],
                "Category": metadata["category"] or "N/A",
                "Provider": metadata["family"] or "N/A",
                "AUM": "N/A",
            }
        )

    seen = {peer_source, *manual_peers}
    for peer in auto_peers:
        ticker = peer["Ticker"]
        if ticker in seen:
            continue
        seen.add(ticker)
        rows.append(
            {
                "Ticker": ticker,
                "Name": peer["Name"],
                "Category": peer["Category"] or "N/A",
                "Provider": peer["Family"] or "N/A",
                "AUM": _format_assets(peer["Assets"]),
            }
        )

    return pd.DataFrame(rows)


st.title("Stock Performance Tracker")

saved_state = load_saved_state()

for state_key, default_value in saved_state.items():
    if state_key not in st.session_state:
        st.session_state[state_key] = default_value

st.sidebar.header("Configuration")
tickers_input = st.sidebar.text_area(
    "Edit Tickers (comma-separated)",
    key="tickers_input",
    height=100,
    help="""Enter stock tickers separated by commas. Some common indexes
            include the DJIA (^DJI), S&P 500 (^GSPC), Russell 2000 (^RUT),
            and NASDAQ (^IXIC). E.g. ^dji, ^rut, ^ixic, vtsax, fcntx,
            ponax, ORCL, MSFT,""",
)
tickers = parse_ticker_input(tickers_input)

if not tickers:
    st.error("Please enter at least one ticker symbol.")
    st.stop()

market_input = st.sidebar.text_input(
    "Baseline Index",
    key="market_input",
    help="""Enter the ticker for the baseline index.
            Common ones: ^GSPC (S&P 500), ^DJI (Dow Jones),
            ^IXIC (NASDAQ), ^RUT (Russell 2000)""",
)
market = market_input.upper()

st.sidebar.header("Peer Fund Lookup")
default_peer_source = saved_state["peer_source_input"] or next(
    (ticker for ticker in tickers if not ticker.startswith("^")),
    tickers[0],
)
peer_source_input = st.sidebar.text_input(
    "Peer Source Fund",
    key="peer_source_input",
    value=default_peer_source,
    help="Fund or ETF ticker used for category-based peer discovery.",
)
peer_source = peer_source_input.strip().upper()
manual_peer_input = st.sidebar.text_area(
    "Manual Peer Overrides",
    key="manual_peer_input",
    height=80,
    help="Optional peer funds to add manually. These are appended to the chart if enabled below.",
)
manual_peers = parse_ticker_input(manual_peer_input)
enable_auto_peers = st.sidebar.checkbox(
    "Find Category-Based Peers",
    key="enable_auto_peers",
    help="Uses Yahoo Finance fund categories through yfinance to suggest same-category peer funds.",
)
include_peers_in_chart = st.sidebar.checkbox(
    "Add Peers to Comparison Chart",
    key="include_peers_in_chart",
    help="Appends manual peers and discovered peer funds to the downloaded ticker set.",
)
peer_count = int(
    st.sidebar.slider(
        "Target Auto Peer Count",
        min_value=1,
        max_value=MAX_AUTO_PEERS,
        key="peer_count",
    )
)

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
    key="selected_period",
)

st.sidebar.header("Options")
show_beta = st.sidebar.checkbox(
    "Show Beta",
    key="show_beta",
    help="Plots the 30-day rolling beta relative to the selected baseline index.",
)
show_sharpe = st.sidebar.checkbox(
    "Show Sharpe Ratio",
    key="show_sharpe",
    help="Plots the 30-day rolling Sharpe ratio on the secondary axis.",
)
risk_free_rate_pct = st.sidebar.number_input(
    "Annual Risk-Free Rate (%)",
    min_value=-10.0,
    max_value=20.0,
    key="risk_free_rate_pct",
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
        "peer_source_input": peer_source,
        "manual_peer_input": manual_peer_input,
        "enable_auto_peers": enable_auto_peers,
        "include_peers_in_chart": include_peers_in_chart,
        "peer_count": peer_count,
    }
)

peer_source_metadata = None
auto_peers = []
peer_lookup_error = None
peer_lookup_note = None
requested_auto_peer_count = peer_count

if peer_source:
    peer_source_metadata = fetch_security_metadata(peer_source)
    if enable_auto_peers:
        auto_peers, peer_lookup_error, peer_lookup_note = discover_peer_funds(
            peer_source,
            peer_source_metadata["category"],
            peer_source_metadata["quote_type"],
            peer_source_metadata["exchange"],
            peer_source_metadata["family"],
            peer_count,
        )
        excluded = {peer_source, *manual_peers}
        auto_peers = [
            peer for peer in auto_peers if peer["Ticker"] not in excluded
        ][:peer_count]

peer_panel = st.container()
with peer_panel:
    st.subheader("Peer Funds")
    if not peer_source:
        st.info("Enter a fund ticker in Peer Source Fund to discover same-category peers.")
    elif peer_source_metadata and peer_source_metadata["error"]:
        st.warning(
            f"Could not load fund metadata for {peer_source}: {peer_source_metadata['error']}"
        )
    else:
        category_label = (
            peer_source_metadata["category"] if peer_source_metadata else None
        ) or "N/A"
        fund_name = (peer_source_metadata["name"] if peer_source_metadata else None) or peer_source
        family_label = (
            peer_source_metadata["family"] if peer_source_metadata else None
        ) or "N/A"
        st.caption(
            f"Peer source: {peer_source} ({fund_name}) | Category: {category_label} | Family: {family_label}"
        )
        if auto_peers:
            st.caption(
                f"Showing {len(auto_peers)} automatic peers for a target of {requested_auto_peer_count}. "
                "The list keeps providers distinct where possible and leans toward larger AUM."
            )
            if peer_lookup_note:
                st.caption(peer_lookup_note)
        elif enable_auto_peers and not peer_lookup_error:
            st.caption(
                f"No automatic peers were available for a target of {requested_auto_peer_count}."
            )
        if peer_lookup_error and enable_auto_peers:
            st.info(f"Automatic peer lookup is unavailable right now: {peer_lookup_error}")

        peer_table = build_peer_table(manual_peers, auto_peers, peer_source)
        if peer_table.empty:
            st.info("No peers to show yet. Add manual peers or enable category-based discovery.")
        else:
            st.dataframe(peer_table, width="stretch", hide_index=True)

comparison_tickers = list(tickers)
if include_peers_in_chart:
    comparison_tickers = dedupe_tickers(
        comparison_tickers
        + ([peer_source] if peer_source else [])
        + manual_peers
        + [peer["Ticker"] for peer in auto_peers]
    )
else:
    comparison_tickers = dedupe_tickers(comparison_tickers)

end = pd.Timestamp.today()
start = end - period_options[selected_period]
download_symbols = dedupe_tickers(comparison_tickers + [market])

st.sidebar.info(f'Downloading data for: {", ".join(download_symbols)}')
try:
    data = yf.download(download_symbols, start=start, end=end, progress=False)
    prices = _ensure_price_frame(data["Close"])

    required_symbols = dedupe_tickers(comparison_tickers + [market])
    missing_symbols = [symbol for symbol in required_symbols if symbol not in prices.columns]
    if missing_symbols:
        st.error(f"Missing price data for: {', '.join(missing_symbols)}")
        st.stop()

    prices = prices[required_symbols].dropna(how="all")
    if prices.empty:
        st.error("No price data was returned for the selected symbols and period.")
        st.stop()

    prices_norm = prices.div(prices.ffill().bfill().iloc[0]).mul(100)

    fig, ax = plt.subplots(figsize=(12, 6))
    ticker_colors = {}
    for ticker in comparison_tickers:
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

    if show_beta or show_sharpe:
        ax2 = ax.twinx()

        for ticker in comparison_tickers:
            stock_returns = prices[ticker].pct_change()

            if show_beta:
                beta_series = (
                    stock_returns.rolling(window=ROLLING_WINDOW).cov(market_returns)
                    / market_returns.rolling(window=ROLLING_WINDOW).var()
                )
                ax2.plot(
                    beta_series.index,
                    beta_series,
                    label="_nolegend_",
                    linestyle="dotted",
                    alpha=0.7,
                    color=ticker_colors.get(ticker),
                )

            if show_sharpe:
                rolling_sharpe = stock_returns.rolling(window=ROLLING_WINDOW).apply(
                    lambda values: annualized_sharpe_ratio(
                        pd.Series(values), risk_free_rate
                    )
                    if len(values) == ROLLING_WINDOW
                    else None,
                    raw=False,
                )
                ax2.plot(
                    rolling_sharpe.index,
                    rolling_sharpe,
                    label="_nolegend_",
                    linestyle="dashdot",
                    alpha=0.8,
                    color=ticker_colors.get(ticker),
                )

        ax2.set_ylabel("Rolling Risk Metrics (30-day)", fontsize=12)

    primary_lines, primary_labels = ax.get_legend_handles_labels()
    ax.legend(primary_lines, primary_labels, loc="upper left")

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader(f"Risk Metrics for {selected_period} (relative to Baseline Index)")
    market_returns = market_prices.pct_change().dropna()
    metrics_data = []
    for ticker in comparison_tickers:
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

    st.subheader("Normalized Price Data")
    st.dataframe(prices_norm[comparison_tickers], width="stretch")

except Exception as exc:
    st.error(f"Error downloading data: {exc}")
