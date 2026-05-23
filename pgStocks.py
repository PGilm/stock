import datetime
import hashlib
import json
import math
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
from peewee import CharField, DateTimeField, ForeignKeyField, Model, SqliteDatabase, TextField
from platformdirs import user_config_dir
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Stock Performance Tracker")

APP_NAME = "pgStocks"
LEGACY_STATE_FILE = Path(__file__).with_name("pgStocks_state.json")
STATE_FILE = Path(user_config_dir(APP_NAME, APP_NAME)) / "state.json"
TRADING_DAYS_PER_YEAR = 252
ROLLING_WINDOW = 30
MAX_AUTO_PEERS = 10
MAJOR_US_EQUITY_EXCHANGES = {"NCM", "NGM", "NMS", "NYQ", "ASE", "BTS"}
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

AUTH_SALT = b"pgStocks_auth_salt"
DB_FILE = Path(user_config_dir(APP_NAME, APP_NAME)) / "pgStocks_users.db"
DATABASE = SqliteDatabase(DB_FILE, pragmas={"foreign_keys": 1})


class BaseModel(Model):
    class Meta:
        database = DATABASE


class User(BaseModel):
    username = CharField(unique=True)
    password_hash = CharField()
    created_at = DateTimeField(default=datetime.datetime.utcnow)
    last_login = DateTimeField(null=True)
    last_used_config = CharField(null=True)
    current_state = TextField(null=True)


class UserConfiguration(BaseModel):
    user = ForeignKeyField(User, backref="configurations", on_delete="CASCADE")
    name = CharField()
    state = TextField()
    data_cache = TextField(null=True)
    created_at = DateTimeField(default=datetime.datetime.utcnow)
    updated_at = DateTimeField(default=datetime.datetime.utcnow)

    class Meta:
        indexes = ((("user", "name"), True),)


def _hash_password(password):
    if password is None:
        return None
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        AUTH_SALT,
        100_000,
    ).hex()


def _ensure_user_database():
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATABASE.connect(reuse_if_open=True)
    DATABASE.create_tables([User, UserConfiguration])


_ensure_user_database()


def _json_dumps(value):
    return json.dumps(value, indent=2, default=str)


def _json_loads(value):
    if not value:
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


def _authenticate_user(username, password):
    if not username or not password:
        return None
    user = User.get_or_none(User.username == str(username).strip())
    if user and user.password_hash == _hash_password(password):
        user.last_login = datetime.datetime.utcnow()
        user.save()
        return user
    return None


def _create_user(username, password):
    name = str(username or "").strip()
    if not name or not password:
        return None
    if User.get_or_none(User.username == name):
        return None
    return User.create(
        username=name,
        password_hash=_hash_password(password),
        created_at=datetime.datetime.utcnow(),
    )


def _load_user_store(username):
    user = User.get_or_none(User.username == str(username).strip())
    if not user:
        return _default_store()

    saved_configurations = {}
    for config in UserConfiguration.select().where(UserConfiguration.user == user):
        configuration_state = _normalize_state(_json_loads(config.state) or {})
        configuration_data_cache = _json_loads(config.data_cache)
        saved_configurations[config.name] = {
            "name": config.name,
            "state": configuration_state,
            "data_cache": configuration_data_cache,
            "created_at": config.created_at.isoformat(),
            "updated_at": config.updated_at.isoformat(),
        }

    selected_configuration = (
        user.last_used_config
        if user.last_used_config in saved_configurations
        else None
    )

    if selected_configuration:
        current_state = _normalize_state(
            saved_configurations[selected_configuration]["state"]
        )
    else:
        current_state = _normalize_state(_json_loads(user.current_state) or DEFAULT_STATE.copy())

    return {
        "current_state": current_state,
        "saved_configurations": saved_configurations,
        "selected_configuration": selected_configuration,
    }


def _save_user_store(username, store):
    user = User.get_or_none(User.username == str(username).strip())
    if not user:
        return

    user.current_state = _json_dumps(store.get("current_state", DEFAULT_STATE.copy()))
    user.last_used_config = store.get("selected_configuration")
    user.save()

    existing_configs = {
        config.name: config
        for config in UserConfiguration.select().where(UserConfiguration.user == user)
    }

    for name, payload in (store.get("saved_configurations") or {}).items():
        data_cache = payload.get("data_cache")
        if name in existing_configs:
            existing_configs[name].state = _json_dumps(payload.get("state", {}))
            existing_configs[name].data_cache = _json_dumps(data_cache) if data_cache else None
            existing_configs[name].updated_at = datetime.datetime.utcnow()
            existing_configs[name].save()
        else:
            UserConfiguration.create(
                user=user,
                name=name,
                state=_json_dumps(payload.get("state", {})),
                data_cache=_json_dumps(data_cache) if data_cache else None,
                created_at=datetime.datetime.utcnow(),
                updated_at=datetime.datetime.utcnow(),
            )


def load_application_store():
    if st.session_state.get("logged_in_user"):
        return _load_user_store(st.session_state["logged_in_user"])
    return load_state_store()


def save_application_store(store):
    if st.session_state.get("logged_in_user"):
        _save_user_store(st.session_state["logged_in_user"], store)
    else:
        save_state_store(store)


class TransientLookupError(RuntimeError):
    """Represents a temporary upstream lookup failure that should not be cached."""


def _default_store():
    return {
        "current_state": DEFAULT_STATE.copy(),
        "saved_configurations": {},
        "selected_configuration": None,
    }


def _normalize_state(state):
    normalized = DEFAULT_STATE.copy()
    if isinstance(state, dict):
        normalized.update({key: state[key] for key in DEFAULT_STATE if key in state})
    return normalized


def _normalize_saved_configurations(saved_configurations):
    normalized = {}
    if not isinstance(saved_configurations, dict):
        return normalized

    for raw_name, payload in saved_configurations.items():
        if isinstance(payload, dict):
            name = str(payload.get("name") or raw_name).strip()
            state = _normalize_state(payload.get("state", payload))
            data_cache = payload.get("data_cache") if isinstance(payload.get("data_cache"), dict) else None
            created_at = payload.get("created_at")
            updated_at = payload.get("updated_at")
        else:
            name = str(raw_name).strip()
            state = _normalize_state({})
            data_cache = None
            created_at = None
            updated_at = None

        if not name:
            continue

        normalized[name] = {
            "name": name,
            "state": state,
            "data_cache": data_cache,
            "created_at": created_at,
            "updated_at": updated_at,
        }

    return normalized


def load_state_store():
    state_file = STATE_FILE if STATE_FILE.exists() else LEGACY_STATE_FILE
    if not state_file.exists():
        return _default_store()

    try:
        saved_payload = json.loads(state_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _default_store()

    if not isinstance(saved_payload, dict):
        return _default_store()

    if "current_state" in saved_payload or "saved_configurations" in saved_payload:
        current_state = _normalize_state(saved_payload.get("current_state"))
        saved_configurations = _normalize_saved_configurations(
            saved_payload.get("saved_configurations")
        )
        selected_configuration = saved_payload.get("selected_configuration")
    else:
        current_state = _normalize_state(saved_payload)
        saved_configurations = {}
        selected_configuration = None

    if selected_configuration not in saved_configurations:
        selected_configuration = None

    return {
        "current_state": current_state,
        "saved_configurations": saved_configurations,
        "selected_configuration": selected_configuration,
    }


def save_state_store(store):
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(store, indent=2), encoding="utf-8")
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


def _empty_security_metadata(ticker):
    return {
        "ticker": ticker,
        "name": ticker,
        "quote_type": None,
        "category": None,
        "family": None,
        "sector": None,
        "industry": None,
        "exchange": None,
        "market_cap": None,
        "price": None,
        "change_percent": None,
        "currency": None,
        "error": None,
    }


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


def _infer_provider_from_fund_name(name):
    normalized_name = _normalize_provider_name(name)
    if not normalized_name:
        return None

    known_provider_markers = (
        ("STRATEGIC ADVISERS FIDELITY", "FIDELITY"),
        ("AMERICAN FUNDS", "AMERICAN FUNDS"),
        ("T ROWE PRICE", "T ROWE PRICE"),
        ("T. ROWE PRICE", "T ROWE PRICE"),
        ("DODGE & COX", "DODGE & COX"),
        ("DIMENSIONAL", "DIMENSIONAL"),
        ("FIRST EAGLE", "FIRST EAGLE"),
        ("JPMORGAN", "JPMORGAN"),
        ("MORGAN STANLEY", "MORGAN STANLEY"),
        ("FRANKLIN", "FRANKLIN"),
        ("INVESCO", "INVESCO"),
        ("FIDELITY", "FIDELITY"),
        ("VANGUARD", "VANGUARD"),
        ("SCHWAB", "SCHWAB"),
        ("BLACKROCK", "BLACKROCK"),
        ("ISHARES", "ISHARES"),
        ("SPDR", "SPDR"),
        ("PIMCO", "PIMCO"),
        ("MFS", "MFS"),
    )
    for marker, provider in known_provider_markers:
        if marker in normalized_name:
            return provider

    return None


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


def _format_price(value, currency="USD"):
    if value is None or pd.isna(value):
        return "N/A"
    currency_code = (currency or "USD").upper()
    return f"{float(value):,.2f} {currency_code}"


def _format_percent(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.2f}%"


def _is_fund_quote_type(quote_type):
    return (quote_type or "").upper() in {"ETF", "MUTUALFUND"}


def _is_equity_quote_type(quote_type):
    return (quote_type or "").upper() == "EQUITY"


def _is_transient_lookup_error(error_message):
    if not error_message:
        return False

    normalized = str(error_message).upper()
    transient_markers = (
        "429",
        "CONNECTION",
        "RATE LIMIT",
        "TEMPORAR",
        "TIMEOUT",
        "TIMED OUT",
        "TOO MANY REQUESTS",
        "TRY AFTER A WHILE",
    )
    return any(marker in normalized for marker in transient_markers)


def _ensure_price_frame(close_data):
    if isinstance(close_data, pd.Series):
        return close_data.to_frame()
    return close_data.copy()


def _serialize_price_frame(frame):
    if frame is None or frame.empty:
        return None

    serializable = frame.sort_index().copy()
    serializable.index = pd.to_datetime(serializable.index).strftime("%Y-%m-%d")
    rows = []
    for row in serializable.itertuples(index=False, name=None):
        rows.append([None if pd.isna(value) else float(value) for value in row])

    return {
        "index": serializable.index.tolist(),
        "columns": [str(column) for column in serializable.columns],
        "data": rows,
    }


def _deserialize_price_frame(payload):
    if not isinstance(payload, dict):
        return None

    try:
        frame = pd.DataFrame(
            payload.get("data", []),
            index=pd.to_datetime(payload.get("index", [])),
            columns=payload.get("columns", []),
        )
    except (TypeError, ValueError):
        return None

    if frame.empty:
        return None

    return frame.apply(pd.to_numeric, errors="coerce").sort_index()


def _format_timestamp(value, *, include_time=False):
    if not value:
        return "N/A"

    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return str(value)

    timestamp = timestamp.tz_localize(None) if timestamp.tzinfo else timestamp
    return timestamp.strftime("%Y-%m-%d %H:%M" if include_time else "%Y-%m-%d")


def _build_current_state():
    return {
        "tickers_input": st.session_state.get("tickers_input", DEFAULT_STATE["tickers_input"]),
        "market_input": str(
            st.session_state.get("market_input", DEFAULT_STATE["market_input"])
        ).strip().upper(),
        "selected_period": st.session_state.get(
            "selected_period", DEFAULT_STATE["selected_period"]
        ),
        "show_beta": bool(st.session_state.get("show_beta", DEFAULT_STATE["show_beta"])),
        "show_sharpe": bool(
            st.session_state.get("show_sharpe", DEFAULT_STATE["show_sharpe"])
        ),
        "risk_free_rate": float(
            st.session_state.get(
                "risk_free_rate_pct", float(DEFAULT_STATE["risk_free_rate"]) * 100
            )
        )
        / 100,
        "peer_source_input": str(
            st.session_state.get(
                "peer_source_input", DEFAULT_STATE["peer_source_input"]
            )
        ).strip().upper(),
        "manual_peer_input": st.session_state.get(
            "manual_peer_input", DEFAULT_STATE["manual_peer_input"]
        ),
        "enable_auto_peers": bool(
            st.session_state.get(
                "enable_auto_peers", DEFAULT_STATE["enable_auto_peers"]
            )
        ),
        "include_peers_in_chart": bool(
            st.session_state.get(
                "include_peers_in_chart", DEFAULT_STATE["include_peers_in_chart"]
            )
        ),
        "peer_count": int(
            st.session_state.get("peer_count", DEFAULT_STATE["peer_count"])
        ),
    }


def _apply_state_to_session(state, configuration_name=None):
    normalized = _normalize_state(state)
    for key, value in normalized.items():
        st.session_state[key] = value
    st.session_state["risk_free_rate_pct"] = float(normalized["risk_free_rate"]) * 100
    st.session_state["active_config_name"] = configuration_name
    st.session_state["saved_config_selector"] = configuration_name or ""
    st.session_state["config_name_input"] = configuration_name or ""


def _build_saved_configuration(name, state, data_cache=None, existing_entry=None):
    timestamp = pd.Timestamp.utcnow().isoformat()
    return {
        "name": name,
        "state": _normalize_state(state),
        "data_cache": data_cache,
        "created_at": (
            (existing_entry or {}).get("created_at") if existing_entry else None
        )
        or timestamp,
        "updated_at": timestamp,
    }


def _saved_configuration_summary(configuration_entry):
    if not configuration_entry:
        return "No saved configuration selected."

    data_cache = configuration_entry.get("data_cache") or {}
    if not data_cache:
        return "Saved settings available. No price snapshot has been stored yet."

    symbol_count = len(data_cache.get("symbols", []))
    row_count = data_cache.get("rows", 0)
    start_label = data_cache.get("data_start") or "N/A"
    end_label = data_cache.get("data_end") or "N/A"
    updated_label = _format_timestamp(data_cache.get("stored_at"), include_time=True)
    return (
        f"{symbol_count} symbols | {row_count} rows | {start_label} to {end_label} "
        f"| cached {updated_label}"
    )


def _load_cached_prices(data_cache, expected_symbols):
    if not isinstance(data_cache, dict):
        return None

    frame = _deserialize_price_frame(data_cache.get("prices"))
    if frame is None:
        return None

    expected_symbols = dedupe_tickers(expected_symbols)
    if any(symbol not in frame.columns for symbol in expected_symbols):
        return None

    return frame[expected_symbols].sort_index()


def _download_close_prices(symbols, start, end):
    if pd.Timestamp(start) >= pd.Timestamp(end):
        return pd.DataFrame(columns=dedupe_tickers(symbols))

    data = yf.download(symbols, start=start, end=end, progress=False)
    if data.empty:
        return pd.DataFrame(columns=dedupe_tickers(symbols))

    prices = _ensure_price_frame(data["Close"])
    prices.columns = [str(column).upper() for column in prices.columns]
    return prices.sort_index()


def _combine_price_frames(frames, expected_symbols):
    valid_frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid_frames:
        return pd.DataFrame(columns=dedupe_tickers(expected_symbols))

    combined = pd.concat(valid_frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    ordered_columns = [symbol for symbol in dedupe_tickers(expected_symbols) if symbol in combined]
    return combined[ordered_columns]


def fetch_prices_with_cache(symbols, start, end, data_cache=None):
    expected_symbols = dedupe_tickers(symbols)
    cached_prices = _load_cached_prices(data_cache, expected_symbols)
    request_start = pd.Timestamp(start).normalize()
    request_end = pd.Timestamp(end).normalize()
    cache_message = None
    refreshed_cache = False

    if cached_prices is None or cached_prices.empty:
        return _download_close_prices(expected_symbols, start, end), {
            "message": None,
            "used_saved_cache": False,
            "refreshed_cache": False,
        }

    frames = []
    cache_start = cached_prices.index.min().normalize()
    cache_end = cached_prices.index.max().normalize()
    refreshed_parts = []

    if request_start < cache_start:
        earlier_prices = _download_close_prices(expected_symbols, request_start, cache_start)
        if not earlier_prices.empty:
            frames.append(earlier_prices)
            refreshed_parts.append("backfilled earlier history")
            refreshed_cache = True

    frames.append(cached_prices)

    if cache_end < request_end:
        later_start = cache_end + pd.Timedelta(days=1)
        later_prices = _download_close_prices(expected_symbols, later_start, end)
        if not later_prices.empty:
            frames.append(later_prices)
            refreshed_parts.append("added newer dates")
            refreshed_cache = True

    combined = _combine_price_frames(frames, expected_symbols)
    if not combined.empty:
        cache_message = "Loaded prices from the saved snapshot."
        if refreshed_parts:
            cache_message += f" Also {' and '.join(refreshed_parts)}."

    return combined, {
        "message": cache_message,
        "used_saved_cache": True,
        "refreshed_cache": refreshed_cache,
    }


def _build_price_cache_payload(symbols, prices, start, end):
    if prices is None or prices.empty:
        return None

    ordered_symbols = [symbol for symbol in dedupe_tickers(symbols) if symbol in prices.columns]
    if not ordered_symbols:
        return None

    cached_prices = prices[ordered_symbols].sort_index()
    return {
        "symbols": ordered_symbols,
        "requested_start": _format_timestamp(start),
        "requested_end": _format_timestamp(end),
        "data_start": _format_timestamp(cached_prices.index.min()),
        "data_end": _format_timestamp(cached_prices.index.max()),
        "rows": int(len(cached_prices.index)),
        "stored_at": pd.Timestamp.utcnow().isoformat(),
        "prices": _serialize_price_frame(cached_prices),
    }


def _data_cache_changed(existing_cache, new_cache):
    if not existing_cache and not new_cache:
        return False
    if not existing_cache or not new_cache:
        return True

    comparison_keys = ("symbols", "rows", "data_start", "data_end")
    return any(existing_cache.get(key) != new_cache.get(key) for key in comparison_keys)


def _build_security_display_names(tickers):
    raw_names = {}
    for ticker in tickers:
        metadata = fetch_security_metadata(ticker)
        raw_names[ticker] = metadata["name"] or ticker

    name_counts = {}
    for name in raw_names.values():
        name_counts[name] = name_counts.get(name, 0) + 1

    display_names = {}
    for ticker, name in raw_names.items():
        if name_counts.get(name, 0) > 1 and name != ticker:
            display_names[ticker] = f"{name} ({ticker})"
        else:
            display_names[ticker] = name

    return display_names


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
    normalized = re.sub(r"[^A-Z0-9]+", " ", normalized)

    token_map = {
        "IDX": ["INDEX"],
        "MKT": ["MARKET"],
        "STK": ["STOCK"],
        "TTL": ["TOTAL"],
        "TTLSTK": ["TOTAL", "STOCK"],
        "TTLSTOCK": ["TOTAL", "STOCK"],
        "TOT": ["TOTAL"],
        "INVMT": ["INVESTMENT"],
        "AMER": ["AMERICA"],
        "CO": ["COMPANY"],
    }
    share_class_tokens = {
        "A",
        "B",
        "C",
        "ADM",
        "F1",
        "F2",
        "F3",
        "I",
        "INV",
        "K",
        "L",
        "PL",
        "PLS",
        "PLUS",
        "R",
        "R2",
        "R3",
        "R4",
        "R5",
        "R6",
        "SEL",
        "SELECT",
        "Y",
        "Z",
        "529A",
        "529B",
        "529C",
        "529D",
        "529E",
        "529F",
    }

    normalized_tokens = []
    for token in normalized.split():
        if token in share_class_tokens:
            continue
        normalized_tokens.extend(token_map.get(token, [token]))

    if normalized_tokens[:3] == ["S", "P", "500"]:
        normalized_tokens = normalized_tokens[2:]

    return " ".join(normalized_tokens)


@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_security_metadata_cached(ticker):
    metadata = _empty_security_metadata(ticker)
    try:
        instrument = yf.Ticker(ticker)
    except Exception as exc:
        if _is_transient_lookup_error(exc):
            raise TransientLookupError(str(exc)) from exc
        metadata["error"] = str(exc)
        return metadata

    info = {}
    info_error = None
    try:
        info = instrument.info or {}
    except Exception as exc:
        info_error = str(exc)

    metadata["name"] = _first_present(
        info, ("longName", "shortName", "displayName", "name")
    ) or ticker
    metadata["quote_type"] = (
        _first_present(info, ("quoteType", "quote_type")) or ""
    ).upper() or None
    metadata["category"] = _first_present(info, ("category", "categoryName"))
    metadata["family"] = _first_present(info, ("fundFamily", "family"))
    metadata["sector"] = _first_present(info, ("sector", "sectorDisp"))
    metadata["industry"] = _first_present(info, ("industry", "industryDisp"))
    metadata["exchange"] = _first_present(info, ("exchange", "fullExchangeName"))
    metadata["market_cap"] = _coerce_assets(
        _first_present(info, ("marketCap", "intradayMarketCap"))
    )
    metadata["price"] = _coerce_assets(
        _first_present(info, ("regularMarketPrice", "currentPrice", "navPrice"))
    )
    metadata["change_percent"] = _coerce_assets(
        _first_present(
            info,
            ("regularMarketChangePercent", "regularMarketPercentChange", "ytdReturn"),
        )
    )
    metadata["currency"] = _first_present(info, ("currency", "financialCurrency"))

    if _is_equity_quote_type(metadata["quote_type"]):
        return metadata

    if metadata["category"] and metadata["family"]:
        return metadata

    fund_overview_error = None
    try:
        fund_overview = instrument.funds_data.fund_overview or {}
    except Exception as exc:
        fund_overview = {}
        fund_overview_error = str(exc)

    metadata["category"] = metadata["category"] or _first_present(
        fund_overview, ("categoryName", "category")
    )
    metadata["family"] = metadata["family"] or _first_present(
        fund_overview, ("family", "fundFamily")
    )

    if not metadata["category"] and not metadata["family"]:
        error_messages = [message for message in (info_error, fund_overview_error) if message]
        if error_messages:
            combined_error = " | ".join(error_messages)
            if _is_transient_lookup_error(combined_error):
                raise TransientLookupError(combined_error)
            metadata["error"] = combined_error

    return metadata


def fetch_security_metadata(ticker):
    try:
        return _fetch_security_metadata_cached(ticker)
    except TransientLookupError as exc:
        metadata = _empty_security_metadata(ticker)
        metadata["error"] = str(exc)
        return metadata


def _make_screen_query(query_class, category, exchange):
    clauses = [query_class("eq", ["categoryname", category])]
    if exchange:
        clauses.append(query_class("eq", ["exchange", exchange]))
    if len(clauses) == 1:
        return clauses[0]
    return query_class("and", clauses)


@st.cache_data(show_spinner=False, ttl=3600)
def _discover_peer_funds_cached(
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
        if _is_transient_lookup_error(exc):
            raise TransientLookupError(str(exc)) from exc
        return [], str(exc), None

    quotes = response.get("quotes", []) if isinstance(response, dict) else []
    candidates = []
    for quote in quotes:
        symbol = (quote.get("symbol") or quote.get("ticker") or "").upper()
        if not symbol or symbol == ticker.upper():
            continue

        metadata = None
        family = (
            quote.get("fundFamily")
            or quote.get("family")
            or quote.get("issuerName")
            or ""
        )
        display_name = (
            quote.get("longName")
            or quote.get("shortName")
            or quote.get("displayName")
            or symbol
        )
        quote_category = _first_present(quote, ("categoryName", "category"))
        inferred_provider = _infer_provider_from_fund_name(display_name)
        if (not family and not inferred_provider) or not quote_category:
            metadata = fetch_security_metadata(symbol)
            family = metadata["family"] or family
            quote_category = metadata["category"] or quote_category
            display_name = metadata["name"] or display_name

        provider = _normalize_provider_name(family) or inferred_provider
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
                "Category": quote_category or category,
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


def discover_peer_funds(
    ticker, category, quote_type, exchange, source_family, peer_limit
):
    try:
        return _discover_peer_funds_cached(
            ticker, category, quote_type, exchange, source_family, peer_limit
        )
    except TransientLookupError as exc:
        return [], str(exc), None


def _coerce_equity_market_cap(quote):
    return _coerce_assets(
        _first_present(
            quote,
            (
                "marketCap",
                "intradaymarketcap",
                "intradayMarketCap",
                "lastclosemarketcap.lasttwelvemonths",
            ),
        )
    )


def _score_equity_peer(candidate, source_market_cap):
    market_cap = candidate["MarketCap"]
    if market_cap is None or source_market_cap is None:
        return (1, 1, 999.0, candidate["Ticker"])

    market_cap = float(market_cap)
    source_market_cap = float(source_market_cap)
    if (
        not math.isfinite(market_cap)
        or not math.isfinite(source_market_cap)
        or market_cap <= 0
        or source_market_cap <= 0
    ):
        return (1, 1, 999.0, candidate["Ticker"])

    size_ratio = market_cap / source_market_cap
    if size_ratio <= 0 or not math.isfinite(size_ratio):
        return (1, 1, 999.0, candidate["Ticker"])

    size_gap = abs(math.log(size_ratio))
    if 0.5 <= size_ratio <= 8:
        size_band = 0
    elif 0.25 <= size_ratio <= 15:
        size_band = 1
    else:
        size_band = 2

    exchange_band = 0 if candidate["PrimaryExchange"] else 1
    return (exchange_band, size_band, size_gap, candidate["Ticker"])


@st.cache_data(show_spinner=False, ttl=3600)
def _discover_peer_stocks_cached(
    ticker, sector, industry, exchange, source_market_cap, peer_limit
):
    peer_field = "industry" if industry else "sector"
    peer_value = industry or sector
    if not peer_value:
        return [], "No industry or sector was found for the selected stock.", None

    if not hasattr(yf, "EquityQuery") or not hasattr(yf, "screen"):
        return [], "This yfinance version does not expose stock screening.", None

    request_size = min(max(peer_limit * 25, 60), 250)
    try:
        response = yf.screen(
            yf.EquityQuery("eq", [peer_field, peer_value]),
            size=request_size,
            sortField="ticker",
            sortAsc=True,
        )
    except Exception as exc:
        if _is_transient_lookup_error(exc):
            raise TransientLookupError(str(exc)) from exc
        return [], str(exc), None

    quotes = response.get("quotes", []) if isinstance(response, dict) else []
    candidates = []
    for quote in quotes:
        symbol = (quote.get("symbol") or quote.get("ticker") or "").upper()
        if not symbol or symbol == ticker.upper():
            continue

        quote_exchange = (quote.get("exchange") or "").upper()
        market_cap = _coerce_equity_market_cap(quote)
        price = _coerce_assets(
            _first_present(quote, ("regularMarketPrice", "intradayprice", "currentPrice"))
        )
        change_percent = _coerce_assets(
            _first_present(quote, ("regularMarketChangePercent", "percentchange"))
        )
        display_name = (
            quote.get("longName")
            or quote.get("shortName")
            or quote.get("displayName")
            or symbol
        )
        candidates.append(
            {
                "Ticker": symbol,
                "Name": display_name,
                "Industry": quote.get("industry") or industry or "N/A",
                "Sector": quote.get("sector") or sector or "N/A",
                "Exchange": quote_exchange or exchange or "N/A",
                "MarketCap": market_cap,
                "Price": price,
                "ChangePercent": change_percent,
                "Currency": quote.get("currency") or "USD",
                "PrimaryExchange": (
                    quote_exchange in MAJOR_US_EQUITY_EXCHANGES
                    or quote_exchange == (exchange or "").upper()
                ),
            }
        )

    candidates.sort(key=lambda candidate: _score_equity_peer(candidate, source_market_cap))

    selected_peers = []
    selection_note = None
    phase_counts = []

    def add_phase(*, require_primary_exchange, max_size_band):
        added = 0
        for candidate in candidates:
            if len(selected_peers) >= peer_limit:
                break
            if candidate in selected_peers:
                continue

            exchange_band, size_band, _, _ = _score_equity_peer(
                candidate, source_market_cap
            )
            if require_primary_exchange and exchange_band != 0:
                continue
            if max_size_band is not None and size_band > max_size_band:
                continue

            selected_peers.append(candidate)
            added += 1
        return added

    phase_counts.append(add_phase(require_primary_exchange=True, max_size_band=0))
    phase_counts.append(add_phase(require_primary_exchange=True, max_size_band=1))
    phase_counts.append(add_phase(require_primary_exchange=True, max_size_band=None))
    phase_counts.append(add_phase(require_primary_exchange=False, max_size_band=1))
    phase_counts.append(add_phase(require_primary_exchange=False, max_size_band=None))

    if len(selected_peers) < peer_limit:
        selection_note = (
            f"Only {len(selected_peers)} related stocks were available for the selected screening criteria."
        )
    elif phase_counts[2] > 0 or phase_counts[3] > 0 or phase_counts[4] > 0:
        selection_note = (
            "Exchange and market-cap matching were relaxed to fill the requested peer count."
        )

    return selected_peers, None, selection_note


def discover_peer_stocks(
    ticker, sector, industry, exchange, source_market_cap, peer_limit
):
    try:
        return _discover_peer_stocks_cached(
            ticker, sector, industry, exchange, source_market_cap, peer_limit
        )
    except TransientLookupError as exc:
        return [], str(exc), None


def build_peer_table(manual_peers, auto_peers, peer_source, peer_source_metadata):
    if _is_equity_quote_type((peer_source_metadata or {}).get("quote_type")):
        return build_stock_peer_table(manual_peers, auto_peers, peer_source)
    return build_fund_peer_table(manual_peers, auto_peers, peer_source)


def build_fund_peer_table(manual_peers, auto_peers, peer_source):
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


def build_stock_peer_table(manual_peers, auto_peers, peer_source):
    rows = []
    for ticker in manual_peers:
        metadata = fetch_security_metadata(ticker)
        rows.append(
            {
                "Ticker": ticker,
                "Name": metadata["name"],
                "Industry": metadata["industry"] or "N/A",
                "Sector": metadata["sector"] or "N/A",
                "Last Price": _format_price(metadata["price"], metadata["currency"]),
                "Daily Change": _format_percent(metadata["change_percent"]),
                "Market Cap": _format_assets(metadata["market_cap"]),
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
                "Industry": peer["Industry"] or "N/A",
                "Sector": peer["Sector"] or "N/A",
                "Last Price": _format_price(peer["Price"], peer["Currency"]),
                "Daily Change": _format_percent(peer["ChangePercent"]),
                "Market Cap": _format_assets(peer["MarketCap"]),
            }
        )

    return pd.DataFrame(rows)


st.title("Stock Performance Tracker")

if "logged_in_user" not in st.session_state:
    st.session_state["logged_in_user"] = None
if "login_feedback" not in st.session_state:
    st.session_state["login_feedback"] = None
if "login_username" not in st.session_state:
    st.session_state["login_username"] = ""
if "login_password" not in st.session_state:
    st.session_state["login_password"] = ""

with st.sidebar.expander("User sign-in", expanded=True):
    if st.session_state["logged_in_user"]:
        st.success(f"Signed in as {st.session_state['logged_in_user']}")
        if st.button("Sign Out", key="sign_out"):
            st.session_state["logged_in_user"] = None
            st.session_state["login_feedback"] = {
                "level": "info",
                "message": "Signed out successfully."
            }
            st.rerun()
    else:
        st.text_input("Username", key="login_username")
        st.text_input("Password", type="password", key="login_password")
        auth_col1, auth_col2 = st.columns(2)
        with auth_col1:
            if st.button("Sign In", key="sign_in"):
                user = _authenticate_user(
                    st.session_state["login_username"],
                    st.session_state["login_password"],
                )
                if user:
                    st.session_state["logged_in_user"] = user.username
                    st.session_state["login_feedback"] = {
                        "level": "success",
                        "message": f"Signed in as {user.username}."
                    }
                    st.rerun()
                else:
                    st.session_state["login_feedback"] = {
                        "level": "error",
                        "message": "Invalid username or password."
                    }
        with auth_col2:
            if st.button("Register", key="register"):
                user = _create_user(
                    st.session_state["login_username"],
                    st.session_state["login_password"],
                )
                if user:
                    st.session_state["logged_in_user"] = user.username
                    st.session_state["login_feedback"] = {
                        "level": "success",
                        "message": f"Account created and signed in as {user.username}."
                    }
                    st.rerun()
                else:
                    st.session_state["login_feedback"] = {
                        "level": "warning",
                        "message": "Choose a unique username and a non-empty password."
                    }

    if st.session_state["login_feedback"]:
        getattr(
            st,
            st.session_state["login_feedback"].get("level", "info"),
        )(st.session_state["login_feedback"]["message"])

state_store = load_application_store()
saved_state = state_store["current_state"]
saved_configurations = state_store["saved_configurations"]
selected_configuration = state_store["selected_configuration"]

current_store_identity = (
    f"{st.session_state.get('logged_in_user') or 'guest'}:"
    f"{selected_configuration or ''}"
)
if st.session_state.get("loaded_store_identity") != current_store_identity:
    _apply_state_to_session(saved_state, selected_configuration)
    st.session_state["loaded_store_identity"] = current_store_identity

period_options = {
    "5 Years": pd.DateOffset(years=5),
    "3 Years": pd.DateOffset(years=3),
    "1 Year": pd.DateOffset(years=1),
    "6 Months": pd.DateOffset(months=6),
    "3 Months": pd.DateOffset(months=3),
    "1 Month": pd.DateOffset(months=1),
}
period_labels = list(period_options.keys())

for state_key, default_value in saved_state.items():
    if state_key not in st.session_state:
        st.session_state[state_key] = default_value

if "risk_free_rate_pct" not in st.session_state:
    st.session_state["risk_free_rate_pct"] = float(saved_state["risk_free_rate"]) * 100
if "active_config_name" not in st.session_state:
    st.session_state["active_config_name"] = selected_configuration
if "saved_config_selector" not in st.session_state:
    st.session_state["saved_config_selector"] = selected_configuration or ""
if "config_name_input" not in st.session_state:
    st.session_state["config_name_input"] = selected_configuration or ""

pending_configuration = st.session_state.pop("pending_configuration", None)
if pending_configuration:
    _apply_state_to_session(
        pending_configuration.get("state"),
        pending_configuration.get("name"),
    )

if st.session_state["active_config_name"] not in saved_configurations:
    st.session_state["active_config_name"] = None
if st.session_state["saved_config_selector"] not in saved_configurations:
    st.session_state["saved_config_selector"] = ""
if st.session_state.get("selected_period") not in period_labels:
    st.session_state["selected_period"] = DEFAULT_STATE["selected_period"]

initial_tickers = parse_ticker_input(
    st.session_state.get("tickers_input", DEFAULT_STATE["tickers_input"])
)
if not str(st.session_state.get("peer_source_input", "")).strip():
    if initial_tickers:
        st.session_state["peer_source_input"] = next(
            (ticker for ticker in initial_tickers if not ticker.startswith("^")),
            initial_tickers[0],
        )
    else:
        st.session_state["peer_source_input"] = DEFAULT_STATE["peer_source_input"]

st.sidebar.header("Saved Configurations")
feedback = st.session_state.pop("config_feedback", None)
if feedback:
    getattr(st.sidebar, feedback.get("level", "info"))(feedback["message"])

saved_config_names = sorted(saved_configurations)
selected_saved_name = st.sidebar.selectbox(
    "Saved named configuration",
    options=[""] + saved_config_names,
    key="saved_config_selector",
    format_func=lambda value: value or "Current unsaved configuration",
    help="Choose a saved configuration to load into the sidebar controls below.",
)
selected_config_entry = (
    saved_configurations.get(selected_saved_name) if selected_saved_name else None
)
if selected_config_entry:
    st.sidebar.caption(_saved_configuration_summary(selected_config_entry))

st.sidebar.text_input(
    "Name for saving",
    key="config_name_input",
    placeholder="e.g. Retirement Funds",
    help="Used when saving the current sidebar settings as a new named configuration.",
)
saved_button_col1, saved_button_col2 = st.sidebar.columns(2)
with saved_button_col1:
    load_selected_clicked = st.button(
        "Load Selected",
        disabled=not selected_saved_name,
        use_container_width=True,
    )
with saved_button_col2:
    save_new_clicked = st.button("Save As New", use_container_width=True)
update_loaded_clicked = st.sidebar.button(
    "Update Loaded Config",
    disabled=not st.session_state.get("active_config_name"),
    use_container_width=True,
    help="Overwrites the currently loaded configuration with any edits and refreshes its saved data snapshot.",
)
sidebar_status = st.sidebar.empty()

if load_selected_clicked:
    if not selected_saved_name or selected_saved_name not in saved_configurations:
        st.session_state["config_feedback"] = {
            "level": "warning",
            "message": "Select a saved configuration to load.",
        }
    else:
        st.session_state["pending_configuration"] = {
            "name": selected_saved_name,
            "state": saved_configurations[selected_saved_name]["state"],
        }
        state_store["current_state"] = _normalize_state(
            saved_configurations[selected_saved_name]["state"]
        )
        state_store["saved_configurations"] = saved_configurations
        state_store["selected_configuration"] = selected_saved_name
        save_application_store(state_store)
        st.session_state["config_feedback"] = {
            "level": "success",
            "message": f'Loaded "{selected_saved_name}".',
        }
    st.rerun()

st.sidebar.header("Configuration Details")
tickers_input = st.sidebar.text_area(
    "Edit Tickers (comma-separated)",
    key="tickers_input",
    height=110,
    help="""Enter stock tickers separated by commas. Some common indexes
            include the DJIA (^DJI), S&P 500 (^GSPC), Russell 2000 (^RUT),
            and NASDAQ (^IXIC). E.g. ^dji, ^rut, ^ixic, vtsax, fcntx,
            ponax, ORCL, MSFT,""",
)
tickers = parse_ticker_input(tickers_input)

market_input = st.sidebar.text_input(
    "Baseline Index",
    key="market_input",
    help="""Enter the ticker for the baseline index.
            Common ones: ^GSPC (S&P 500), ^DJI (Dow Jones),
            ^IXIC (NASDAQ), ^RUT (Russell 2000)""",
)

st.sidebar.header("Peer Lookup")
peer_source_input = st.sidebar.text_input(
    "Peer Source Security",
    key="peer_source_input",
    help="Fund, ETF, or stock ticker used for peer discovery.",
)
manual_peer_input = st.sidebar.text_area(
    "Manual Peer Overrides",
    key="manual_peer_input",
    height=80,
    help="Optional peers to add manually. These are appended to the chart if enabled below.",
)
enable_auto_peers = st.sidebar.checkbox(
    "Find Automatic Peers",
    key="enable_auto_peers",
    help="Uses Yahoo Finance categories for funds and industries for stocks to suggest related peers.",
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
selected_period = st.sidebar.selectbox(
    "Select period:",
    period_labels,
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
    step=0.25,
    help="Used for Sharpe ratio calculations in the table and optional plot.",
)

current_state = _build_current_state()
market = current_state["market_input"]
peer_source = current_state["peer_source_input"]
manual_peers = parse_ticker_input(current_state["manual_peer_input"])
risk_free_rate = current_state["risk_free_rate"]
active_config_name = st.session_state.get("active_config_name")
active_config_entry = (
    saved_configurations.get(active_config_name) if active_config_name else None
)
config_is_dirty = bool(
    active_config_entry
    and current_state != _normalize_state(active_config_entry.get("state"))
)

if active_config_entry:
    status_text = f'Loaded configuration: "{active_config_name}"'
    if config_is_dirty:
        sidebar_status.caption(f"{status_text} | unsaved edits")
    else:
        sidebar_status.caption(f"{status_text} | synced with saved version")
else:
    sidebar_status.caption("Working in an unsaved configuration.")

state_store["current_state"] = current_state
state_store["saved_configurations"] = saved_configurations
state_store["selected_configuration"] = active_config_name if active_config_entry else None
save_application_store(state_store)

if not tickers:
    st.error("Please enter at least one ticker symbol.")
    st.stop()

peer_source_metadata = None
auto_peers = []
peer_lookup_error = None
peer_lookup_note = None
requested_auto_peer_count = peer_count

if peer_source:
    peer_source_metadata = fetch_security_metadata(peer_source)
    if enable_auto_peers:
        if _is_fund_quote_type(peer_source_metadata["quote_type"]):
            auto_peers, peer_lookup_error, peer_lookup_note = discover_peer_funds(
                peer_source,
                peer_source_metadata["category"],
                peer_source_metadata["quote_type"],
                peer_source_metadata["exchange"],
                peer_source_metadata["family"],
                peer_count,
            )
        elif _is_equity_quote_type(peer_source_metadata["quote_type"]):
            auto_peers, peer_lookup_error, peer_lookup_note = discover_peer_stocks(
                peer_source,
                peer_source_metadata["sector"],
                peer_source_metadata["industry"],
                peer_source_metadata["exchange"],
                peer_source_metadata["market_cap"],
                peer_count,
            )
        else:
            peer_lookup_error = (
                "Automatic peer lookup currently supports mutual funds, ETFs, and stocks."
            )
        excluded = {peer_source, *manual_peers}
        auto_peers = [
            peer for peer in auto_peers if peer["Ticker"] not in excluded
        ][:peer_count]

comparison_tickers = list(tickers)
if include_peers_in_chart:
    comparison_tickers = dedupe_tickers(
        comparison_tickers
        + manual_peers
        + [peer["Ticker"] for peer in auto_peers]
    )
else:
    comparison_tickers = dedupe_tickers(comparison_tickers)

end = pd.Timestamp.today()
start = end - period_options[selected_period]
required_symbols = dedupe_tickers(comparison_tickers + [market])
download_symbols = list(required_symbols)

active_config_matches_current = bool(
    active_config_entry
    and current_state == _normalize_state(active_config_entry.get("state"))
)
saved_data_cache = (
    active_config_entry.get("data_cache") if active_config_matches_current else None
)

st.sidebar.info(f'Preparing data for: {", ".join(download_symbols)}')
prices = None
price_fetch_error = None
price_fetch_info = {
    "message": None,
    "used_saved_cache": False,
    "refreshed_cache": False,
}

try:
    prices, price_fetch_info = fetch_prices_with_cache(
        download_symbols,
        start,
        end,
        saved_data_cache,
    )
    missing_symbols = [
        symbol for symbol in required_symbols if symbol not in prices.columns
    ]
    if missing_symbols:
        price_fetch_error = f"Missing price data for: {', '.join(missing_symbols)}"
    else:
        prices = prices[required_symbols].dropna(how="all")
        if prices.empty:
            price_fetch_error = (
                "No price data was returned for the selected symbols and period."
            )
except Exception as exc:
    price_fetch_error = f"Error downloading data: {exc}"

current_data_cache = (
    _build_price_cache_payload(required_symbols, prices, start, end)
    if price_fetch_error is None
    else None
)
auto_cache_message = None
if (
    active_config_matches_current
    and current_data_cache
    and _data_cache_changed(active_config_entry.get("data_cache"), current_data_cache)
):
    saved_configurations[active_config_name] = _build_saved_configuration(
        active_config_name,
        current_state,
        current_data_cache,
        active_config_entry,
    )
    active_config_entry = saved_configurations[active_config_name]
    state_store["saved_configurations"] = saved_configurations
    state_store["selected_configuration"] = active_config_name
    save_application_store(state_store)
    auto_cache_message = (
        f'Saved data snapshot refreshed through {current_data_cache["data_end"]}.'
    )

if save_new_clicked:
    new_name = str(st.session_state.get("config_name_input", "")).strip()
    if not new_name:
        st.sidebar.warning("Enter a configuration name before saving.")
    elif new_name in saved_configurations:
        st.sidebar.warning(
            f'"{new_name}" already exists. Load it and use Update Loaded Config, or choose a new name.'
        )
    else:
        saved_configurations[new_name] = _build_saved_configuration(
            new_name,
            current_state,
            current_data_cache,
        )
        st.session_state["pending_configuration"] = {
            "name": new_name,
            "state": current_state,
        }
        state_store["saved_configurations"] = saved_configurations
        state_store["selected_configuration"] = new_name
        save_application_store(state_store)
        if current_data_cache:
            message = f'Saved "{new_name}" with the current price snapshot.'
        else:
            message = (
                f'Saved "{new_name}" without a price snapshot because the latest download did not succeed.'
            )
        st.session_state["config_feedback"] = {
            "level": "success",
            "message": message,
        }
        st.rerun()

if update_loaded_clicked:
    target_name = st.session_state.get("active_config_name")
    if not target_name or target_name not in saved_configurations:
        st.sidebar.warning("Load a saved configuration before updating it.")
    else:
        saved_configurations[target_name] = _build_saved_configuration(
            target_name,
            current_state,
            current_data_cache,
            saved_configurations.get(target_name),
        )
        st.session_state["pending_configuration"] = {
            "name": target_name,
            "state": current_state,
        }
        state_store["saved_configurations"] = saved_configurations
        state_store["selected_configuration"] = target_name
        save_application_store(state_store)
        if current_data_cache:
            message = f'Updated "{target_name}" and refreshed its saved price snapshot.'
        else:
            message = (
                f'Updated "{target_name}" but could not refresh its price snapshot because the latest download failed.'
            )
        st.session_state["config_feedback"] = {
            "level": "success",
            "message": message,
        }
        st.rerun()

if price_fetch_info["message"]:
    st.sidebar.caption(price_fetch_info["message"])
if auto_cache_message:
    st.sidebar.caption(auto_cache_message)

if price_fetch_error:
    st.error(price_fetch_error)
    st.stop()

security_display_names = _build_security_display_names(comparison_tickers)
prices_norm = prices.div(prices.ffill().bfill().iloc[0]).mul(100)

fig, ax = plt.subplots(figsize=(12, 6))
ticker_colors = {}
for ticker in comparison_tickers:
    series = prices_norm[ticker].dropna()
    if not series.empty:
        price_line = ax.plot(
            series.index,
            series,
            label=security_display_names.get(ticker, ticker),
            linewidth=2,
        )[0]
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
                "Name": security_display_names.get(ticker, ticker),
                "Beta": round(beta, 2),
                "Std Dev of Returns": round(std_dev, 4),
                "Sharpe Ratio": (
                    round(sharpe_ratio, 2) if sharpe_ratio is not None else "N/A"
                ),
            }
        )
    else:
        metrics_data.append(
            {
                "Ticker": ticker,
                "Name": security_display_names.get(ticker, ticker),
                "Beta": "N/A",
                "Std Dev of Returns": "N/A",
                "Sharpe Ratio": "N/A",
            }
        )
metrics_df = pd.DataFrame(metrics_data)
st.dataframe(metrics_df, width="stretch")

peer_source_mode = "unknown"
if peer_source_metadata:
    if _is_fund_quote_type(peer_source_metadata["quote_type"]):
        peer_source_mode = "fund"
    elif _is_equity_quote_type(peer_source_metadata["quote_type"]):
        peer_source_mode = "stock"

st.subheader("Peers")
if not peer_source:
    st.info(
        "Enter a fund, ETF, or stock ticker in Peer Source Security to discover related peers."
    )
elif peer_source_metadata and peer_source_metadata["error"]:
    st.warning(
        f"Could not load peer source metadata for {peer_source}: {peer_source_metadata['error']}"
    )
else:
    source_name = (
        (peer_source_metadata["name"] if peer_source_metadata else None) or peer_source
    )
    if peer_source_mode == "fund":
        category_label = (
            peer_source_metadata["category"] if peer_source_metadata else None
        ) or "N/A"
        family_label = (
            peer_source_metadata["family"] if peer_source_metadata else None
        ) or "N/A"
        st.caption(
            f"Peer source: {peer_source} ({source_name}) | Category: {category_label} | Family: {family_label}"
        )
    elif peer_source_mode == "stock":
        industry_label = (
            peer_source_metadata["industry"] if peer_source_metadata else None
        ) or "N/A"
        sector_label = (
            peer_source_metadata["sector"] if peer_source_metadata else None
        ) or "N/A"
        st.caption(
            f"Peer source: {peer_source} ({source_name}) | Industry: {industry_label} | Sector: {sector_label}"
        )
    else:
        st.caption(f"Peer source: {peer_source} ({source_name})")

    if auto_peers:
        if peer_source_mode == "fund":
            st.caption(
                f"Showing {len(auto_peers)} automatic peers for a target of {requested_auto_peer_count}. "
                "The list keeps providers distinct where possible and leans toward larger AUM."
            )
        elif peer_source_mode == "stock":
            st.caption(
                f"Showing {len(auto_peers)} automatic stock peers for a target of {requested_auto_peer_count}. "
                "The list prefers same-industry companies on primary exchanges and leans toward similar market caps."
            )
        else:
            st.caption(
                f"Showing {len(auto_peers)} automatic peers for a target of {requested_auto_peer_count}."
            )
        if peer_lookup_note:
            st.caption(peer_lookup_note)
    elif enable_auto_peers and not peer_lookup_error:
        st.caption(
            f"No automatic peers were available for a target of {requested_auto_peer_count}."
        )
    if peer_lookup_error and enable_auto_peers:
        st.info(f"Automatic peer lookup is unavailable right now: {peer_lookup_error}")

    peer_table = build_peer_table(
        manual_peers, auto_peers, peer_source, peer_source_metadata
    )
    if peer_table.empty:
        st.info("No peers to show yet. Add manual peers or enable automatic peer discovery.")
    else:
        st.dataframe(peer_table, width="stretch", hide_index=True)

st.subheader("Normalized Price Data")
st.dataframe(prices_norm[comparison_tickers], width="stretch")
