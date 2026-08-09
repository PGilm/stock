import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from peewee import (
    CharField,
    DateTimeField,
    FloatField,
    Model,
    SqliteDatabase,
    TextField,
)
from platformdirs import user_config_dir

APP_NAME = "pgStocks"
DEFAULT_CACHE_DB = Path(user_config_dir(APP_NAME, APP_NAME)) / "pgStocks_ticker_cache.db"
DATABASE = SqliteDatabase(str(DEFAULT_CACHE_DB), pragmas={"foreign_keys": 1})
def _utc_now():
    return dt.datetime.now(dt.UTC).replace(tzinfo=None)


class TickerMetadataCacheEntryModel(Model):
    symbol = CharField(unique=True)
    name = CharField(null=True)
    quote_type = CharField(null=True)
    category = CharField(null=True)
    family = CharField(null=True)
    sector = CharField(null=True)
    industry = CharField(null=True)
    exchange = CharField(null=True)
    market_cap = FloatField(null=True)
    price = FloatField(null=True)
    change_percent = FloatField(null=True)
    currency = CharField(null=True)
    error = TextField(null=True)
    metadata_json = TextField(null=True)
    last_checked_at = DateTimeField(null=True)
    created_at = DateTimeField(default=_utc_now)
    updated_at = DateTimeField(default=_utc_now)

    class Meta:
        database = DATABASE


class TickerHistoryCacheEntryModel(Model):
    symbol = CharField(unique=True)
    history_json = TextField(null=True)
    last_record_date = DateTimeField(null=True)
    last_checked_at = DateTimeField(null=True)
    latest_adjusted_close = FloatField(null=True)
    created_at = DateTimeField(default=_utc_now)
    updated_at = DateTimeField(default=_utc_now)

    class Meta:
        database = DATABASE


@dataclass
class TickerCacheEntry:
    symbol: str
    name: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    history_frame: Optional[pd.DataFrame] = None
    last_record_date: Optional[pd.Timestamp] = None
    last_checked_at: Optional[pd.Timestamp] = None
    latest_adjusted_close: Optional[float] = None
    created_at: Optional[pd.Timestamp] = None
    updated_at: Optional[pd.Timestamp] = None


def initialize_ticker_cache_database(db_path=None):
    target_path = Path(db_path) if db_path else DEFAULT_CACHE_DB
    target_path.parent.mkdir(parents=True, exist_ok=True)
    global DATABASE
    DATABASE.init(str(target_path), pragmas={"foreign_keys": 1})
    DATABASE.connect(reuse_if_open=True)
    DATABASE.create_tables([TickerMetadataCacheEntryModel, TickerHistoryCacheEntryModel])
    return str(target_path)


def _normalize_symbol(symbol):
    value = (symbol or "").strip().upper()
    return value or None


def _coerce_timestamp(value):
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        timestamp = value
    elif isinstance(value, dt.datetime):
        timestamp = pd.Timestamp(value)
    else:
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError):
            return None
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp.normalize()


def _format_datetime(value):
    if value is None:
        return None
    timestamp = _coerce_timestamp(value)
    if timestamp is None:
        return None
    return timestamp.to_pydatetime()


def _ensure_history_frame(frame):
    if frame is None:
        return None
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    if not isinstance(frame, pd.DataFrame):
        return None
    if frame.empty:
        return None
    sanitized = frame.copy()
    sanitized.index = pd.to_datetime(sanitized.index)
    sanitized = sanitized.sort_index()
    return sanitized


def _serialize_history_frame(frame):
    frame = _ensure_history_frame(frame)
    if frame is None:
        return None

    serializable = frame.sort_index().copy()
    serializable.index = pd.to_datetime(serializable.index).strftime("%Y-%m-%d")
    rows = []
    for row in serializable.itertuples(index=False, name=None):
        rows.append([None if pd.isna(value) else float(value) for value in row])

    return json.dumps(
        {
            "index": serializable.index.tolist(),
            "columns": [str(column) for column in serializable.columns],
            "data": rows,
        }
    )


def _deserialize_history_frame(payload):
    if not payload:
        return None
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            return None
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


def _coerce_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pick_adj_close_column(frame):
    if frame is None or frame.empty:
        return None
    lowered_columns = {str(column).lower(): column for column in frame.columns}
    for preferred in ["adj close", "adj_close", "adjusted close", "adjusted_close", "close", "close price"]:
        if preferred in lowered_columns:
            return lowered_columns[preferred]
    return frame.columns[0]


def _latest_adjusted_close(frame):
    if frame is None or frame.empty:
        return None
    column = _pick_adj_close_column(frame)
    if column is None:
        return None
    latest_index = frame.index.max()
    if latest_index is None:
        return None
    value = frame.loc[latest_index, column]
    return _coerce_float(value)


def _filter_history_frame(frame, start, end):
    if frame is None or frame.empty:
        return None
    return frame.loc[(frame.index >= start) & (frame.index <= end)].sort_index()


def _latest_weekday(value):
    timestamp = pd.Timestamp(value).normalize()
    while timestamp.weekday() >= 5:
        timestamp = timestamp - pd.Timedelta(days=1)
    return timestamp


def _combine_history_frames(frames):
    valid_frames = [
        _ensure_history_frame(frame)
        for frame in frames
        if frame is not None and not frame.empty
    ]
    valid_frames = [frame for frame in valid_frames if frame is not None]
    if not valid_frames:
        return None
    combined = pd.concat(valid_frames, sort=False).sort_index()
    return combined[~combined.index.duplicated(keep="last")]


def _price_column_frame(frame):
    if frame is None or frame.empty:
        return None

    if isinstance(frame.columns, pd.MultiIndex):
        for price_label in ("Adj Close", "Close"):
            if price_label in frame.columns.get_level_values(0):
                selected = frame.xs(price_label, axis=1, level=0, drop_level=False)
                if isinstance(selected, pd.DataFrame) and not selected.empty:
                    selected = selected.iloc[:, :1]
                    selected.columns = ["Adj Close"]
                    return selected
            if price_label in frame.columns.get_level_values(-1):
                selected = frame.xs(price_label, axis=1, level=-1, drop_level=False)
                if isinstance(selected, pd.DataFrame) and not selected.empty:
                    selected = selected.iloc[:, :1]
                    selected.columns = ["Adj Close"]
                    return selected

        selected = frame.iloc[:, :1].copy()
        selected.columns = ["Adj Close"]
        return selected

    if "Close" in frame.columns and "Adj Close" not in frame.columns:
        frame = frame.rename(columns={"Close": "Adj Close"})
    if "Adj Close" in frame.columns:
        return frame[["Adj Close"]]

    selected = frame.iloc[:, :1].copy()
    selected.columns = ["Adj Close"]
    return selected


def _build_entry_from_model(metadata_record, history_record):
    if metadata_record is None and history_record is None:
        return None

    metadata_payload = {}
    if metadata_record is not None and metadata_record.metadata_json:
        try:
            metadata_payload = json.loads(metadata_record.metadata_json)
        except (TypeError, json.JSONDecodeError):
            metadata_payload = {}

    history_frame = _deserialize_history_frame(history_record.history_json) if history_record is not None else None
    return TickerCacheEntry(
        symbol=(metadata_record or history_record).symbol,
        name=(metadata_record.name if metadata_record is not None else None),
        metadata=metadata_payload,
        history_frame=history_frame,
        last_record_date=_coerce_timestamp(history_record.last_record_date) if history_record is not None else None,
        last_checked_at=_coerce_timestamp((metadata_record.last_checked_at if metadata_record is not None else None) or (history_record.last_checked_at if history_record is not None else None)),
        latest_adjusted_close=_coerce_float(history_record.latest_adjusted_close) if history_record is not None else None,
        created_at=_coerce_timestamp((metadata_record.created_at if metadata_record is not None else None) or (history_record.created_at if history_record is not None else None)),
        updated_at=_coerce_timestamp((metadata_record.updated_at if metadata_record is not None else None) or (history_record.updated_at if history_record is not None else None)),
    )


def load_ticker_cache_entry(symbol):
    normalized_symbol = _normalize_symbol(symbol)
    if not normalized_symbol:
        return None
    metadata_record = TickerMetadataCacheEntryModel.get_or_none(TickerMetadataCacheEntryModel.symbol == normalized_symbol)
    history_record = TickerHistoryCacheEntryModel.get_or_none(TickerHistoryCacheEntryModel.symbol == normalized_symbol)
    if metadata_record is None and history_record is None:
        return None
    return _build_entry_from_model(metadata_record, history_record)


def refresh_ticker_metadata(symbol, metadata, *, last_checked_at=None):
    return upsert_ticker_cache_entry(
        symbol,
        history_frame=None,
        metadata=metadata,
        last_checked_at=last_checked_at,
    )


def refresh_ticker_history(symbol, history_frame, *, metadata=None, last_checked_at=None):
    return upsert_ticker_cache_entry(
        symbol,
        history_frame=history_frame,
        metadata=metadata,
        last_checked_at=last_checked_at,
    )


def upsert_ticker_cache_entry(symbol, *, history_frame=None, metadata=None, last_checked_at=None):
    normalized_symbol = _normalize_symbol(symbol)
    if not normalized_symbol:
        return None

    now = _utc_now()
    metadata_record = TickerMetadataCacheEntryModel.get_or_none(TickerMetadataCacheEntryModel.symbol == normalized_symbol)
    history_record = TickerHistoryCacheEntryModel.get_or_none(TickerHistoryCacheEntryModel.symbol == normalized_symbol)
    existing_history = None
    if history_record is not None:
        existing_history = _deserialize_history_frame(history_record.history_json)

    combined_history = None
    should_update_history = history_frame is not None
    if should_update_history:
        incoming_history = _ensure_history_frame(history_frame)
        if incoming_history is None:
            combined_history = existing_history
        elif existing_history is None:
            combined_history = incoming_history
        else:
            combined = pd.concat([existing_history, incoming_history], sort=False).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
            combined_history = combined
    elif existing_history is not None:
        combined_history = existing_history

    if metadata_record is None:
        metadata_record = TickerMetadataCacheEntryModel.create(symbol=normalized_symbol, created_at=now, updated_at=now)
    if history_record is None:
        history_record = TickerHistoryCacheEntryModel.create(symbol=normalized_symbol, created_at=now, updated_at=now)

    metadata_payload = None
    if metadata_record.metadata_json:
        try:
            metadata_payload = json.loads(metadata_record.metadata_json)
        except (TypeError, json.JSONDecodeError):
            metadata_payload = {}
    if isinstance(metadata, dict):
        metadata_payload = dict(metadata_payload or {})
        metadata_payload.update(metadata)
    elif metadata is None:
        metadata_payload = metadata_payload or {}

    if should_update_history and combined_history is not None and not combined_history.empty:
        history_record.history_json = _serialize_history_frame(combined_history)
        history_record.last_record_date = _format_datetime(combined_history.index.max())
        history_record.latest_adjusted_close = _latest_adjusted_close(combined_history)
    elif should_update_history and (combined_history is None or combined_history.empty):
        history_record.history_json = None
        history_record.last_record_date = None
        history_record.latest_adjusted_close = None

    if metadata_payload:
        metadata_record.metadata_json = json.dumps(metadata_payload)
        if metadata_payload.get("name"):
            metadata_record.name = str(metadata_payload["name"])
        metadata_record.quote_type = (metadata_payload.get("quote_type") or "") or None
        metadata_record.category = (metadata_payload.get("category") or "") or None
        metadata_record.family = (metadata_payload.get("family") or "") or None
        metadata_record.sector = (metadata_payload.get("sector") or "") or None
        metadata_record.industry = (metadata_payload.get("industry") or "") or None
        metadata_record.exchange = (metadata_payload.get("exchange") or "") or None
        metadata_record.market_cap = _coerce_float(metadata_payload.get("market_cap"))
        metadata_record.price = _coerce_float(metadata_payload.get("price"))
        metadata_record.change_percent = _coerce_float(metadata_payload.get("change_percent"))
        metadata_record.currency = (metadata_payload.get("currency") or "") or None
        metadata_record.error = (metadata_payload.get("error") or "") or None

    metadata_record.last_checked_at = _format_datetime(last_checked_at or now)
    metadata_record.updated_at = now
    history_record.last_checked_at = _format_datetime(last_checked_at or now)
    history_record.updated_at = now
    metadata_record.save()
    history_record.save()
    return _build_entry_from_model(metadata_record, history_record)


def fetch_history_with_cache(symbol, start, end):
    normalized_symbol = _normalize_symbol(symbol)
    if not normalized_symbol:
        return None, {"used_saved_cache": False, "refreshed_cache": False}

    requested_start = pd.Timestamp(start).normalize()
    requested_end = _latest_weekday(end)
    if requested_start > requested_end:
        return None, {"used_saved_cache": False, "refreshed_cache": False}
    cached_entry = load_ticker_cache_entry(normalized_symbol)

    if cached_entry is None or cached_entry.history_frame is None or cached_entry.history_frame.empty:
        full_frame = _download_full_history_frame(normalized_symbol, requested_end)
        refreshed_cache = full_frame is not None and not full_frame.empty
        if refreshed_cache:
            refresh_ticker_history(normalized_symbol, full_frame, metadata={})
        filtered = _filter_history_frame(full_frame, requested_start, requested_end)
        return filtered, {"used_saved_cache": False, "refreshed_cache": refreshed_cache}

    cached_history = cached_entry.history_frame.copy()
    cached_start = cached_history.index.min().normalize()
    cached_end = cached_history.index.max().normalize()
    missing_frames = []

    if requested_start < cached_start:
        earlier_end = (cached_start - pd.Timedelta(days=1)).normalize()
        earlier_history = _download_full_history_frame(normalized_symbol, earlier_end)
        if earlier_history is not None and not earlier_history.empty:
            missing_frames.append(earlier_history)

    if cached_end < requested_end:
        later_start = (cached_end + pd.Timedelta(days=1)).normalize()
        later_history = _download_history_frame(normalized_symbol, later_start, requested_end)
        if later_history is not None and not later_history.empty:
            missing_frames.append(later_history)

    if missing_frames:
        combined_history = _combine_history_frames([cached_history, *missing_frames])
        refresh_ticker_history(normalized_symbol, combined_history, metadata={})
        filtered = _filter_history_frame(combined_history, requested_start, requested_end)
        return filtered, {"used_saved_cache": True, "refreshed_cache": True}

    filtered = _filter_history_frame(cached_history, requested_start, requested_end)
    return filtered, {"used_saved_cache": True, "refreshed_cache": False}


def _prepare_downloaded_history(data):
    if data is None or data.empty:
        return None
    if isinstance(data, pd.Series):
        frame = data.to_frame()
    else:
        frame = data.copy()
    frame = _price_column_frame(frame)
    if frame is None or frame.empty:
        return None
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame.sort_index()


def _download_full_history_frame(symbol, end):
    requested_end = pd.Timestamp(end).normalize()
    try:
        data = yf.download(symbol, period="max", progress=False, auto_adjust=True)
    except Exception:
        return None

    frame = _prepare_downloaded_history(data)
    if frame is None:
        return None
    return frame.loc[frame.index <= requested_end].sort_index()


def _download_history_frame(symbol, start, end):
    requested_start = pd.Timestamp(start).normalize()
    requested_end = pd.Timestamp(end).normalize()
    if requested_start > requested_end:
        return None
    exclusive_end = requested_end + pd.Timedelta(days=1)
    try:
        data = yf.download(symbol, start=requested_start, end=exclusive_end, progress=False, auto_adjust=True)
    except Exception:
        return None
    return _prepare_downloaded_history(data)
