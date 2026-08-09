import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ticker_cache
from ticker_cache import (
    TickerHistoryCacheEntryModel,
    TickerMetadataCacheEntryModel,
    fetch_history_with_cache,
    initialize_ticker_cache_database,
    load_ticker_cache_entry,
    upsert_ticker_cache_entry,
)


def test_upsert_updates_history_and_latest_record_date(tmp_path):
    db_path = tmp_path / "ticker-cache.db"
    initialize_ticker_cache_database(str(db_path))

    initial_frame = pd.DataFrame(
        {"Adj Close": [10.0, 11.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    upsert_ticker_cache_entry(
        "AAPL",
        history_frame=initial_frame,
        metadata={"name": "Apple Inc."},
        last_checked_at=pd.Timestamp("2024-01-02 12:00:00"),
    )

    updated_frame = pd.concat(
        [
            initial_frame,
            pd.DataFrame(
                {"Adj Close": [12.0]},
                index=pd.to_datetime(["2024-01-03"]),
            ),
        ]
    )
    entry = upsert_ticker_cache_entry(
        "AAPL",
        history_frame=updated_frame,
        metadata={"name": "Apple Inc."},
        last_checked_at=pd.Timestamp("2024-01-03 12:00:00"),
    )

    assert entry.last_record_date == pd.Timestamp("2024-01-03").normalize()
    assert entry.latest_adjusted_close == 12.0

    loaded = load_ticker_cache_entry("AAPL")
    assert loaded is not None
    assert loaded.history_frame.index[-1] == pd.Timestamp("2024-01-03").normalize()
    assert loaded.history_frame.iloc[-1, 0] == 12.0


def test_metadata_and_history_are_cached_in_separate_tables(tmp_path):
    db_path = tmp_path / "ticker-cache.db"
    initialize_ticker_cache_database(str(db_path))

    history_frame = pd.DataFrame(
        {"Adj Close": [10.0]},
        index=pd.to_datetime(["2024-01-01"]),
    )
    upsert_ticker_cache_entry(
        "MSFT",
        history_frame=history_frame,
        metadata={"name": "Microsoft", "quote_type": "EQUITY"},
    )

    metadata_row = TickerMetadataCacheEntryModel.get_or_none(
        TickerMetadataCacheEntryModel.symbol == "MSFT"
    )
    history_row = TickerHistoryCacheEntryModel.get_or_none(
        TickerHistoryCacheEntryModel.symbol == "MSFT"
    )

    assert metadata_row is not None
    assert history_row is not None
    assert metadata_row.name == "Microsoft"
    assert history_row.history_json is not None

    upsert_ticker_cache_entry("MSFT", metadata={"price": 123.45})
    metadata_row = TickerMetadataCacheEntryModel.get(TickerMetadataCacheEntryModel.symbol == "MSFT")
    history_row = TickerHistoryCacheEntryModel.get(TickerHistoryCacheEntryModel.symbol == "MSFT")

    assert metadata_row.price == 123.45
    assert history_row.history_json is not None


def test_metadata_only_update_does_not_touch_history(tmp_path):
    db_path = tmp_path / "ticker-cache.db"
    initialize_ticker_cache_database(str(db_path))

    history_frame = pd.DataFrame(
        {"Adj Close": [11.0]},
        index=pd.to_datetime(["2024-02-01"]),
    )
    upsert_ticker_cache_entry(
        "GOOG",
        history_frame=history_frame,
        metadata={"name": "Google", "quote_type": "EQUITY"},
    )

    history_row_before = TickerHistoryCacheEntryModel.get(TickerHistoryCacheEntryModel.symbol == "GOOG")
    upsert_ticker_cache_entry("GOOG", metadata={"name": "Alphabet", "price": 123.45})
    history_row_after = TickerHistoryCacheEntryModel.get(TickerHistoryCacheEntryModel.symbol == "GOOG")

    assert history_row_after.history_json == history_row_before.history_json
    assert history_row_after.last_record_date == history_row_before.last_record_date
    assert history_row_after.latest_adjusted_close == history_row_before.latest_adjusted_close

    metadata_row = TickerMetadataCacheEntryModel.get(TickerMetadataCacheEntryModel.symbol == "GOOG")
    assert metadata_row.name == "Alphabet"
    assert metadata_row.price == 123.45


def test_fetch_history_backfills_earlier_dates_when_cache_is_fresh_at_end(tmp_path, monkeypatch):
    db_path = tmp_path / "ticker-cache.db"
    initialize_ticker_cache_database(str(db_path))

    cached_frame = pd.DataFrame(
        {"Adj Close": [13.0, 14.0]},
        index=pd.to_datetime(["2024-01-03", "2024-01-04"]),
    )
    upsert_ticker_cache_entry("AAPL", history_frame=cached_frame)

    download_calls = []

    def fake_download(symbol, end):
        download_calls.append((symbol, pd.Timestamp(end)))
        return pd.DataFrame(
            {"Adj Close": [11.0, 12.0]},
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        )

    monkeypatch.setattr(ticker_cache, "_download_full_history_frame", fake_download)

    frame, info = fetch_history_with_cache("AAPL", "2024-01-01", "2024-01-04")

    assert info == {"used_saved_cache": True, "refreshed_cache": True}
    assert download_calls == [
        ("AAPL", pd.Timestamp("2024-01-02"))
    ]
    assert frame.index.tolist() == list(pd.to_datetime([
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
    ]))
    assert frame["Adj Close"].tolist() == [11.0, 12.0, 13.0, 14.0]

    loaded = load_ticker_cache_entry("AAPL")
    assert loaded.history_frame.index[0] == pd.Timestamp("2024-01-01")
    assert loaded.history_frame.index[-1] == pd.Timestamp("2024-01-04")


def test_fetch_history_initial_download_stores_full_available_history(tmp_path, monkeypatch):
    db_path = tmp_path / "ticker-cache.db"
    initialize_ticker_cache_database(str(db_path))

    download_calls = []

    def fake_download(symbol, end):
        download_calls.append((symbol, pd.Timestamp(end)))
        return pd.DataFrame(
            {"Adj Close": [8.0, 9.0, 10.0]},
            index=pd.to_datetime(["2020-01-01", "2024-01-01", "2024-01-02"]),
        )

    monkeypatch.setattr(ticker_cache, "_download_full_history_frame", fake_download)

    frame, info = fetch_history_with_cache("AAPL", "2024-01-01", "2024-01-02")

    assert info == {"used_saved_cache": False, "refreshed_cache": True}
    assert download_calls == [
        ("AAPL", pd.Timestamp("2024-01-02"))
    ]
    assert frame["Adj Close"].tolist() == [9.0, 10.0]

    loaded = load_ticker_cache_entry("AAPL")
    assert loaded.history_frame.index[0] == pd.Timestamp("2020-01-01")
    assert loaded.history_frame["Adj Close"].tolist() == [8.0, 9.0, 10.0]


def test_fetch_history_downloads_only_missing_edges(tmp_path, monkeypatch):
    db_path = tmp_path / "ticker-cache.db"
    initialize_ticker_cache_database(str(db_path))

    cached_frame = pd.DataFrame(
        {"Adj Close": [12.0, 13.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    upsert_ticker_cache_entry("MSFT", history_frame=cached_frame)

    download_calls = []

    def fake_full_download(symbol, end):
        download_calls.append(("full", symbol, pd.Timestamp(end)))
        if pd.Timestamp(end) == pd.Timestamp("2024-01-01"):
            return pd.DataFrame(
                {"Adj Close": [11.0]},
                index=pd.to_datetime(["2024-01-01"]),
            )

    def fake_range_download(symbol, start, end):
        download_calls.append(("range", symbol, pd.Timestamp(start), pd.Timestamp(end)))
        return pd.DataFrame(
            {"Adj Close": [14.0]},
            index=pd.to_datetime(["2024-01-04"]),
        )

    monkeypatch.setattr(ticker_cache, "_download_full_history_frame", fake_full_download)
    monkeypatch.setattr(ticker_cache, "_download_history_frame", fake_range_download)

    frame, info = fetch_history_with_cache("MSFT", "2024-01-01", "2024-01-04")

    assert info == {"used_saved_cache": True, "refreshed_cache": True}
    assert download_calls == [
        ("full", "MSFT", pd.Timestamp("2024-01-01")),
        ("range", "MSFT", pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-04")),
    ]
    assert frame["Adj Close"].tolist() == [11.0, 12.0, 13.0, 14.0]


def test_fetch_history_does_not_refresh_weekend_gap(tmp_path, monkeypatch):
    db_path = tmp_path / "ticker-cache.db"
    initialize_ticker_cache_database(str(db_path))

    cached_frame = pd.DataFrame(
        {"Adj Close": [10.0, 11.0]},
        index=pd.to_datetime(["2024-01-04", "2024-01-05"]),
    )
    upsert_ticker_cache_entry("PONAX", history_frame=cached_frame)

    download_calls = []

    def fake_range_download(symbol, start, end):
        download_calls.append((symbol, start, end))
        return pd.DataFrame()

    monkeypatch.setattr(ticker_cache, "_download_history_frame", fake_range_download)

    frame, info = fetch_history_with_cache("PONAX", "2024-01-04", "2024-01-07")

    assert info == {"used_saved_cache": True, "refreshed_cache": False}
    assert download_calls == []
    assert frame.index[-1] == pd.Timestamp("2024-01-05")


def test_download_history_frame_selects_close_from_multiindex_columns(monkeypatch):
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    downloaded = pd.DataFrame(
        {
            ("Open", "AAPL"): [9.0, 10.0],
            ("Close", "AAPL"): [11.0, 12.0],
        },
        index=dates,
    )

    monkeypatch.setattr(ticker_cache.yf, "download", lambda *args, **kwargs: downloaded)

    frame = ticker_cache._download_history_frame("AAPL", "2024-01-01", "2024-01-03")

    assert frame.columns.tolist() == ["Adj Close"]
    assert frame["Adj Close"].tolist() == [11.0, 12.0]
