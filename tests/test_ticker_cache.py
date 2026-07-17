import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ticker_cache import (
    TickerHistoryCacheEntryModel,
    TickerMetadataCacheEntryModel,
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
