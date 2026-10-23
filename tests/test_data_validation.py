import pandas as pd
import pytest

from src.data_validation import data_validation, validate_date_range


def make_raw_candle(date):
    open_time = pd.Timestamp(date, tz="UTC")
    close_time = (
        open_time
        + pd.Timedelta(days=1)
        - pd.Timedelta(milliseconds=1)
    )

    return [
        int(open_time.timestamp() * 1000),
        "100.0",
        "110.0",
        "90.0",
        "105.0",
        "10.0",
        int(close_time.timestamp() * 1000),
        "1000.0",
        50,
        "5.0",
        "500.0",
        "0",
    ]


def test_data_validation_accepts_valid_daily_candles():
    raw_data = [
        make_raw_candle("2024-01-01"),
        make_raw_candle("2024-01-02"),
    ]

    result = data_validation(
        raw_data,
        start="2024-01-01",
        end="2024-01-03",
    )

    expected_index = pd.to_datetime(
        ["2024-01-01", "2024-01-02"],
        utc=True,
    )

    assert result.index.tolist() == expected_index.tolist()
    assert len(result) == 2


def test_validate_date_range_accepts_complete_daily_index():
    dates = pd.date_range(
        start="2024-01-01",
        end="2024-01-03",
        freq="D",
        tz="UTC",
        inclusive="left",
    )

    df = pd.DataFrame(index=dates)

    validate_date_range(
        df,
        start="2024-01-01",
        end="2024-01-03",
    )


def test_validate_date_range_rejects_missing_date():
    dates = pd.to_datetime(
        ["2024-01-01"],
        utc=True,
    )
    df = pd.DataFrame(index=dates)

    with pytest.raises(ValueError, match="Found missing dates"):
        validate_date_range(
            df,
            start="2024-01-01",
            end="2024-01-03",
        )


def test_validate_date_range_rejects_unexpected_date():
    dates = pd.to_datetime(
        [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
        ],
        utc=True,
    )
    df = pd.DataFrame(index=dates)

    with pytest.raises(ValueError, match="Found unexpected dates"):
        validate_date_range(
            df,
            start="2024-01-01",
            end="2024-01-03",
        )
