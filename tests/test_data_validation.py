import pandas as pd
import pytest

from src.data_validation import (
    data_validation,
    save_processed_data,
    validate_date_range,
)


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


def test_save_processed_data_writes_expected_csv(tmp_path):
    dates = pd.to_datetime(
        ["2024-01-01", "2024-01-02"],
        utc=True,
    ).rename("open_time")
    df = pd.DataFrame(
        {"close": [100.0, 105.0]},
        index=dates,
    )

    saved_path = save_processed_data(
        df=df,
        symbol="BTCUSDT",
        interval="1d",
        start="2019-01-01",
        end="2026-01-01",
        output_dir=tmp_path,
    )

    expected_path = tmp_path / "btcusdt_1d_2019-01-01_2026-01-01.csv"

    assert saved_path == expected_path
    assert saved_path.exists()

    saved_df = pd.read_csv(
        saved_path,
        index_col="open_time",
        parse_dates=["open_time"],
    )

    pd.testing.assert_frame_equal(
        saved_df,
        df,
        check_freq=False,
    )
