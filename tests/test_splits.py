import pandas as pd

from src.splits import split_data


def test_split_data_uses_exclusive_end_dates():
    dates = pd.to_datetime(
        [
            "2023-12-30",
            "2023-12-31",
            "2024-01-01",
            "2024-12-31",
            "2025-01-01",
            "2025-12-31",
        ],
        utc=True,
    )

    df = pd.DataFrame(
        {"value": range(len(dates))},
        index=dates,
    )

    train, validation, test = split_data(
        df=df,
        train_end="2024-01-01",
        validation_end="2025-01-01",
        test_end="2026-01-01",
    )

    assert train.index.tolist() == dates[:2].tolist()
    assert validation.index.tolist() == dates[2:4].tolist()
    assert test.index.tolist() == dates[4:].tolist()
