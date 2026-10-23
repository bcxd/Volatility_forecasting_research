import pandas as pd

from src.benchmarks import persistence_var, historical_var


def test_benchmark_variances():
    df = pd.DataFrame({"variance_proxy": [1.0, 4.0, 9.0]})

    df = persistence_var(df)
    df = historical_var(df)

    assert pd.isna(df["persistence_var"].iloc[0])
    assert df["persistence_var"].iloc[1:].tolist() == [1.0, 4.0]

    assert pd.isna(df["historical_var"].iloc[0])
    assert df["historical_var"].iloc[1:].tolist() == [1.0, 2.5]
