import numpy as np
import pandas as pd

from src.features import calculate_log_returns, variance_proxy


def test_calculate_log_returns_matches_known_values():
    df = pd.DataFrame({"close": [100.0, 110.0, 121.0]})
    result = calculate_log_returns(df)

    assert np.isnan(result.loc[0, "log_return"])

    np.testing.assert_allclose(
        result.loc[1:, "log_return"].to_numpy(),
        [np.log(1.1), np.log(1.1)],
    )

    assert "log_return" not in df.columns


def test_variance_proxy_squares_log_returns():
    df = pd.DataFrame(
        {
            "log_return": [np.nan, -0.2, 0.3],
        }
    )
    result = variance_proxy(df)

    assert np.isnan(result.loc[0, "variance_proxy"])

    np.testing.assert_allclose(
        result.loc[1:, "variance_proxy"].to_numpy(),
        [0.04, 0.09],
    )

    assert "variance_proxy" not in df.columns

