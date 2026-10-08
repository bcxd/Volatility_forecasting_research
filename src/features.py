import numpy as np


def calculate_log_returns(df):
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df

def variance_proxy(df):
    df = df.copy()
    df["variance_proxy"] = df["log_return"] ** 2
    return df
    