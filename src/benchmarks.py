def persistence_var(df):
    df = df.copy()
    df["persistence_var"] = df["variance_proxy"].shift(1)
    return df


def historical_var(df):
    df = df.copy()
    df["historical_var"] = df["variance_proxy"].expanding().mean().shift(1)
    return df
