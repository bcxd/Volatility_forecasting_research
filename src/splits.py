import pandas as pd


def split_data(df, train_end, validation_end, test_end):
    train_end = pd.Timestamp(train_end, tz="UTC")
    validation_end = pd.Timestamp(validation_end, tz="UTC")
    test_end = pd.Timestamp(test_end, tz="UTC")

    if not (train_end < validation_end < test_end):
        raise ValueError(
            "Dates must satisfy train_end < validation_end < test_end"
        )

    train = df.loc[df.index < train_end].copy()

    validation = df.loc[
        (df.index >= train_end)
        & (df.index < validation_end)
    ].copy()

    test = df.loc[
        (df.index >= validation_end)
        & (df.index < test_end)
    ].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError("Train, validation, and test must not be empty")

    return train, validation, test
