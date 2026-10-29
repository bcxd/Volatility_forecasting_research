import argparse
from pathlib import Path

from src.data_fetch import BINANCE_KLINES_URL, data_fetch
from src.data_io import save_raw_data


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Download and save an immutable Binance snapshot."
        )
    )
    parser.add_argument(
        "--symbol",
        default="BTCUSDT",
    )
    parser.add_argument(
        "--interval",
        default="1d",
    )
    parser.add_argument(
        "--start",
        default="2019-01-01",
    )
    parser.add_argument(
        "--end",
        default="2026-01-01",
        help="Exclusive end date.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
    )

    return parser


def acquire_snapshot(
    symbol,
    interval,
    start,
    end,
    output_dir=Path("data/raw"),
):
    raw_data = data_fetch(
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
    )

    raw_path = save_raw_data(
        raw_data=raw_data,
        endpoint=BINANCE_KLINES_URL,
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
        output_dir=output_dir,
    )

    return raw_path


def main(argv=None):
    args = build_parser().parse_args(argv)

    raw_path = acquire_snapshot(
        symbol=args.symbol,
        interval=args.interval,
        start=args.start,
        end=args.end,
        output_dir=args.output_dir,
    )

    print(f"Raw snapshot saved to: {raw_path}")

    return raw_path


if __name__ == "__main__":
    main()