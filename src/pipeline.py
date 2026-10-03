from src.data_fetch import data_fetch, save_raw_data
from src.data_validation import data_validation, save_processed_data

def main():
    raw_data = data_fetch(
            symbol="BTCUSDT",
            interval="1d",
            start="2019-01-01",
            end="2026-01-01",
    )
    save_raw_data(raw_data)

    processed_data = data_validation(raw_data)
    save_processed_data(processed_data)
    
    print(f"จำนวนแถว: {len(processed_data)}")
    print(processed_data.index.min(), processed_data.index.max())

if __name__ == "__main__":
    main()