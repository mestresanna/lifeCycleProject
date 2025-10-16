#-------------------------------------------
# Pipeline for Preprocessing
#-------------------------------------------

import pandas as pd
from pathlib import Path
import os


# ---------- PIPELINE FUNCTIONS ----------

def load_data(data_path: str, ibovespa_path: str, output_path: str = "../data/2023_selected_stocks.csv", tickers: list = ['PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'ABEV3'] ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw stock and benchmark (Ibovespa) data from CSV files.
    """
    stock_df = pd.read_csv(data_path, low_memory=False)
    stock_df['date'] = pd.to_datetime(stock_df['date'], format='%Y%m%d', errors='coerce')
    selected_tickers = tickers
    df_filtered = stock_df[stock_df['ticker'].isin(selected_tickers)]
    os.makedirs("data", exist_ok=True)
    OUTPUT_PATH = output_path
    df_filtered.to_csv(OUTPUT_PATH, index=False)

    stock_df = pd.read_csv(OUTPUT_PATH, low_memory=False)
    ibov_df = pd.read_csv(ibovespa_path, low_memory=False)

    return stock_df, ibov_df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare raw stock data by removing unnecessary columns
    and fixing data types.
    """
    cols_to_drop = [
        'currency', 'name', 'marketType', 'bdiCode', 'prazoT', 'paperSpecification',
        'optionPrice', 'priceCorrection', 'paperDueDate', 'quoteFactor'
    ]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    df['date'] = pd.to_datetime(df['date'])
    return df


def merge_benchmark(stock_df: pd.DataFrame, ibov_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Ibovespa index closing prices with stock data based on date.
    """
    ibov_df = ibov_df.rename(columns={'Date': 'date', 'close': 'ibovespa_close'})
    ibov_df['date'] = pd.to_datetime(ibov_df['date'])
    merged = stock_df.merge(ibov_df[['date', 'ibovespa_close']], on='date', how='left')
    return merged


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df['day_of_week'] = df['date'].dt.day_name()
    df['daily_return'] = (df['close'] - df['open']) / df['open']
    df['price_range'] = df['max'] - df['min']
    df['volume_per_quantity'] = df['volume'] / df['quantity']

    # numeric day
    day_map = {'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}
    df['day_of_week'] = df['day_of_week'].map(day_map)

    df.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
    df = df.infer_objects(copy=False)
    df.dropna(subset=['daily_return', 'price_range', 'volume_per_quantity'], inplace=True)

    df['target'] = df.groupby('ticker')['close'].shift(-1)

    # rolling features
    grouped = df.groupby('ticker')
    df['rolling_close_5'] = grouped['close'].shift(1).rolling(5).mean()
    df['rolling_std_5'] = grouped['close'].shift(1).rolling(5).std()
    df['rolling_return_5'] = grouped['daily_return'].shift(1).rolling(5).mean()
    df['momentum_5'] = df['close'] / df['rolling_close_5'] - 1
    df['rolling_volume_5'] = grouped['volume'].shift(1).rolling(5).mean()

    df = df.sort_values('date')
    return df


def save_data(df: pd.DataFrame, output_path: str):
    """
    Save processed dataset to a CSV file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data saved successfully to {output_path}")


# ---------- MAIN EXECUTION PIPELINE ----------
class DataPreprocessor():
    def __init__(self, **kwargs):
        self.ibovespa_path = kwargs.get("ibovespa_path", "../data/ibovespa_2023.csv")
        self.data_path = kwargs.get("data_path", "../data/base/2023_brazil_stocks.csv")
        self.output_path = kwargs.get("output_path", "../data/2023_selected_stocks.csv")
        self.tickers = kwargs.get("tickers", ['PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'ABEV3'])

    def run_pipeline(self):
        """
        Execute the full ETL (Extract–Transform–Load) pipeline:
        1. Load data
        2. Clean stock dataset
        3. Merge Ibovespa benchmark
        4. Engineer new features
        5. Save final dataset
        """
        output_path = "../data/2023_stock_with_features_dif_tickers.csv"

        # Step 1: Load data
        ds, ibov = load_data(self.data_path, self.ibovespa_path, self.output_path, self.tickers)

        # Step 2: Clean
        ds = clean_data(ds)

        # Step 3: Merge benchmark
        ds = merge_benchmark(ds, ibov)

        # Step 4: Feature engineering
        ds = engineer_features(ds)

        # Step 5: Save
        save_data(ds, output_path)
        return ds