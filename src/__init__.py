import pandas as pd
import os

# ===== Load dataset =====
DATA_PATH = "../data/base/2023_brazil_stocks.csv"
df = pd.read_csv(DATA_PATH)

# ===== Ensure date column is datetime (optional, useful later) =====
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])

# ===== Define tickers you want to keep =====
selected_tickers = ['PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'ABEV3']

# ===== Filter dataset =====
df_filtered = df[df['ticker'].isin(selected_tickers)]

# ===== Create output directory if it doesn't exist =====
os.makedirs("data", exist_ok=True)

# ===== Export filtered dataset =====
OUTPUT_PATH = "../data/2023_selected_stocks.csv"
df_filtered.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Filtered dataset saved to {OUTPUT_PATH}, total rows: {len(df_filtered)}")
