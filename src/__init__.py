import pandas as pd
import os

# ===== Load dataset =====
DATA_PATH = "../data/base/2023_brazil_stocks.csv"
df = pd.read_csv(DATA_PATH)

# ===== Convert 'date' column to datetime =====
# If it's an integer like 20230102, first convert to string
df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')

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
