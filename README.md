# LifeCycleProject

## Dataset Selection and Rationale

The **B3 Historical Quotes** dataset was selected to provide a practical foundation for exploring **financial market analysis** using real-world data from the **Brazilian stock exchange (B3)**. This dataset includes historical pricing information for a wide range of publicly traded companies and financial instruments.

Our analysis focuses on two main components:

1. **Top 5 Companies by Data Volume**
   These represent the most actively traded stocks in the dataset. By examining their historical behavior, we aim to identify patterns and market dynamics that influence their price movements.

2. **Market-Representative Funds (Ibovespa)**
   The Ibovespa index serves as a key benchmark for the Brazilian stock market. By including funds that track this index, we can evaluate how **broader market trends** affect individual stock prices.

Through this dataset, our goal is to gain analytical experience in **interpreting financial data**, understanding **market correlations**, and exploring how **macro-level indicators** (such as index performance) interact with **micro-level stock behavior**.


## Features
 - Day of the week : different days of the week may influence if a stock's price change, eg Monday & Fridays more agitated
 - daily_return = (close - open) / open  - Simple return rate
 - price_range = max - min - Intraday volatility
 - volume_per_quantity = volume / quantity - Trade size indicator

## Notes
Only have data from january to 17 November