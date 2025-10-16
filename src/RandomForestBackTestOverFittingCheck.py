# stock_backtest_stepwise.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score, recall_score
import time
import matplotlib.pyplot as plt

# -----------------------------
# 1️⃣ Load Data
# -----------------------------
df = pd.read_csv("../data/2019-2023_stock_with_features_dif_tickers.csv")
df['date'] = pd.to_datetime(df['date'])

# -----------------------------
# 2️⃣ Engineer features & binary target
# -----------------------------
def engineer_features(df: pd.DataFrame):
    # Basic derived features
    df['day_of_week'] = df['date'].dt.day_name()
    df['daily_return'] = (df['close'] - df['open']) / df['open']
    df['price_range'] = df['max'] - df['min']
    df['volume_per_quantity'] = df['volume'] / df['quantity']

    day_map = {'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5}
    df['day_of_week'] = df['day_of_week'].map(day_map)

    df.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
    df.dropna(subset=['daily_return', 'price_range', 'volume_per_quantity'], inplace=True)

    # Tomorrow and binary target
    df['tomorrow'] = df.groupby('ticker')['close'].shift(-1)
    df['target'] = (df['tomorrow'] > df['close']).astype(int)

    # Rolling features
    df['rolling_close_5'] = df.groupby('ticker')['close'].transform(lambda x: x.shift(1).rolling(5).mean())
    df['rolling_std_5'] = df.groupby('ticker')['close'].transform(lambda x: x.shift(1).rolling(5).std())
    df['rolling_return_5'] = df.groupby('ticker')['daily_return'].transform(lambda x: x.shift(1).rolling(5).mean())
    df['rolling_volume_5'] = df.groupby('ticker')['volume'].transform(lambda x: x.shift(1).rolling(5).mean())
    df['momentum_5'] = df['close'] / df['rolling_close_5'] - 1

    # Multi-horizon features
    horizons = [2, 5, 55, 220]
    new_predictors = []
    for horizon in horizons:
        ratio_col = f"Close_Ratio_{horizon}"
        trend_col = f"Trend_{horizon}"
        df[ratio_col] = df.groupby('ticker')['close'].transform(lambda x: x / x.rolling(horizon).mean())
        df[trend_col] = df.groupby('ticker')['target'].transform(lambda x: x.shift(1).rolling(horizon).sum())
        new_predictors += [ratio_col, trend_col]

    # Drop NaNs
    df.dropna(subset=[
        'rolling_close_5','rolling_std_5','rolling_return_5',
        'rolling_volume_5','momentum_5','target'
    ] + new_predictors, inplace=True)

    # Final feature list
    features = [
        'open','close','min','max','avg','quantity','volume',
        'ibovespa_close','day_of_week','daily_return','price_range','volume_per_quantity',
        'rolling_close_5','rolling_std_5','rolling_return_5','momentum_5','rolling_volume_5'
    ] + new_predictors

    return df, features

df, features = engineer_features(df)

# -----------------------------
# 3️⃣ Define model
# -----------------------------
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, n_jobs=-1, random_state=1)

# -----------------------------
# 4️⃣ Helper functions
# -----------------------------
def evaluate_preds(y_true, y_pred, label="Evaluation"):
    print(f"\n=== {label} Metrics ===")
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("Precision:", round(precision_score(y_true, y_pred), 4))
    print("Recall:", round(recall_score(y_true, y_pred), 4))

def predict(train, test, predictors, model, threshold=0.6):
    model.fit(train[predictors], train['target'])
    probs = model.predict_proba(test[predictors])[:, 1]
    preds = (probs >= threshold).astype(int)
    return pd.DataFrame({'target': test['target'], 'Predictions': preds}, index=test.index)

def backtest_stepwise(df, model, features, start=2000, step=1500, threshold=0.6):
    all_preds = []
    total_steps = (df.shape[0] - start) // step + 1
    start_time = time.time()

    for i, idx in enumerate(range(start, df.shape[0], step), 1):
        train = df.iloc[:idx].copy()
        test = df.iloc[idx:idx + step].copy()

        # Predict on test
        preds = predict(train, test, features, model, threshold)
        all_preds.append(preds)

        # Evaluate training performance up to current step
        train_preds = model.predict(train[features])
        evaluate_preds(train['target'], train_preds, label=f"Train Step {i}")

        # Evaluate test slice
        evaluate_preds(test['target'], preds['Predictions'], label=f"Test Step {i}")

        # --- Timing & ETA ---
        elapsed = time.time() - start_time
        avg_per_step = elapsed / i
        remaining_steps = total_steps - i
        eta = remaining_steps * avg_per_step
        print(f"Step {i}/{total_steps} done. Elapsed: {elapsed:.1f}s, ETA: {eta / 60:.1f} min\n")

    return pd.concat(all_preds)

# -----------------------------
# 5️⃣ Run backtest with stepwise evaluation
# -----------------------------
predictions = backtest_stepwise(df, model, features)

# -----------------------------
# 6️⃣ Evaluate overall backtest
# -----------------------------
evaluate_preds(predictions['target'], predictions['Predictions'], label="Overall Backtest")

# -----------------------------
# 7️⃣ Distribution & Save
# -----------------------------
print("\n=== Prediction Distribution ===")
print(predictions['Predictions'].value_counts())
print("\n=== Actual Distribution ===")
print(predictions['target'].value_counts(normalize=True))

predictions.to_csv("../data/backtest_stepwise_predictions.csv", index=False)
print("\n✅ Stepwise backtest predictions saved to '../data/backtest_stepwise_predictions.csv'")
