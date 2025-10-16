import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import precision_score
import time

# -----------------------------
# 1️⃣ Load Data
# -----------------------------
df = pd.read_csv("../data/2019-2023_stock_with_features_dif_tickers.csv")
df['date'] = pd.to_datetime(df['date'])

# -----------------------------
# 2️⃣ Engineer features & binary target
# -----------------------------
def engineer_features(df: pd.DataFrame):
    df['day_of_week'] = df['date'].dt.day_name()
    df['daily_return'] = (df['close'] - df['open']) / df['open']
    df['price_range'] = df['max'] - df['min']
    df['volume_per_quantity'] = df['volume'] / df['quantity']

    day_map = {'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5}
    df['day_of_week'] = df['day_of_week'].map(day_map)

    df.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
    df.dropna(subset=['daily_return', 'price_range', 'volume_per_quantity'], inplace=True)

    df['tomorrow'] = df.groupby('ticker')['close'].shift(-1)
    df['target'] = (df['tomorrow'] > df['close']).astype(int)

    df['rolling_close_5'] = df.groupby('ticker')['close'].transform(lambda x: x.shift(1).rolling(5).mean())
    df['rolling_std_5'] = df.groupby('ticker')['close'].transform(lambda x: x.shift(1).rolling(5).std())
    df['rolling_return_5'] = df.groupby('ticker')['daily_return'].transform(lambda x: x.shift(1).rolling(5).mean())
    df['rolling_volume_5'] = df.groupby('ticker')['volume'].transform(lambda x: x.shift(1).rolling(5).mean())
    df['momentum_5'] = df['close'] / df['rolling_close_5'] - 1

    horizons = [2, 5, 55, 220]
    new_predictors = []
    for horizon in horizons:
        ratio_col = f"Close_Ratio_{horizon}"
        trend_col = f"Trend_{horizon}"
        df[ratio_col] = df.groupby('ticker')['close'].transform(lambda x: x / x.rolling(horizon).mean())
        df[trend_col] = df.groupby('ticker')['target'].transform(lambda x: x.shift(1).rolling(horizon).sum())
        new_predictors += [ratio_col, trend_col]

    df.dropna(subset=['rolling_close_5','rolling_std_5','rolling_return_5','rolling_volume_5','momentum_5','target'] + new_predictors, inplace=True)

    features = [
        'open','close','min','max','avg','quantity','volume',
        'ibovespa_close','day_of_week','daily_return','price_range','volume_per_quantity',
        'rolling_close_5','rolling_std_5','rolling_return_5','momentum_5','rolling_volume_5'
    ] + new_predictors

    return df, features

df, features = engineer_features(df)

# -----------------------------
# 🎯 Drop low-importance features
# -----------------------------
drop_features = ['open','close','min','max','avg','daily_return','rolling_close_5','Trend_220','Close_Ratio_2']
features = [f for f in features if f not in drop_features]

# -----------------------------
# 3️⃣ Define GPU XGBoost model
# -----------------------------
model = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.6,
    colsample_bytree=0.6,
    gamma=0.3,
    reg_alpha=2,
    reg_lambda=8,
    tree_method='gpu_hist',
    random_state=1
)

# -----------------------------
# 4️⃣ Backtest helper
# -----------------------------
def backtest(df, model, features, start=1000, step=1000, threshold=0.6):
    all_preds = []

    train_precisions, test_precisions = [], []

    for i, idx in enumerate(range(start, df.shape[0], step), 1):
        train = df.iloc[:idx].copy()
        test = df.iloc[idx:idx + step].copy()

        model.fit(train[features], train['target'])
        train_preds = (model.predict_proba(train[features])[:,1] >= threshold).astype(int)
        test_preds = (model.predict_proba(test[features])[:,1] >= threshold).astype(int)

        train_prec = precision_score(train['target'], train_preds)
        test_prec = precision_score(test['target'], test_preds)

        train_precisions.append(train_prec)
        test_precisions.append(test_prec)

        preds = pd.DataFrame({'target': test['target'], 'Predictions': test_preds}, index=test.index)
        all_preds.append(preds)

    avg_train = sum(train_precisions) / len(train_precisions)
    avg_test = sum(test_precisions) / len(test_precisions)
    overfit_ratio = avg_train / avg_test
    print(f"Average Train Precision: {avg_train:.3f}")
    print(f"Average Test Precision:  {avg_test:.3f}")
    print(f"Overfitting Ratio: {overfit_ratio:.2f}")

    return pd.concat(all_preds)

# -----------------------------
# 5️⃣ Run backtest
# -----------------------------
predictions = backtest(df, model, features)

# -----------------------------
# 6️⃣ Feature importance
# -----------------------------
importances = model.get_booster().get_score(importance_type='gain')
for feature, score in sorted(importances.items(), key=lambda x: x[1], reverse=True):
    print(f"{feature:<25} {score:.2f}")

# -----------------------------
# 7️⃣ Final evaluation
# -----------------------------
print("=== Prediction Distribution ===")
print(predictions['Predictions'].value_counts())
print("\n=== Actual Distribution ===")
print(predictions['target'].value_counts(normalize=True))
print("\n=== Precision ===")
print(precision_score(predictions['target'], predictions['Predictions']))
