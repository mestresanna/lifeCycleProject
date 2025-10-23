import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
import time

# -----------------------------
# 1️⃣ Load Data
# -----------------------------
df = pd.read_csv("../data/2019-2023_stock_with_features_dif_tickers.csv")
df['date'] = pd.to_datetime(df['date'])

# -----------------------------
# 2️⃣ Engineer features & binary target
# -----------------------------
# def engineer_features(df: pd.DataFrame):
#     df['day_of_week'] = df['date'].dt.day_name()
#     df['daily_return'] = (df['close'] - df['open']) / df['open']
#     df['price_range'] = df['max'] - df['min']
#     df['volume_per_quantity'] = df['volume'] / df['quantity']
#
#     day_map = {'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5}
#     df['day_of_week'] = df['day_of_week'].map(day_map)
#
#     df.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
#     df.dropna(subset=['daily_return', 'price_range', 'volume_per_quantity'], inplace=True)
#
#     df['tomorrow'] = df.groupby('ticker')['close'].shift(-1)
#     df['target'] = (df['tomorrow'] > df['close']).astype(int)
#
#     df['rolling_close_5'] = df.groupby('ticker')['close'].transform(lambda x: x.shift(1).rolling(5).mean())
#     df['rolling_std_5'] = df.groupby('ticker')['close'].transform(lambda x: x.shift(1).rolling(5).std())
#     df['rolling_return_5'] = df.groupby('ticker')['daily_return'].transform(lambda x: x.shift(1).rolling(5).mean())
#     df['rolling_volume_5'] = df.groupby('ticker')['volume'].transform(lambda x: x.shift(1).rolling(5).mean())
#     df['momentum_5'] = df['close'] / df['rolling_close_5'] - 1
#
#     horizons = [2, 5, 55, 220]
#     new_predictors = []
#     for horizon in horizons:
#         ratio_col = f"Close_Ratio_{horizon}"
#         trend_col = f"Trend_{horizon}"
#         df[ratio_col] = df.groupby('ticker')['close'].transform(lambda x: x / x.rolling(horizon).mean())
#         df[trend_col] = df.groupby('ticker')['target'].transform(lambda x: x.shift(1).rolling(horizon).sum())
#         new_predictors += [ratio_col, trend_col]
#
#     df.dropna(subset=['rolling_close_5','rolling_std_5','rolling_return_5','rolling_volume_5','momentum_5','target'] + new_predictors, inplace=True)
#
#     features = [
#         'open','close','min','max','avg','quantity','volume',
#         'ibovespa_close','day_of_week','daily_return','price_range','volume_per_quantity',
#         'rolling_close_5','rolling_std_5','rolling_return_5','momentum_5','rolling_volume_5'
#     ] + new_predictors
#
#     return df, features
#
# df, features = engineer_features(df)
#
# # -----------------------------
# # 🎯 Drop low-importance features
# # -----------------------------
# drop_features = ['open','close','min','max','avg','daily_return','rolling_close_5','Trend_220','Close_Ratio_2']
# features = [f for f in features if f not in drop_features]
features = ['quantity', 'volume', 'ibovespa_close', 'day_of_week', 'price_range', 'volume_per_quantity', 'rolling_std_5', 'rolling_return_5', 'momentum_5', 'rolling_volume_5', 'Trend_2', 'Close_Ratio_5', 'Trend_5', 'Close_Ratio_55', 'Trend_55', 'Close_Ratio_220']


# -----------------------------
# 3️⃣ Define GPU XGBoost model
# -----------------------------
# model = XGBClassifier(
#     n_estimators=200,
#     max_depth=3,
#     learning_rate=0.05,
#     subsample=0.6,
#     colsample_bytree=0.6,
#     gamma=0.3,
#     reg_alpha=2,
#     reg_lambda=8,
#     tree_method='hist',
#     random_state=1
# )

# best grid params model testing
model = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.6,
    colsample_bytree=0.8,
    gamma=0.2,
    reg_alpha=2,
    reg_lambda=5,
    tree_method='hist',
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

    preds_df = pd.concat(all_preds)
    preds_df.attrs['train_precision'] = avg_train
    preds_df.attrs['test_precision'] = avg_test
    return preds_df

# time-series cross-validation technique,ensures the model’s strong precision isn’t just due to one lucky time split.
def walk_forward_validation(df, features, model_params, start_values, step_values, threshold=0.6):
    results = []

    for start in start_values:
        for step in step_values:
            print(f"\n🚶 Walk-forward with start={start}, step={step}")
            model = XGBClassifier(**model_params, tree_method='hist', random_state=1)

            preds = backtest(df, model, features, start=start, step=step, threshold=threshold)

            train_prec = preds.attrs.get('train_precision', None)
            test_prec = preds.attrs.get('test_precision', None)

            overall_prec = precision_score(preds['target'], preds['Predictions'])
            overfit_ratio = train_prec / test_prec if test_prec else None

            print(f"  ▶️ Train Precision: {train_prec:.3f}, Test Precision: {test_prec:.3f}, "
                  f"Overall Precision: {overall_prec:.3f}, Overfit Ratio: {overfit_ratio:.2f}")

            results.append({
                'start': start,
                'step': step,
                'train_precision': train_prec,
                'test_precision': test_prec,
                'overall_precision': overall_prec,
                'overfit_ratio': overfit_ratio
            })

    results_df = pd.DataFrame(results)
    print("\n=== Walk-Forward Summary ===")
    print(results_df.sort_values(by='overall_precision', ascending=False))


    return results_df

# used for walk forward validation
# best_params = {
#     'n_estimators': 200,
#     'max_depth': 3,
#     'learning_rate': 0.05,
#     'subsample': 0.6,
#     'colsample_bytree': 0.6,
#     'gamma': 0.3,
#     'reg_alpha': 2,
#     'reg_lambda': 8
# }
#
# start_values = [500, 1000, 2000]
# step_values = [500, 1000, 2000]
#
# wf_results = walk_forward_validation(df, features, best_params, start_values, step_values)

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

recall = recall_score(predictions['target'], predictions['Predictions'])
print(f"Recall: {recall:.4f}")
accuracy = accuracy_score(predictions['target'], predictions['Predictions'])
print(f"Accuracy: {accuracy:.4f}")
