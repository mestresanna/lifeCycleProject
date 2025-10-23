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

features = ['quantity', 'volume', 'ibovespa_close', 'day_of_week', 'price_range', 'volume_per_quantity',
            'rolling_std_5', 'rolling_return_5', 'momentum_5', 'rolling_volume_5', 'Trend_2',
            'Close_Ratio_5', 'Trend_5', 'Close_Ratio_55', 'Trend_55', 'Close_Ratio_220']

# -----------------------------
# 3️⃣ Define model
# -----------------------------
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, n_jobs=-1, random_state=1)

# -----------------------------
# 4️⃣ Helper functions
# -----------------------------
def evaluate_preds(y_true, y_pred, label="Evaluation"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    print(f"\n=== {label} Metrics ===")
    print("Accuracy:", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall:", round(rec, 4))
    return acc, prec, rec

def predict(train, test, predictors, model, threshold=0.6):
    model.fit(train[predictors], train['target'])
    probs = model.predict_proba(test[predictors])[:, 1]
    preds = (probs >= threshold).astype(int)
    return pd.DataFrame({'target': test['target'], 'Predictions': preds}, index=test.index)

def backtest_stepwise(df, model, features, start=2000, step=1500, threshold=0.6):
    all_preds = []
    overfit_ratios = []
    total_steps = (df.shape[0] - start) // step + 1
    start_time = time.time()

    for i, idx in enumerate(range(start, df.shape[0], step), 1):
        train = df.iloc[:idx].copy()
        test = df.iloc[idx:idx + step].copy()

        # Predict on test
        preds = predict(train, test, features, model, threshold)
        all_preds.append(preds)

        # Evaluate training performance
        train_preds = model.predict(train[features])
        train_acc, _, _ = evaluate_preds(train['target'], train_preds, label=f"Train Step {i}")

        # Evaluate test performance
        test_acc, _, _ = evaluate_preds(test['target'], preds['Predictions'], label=f"Test Step {i}")

        # --- Overfit Ratio ---
        if test_acc > 0:
            overfit_ratio = train_acc / test_acc
            overfit_ratios.append(overfit_ratio)
            print(f"📊 Overfit Ratio (Train/Test): {overfit_ratio:.3f}")
        else:
            print("⚠️ Test accuracy is 0, skipping overfit ratio.")
            overfit_ratios.append(float('nan'))

        # --- Timing & ETA ---
        elapsed = time.time() - start_time
        avg_per_step = elapsed / i
        remaining_steps = total_steps - i
        eta = remaining_steps * avg_per_step
        print(f"Step {i}/{total_steps} done. Elapsed: {elapsed:.1f}s, ETA: {eta / 60:.1f} min\n")

    # --- Final average overfit ratio ---
    valid_ratios = [r for r in overfit_ratios if not pd.isna(r)]
    if valid_ratios:
        avg_ratio = sum(valid_ratios) / len(valid_ratios)
        print(f"\n🏁 Final Average Overfit Ratio across steps: {avg_ratio:.3f}")
    else:
        print("\n⚠️ No valid overfit ratios calculated.")

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