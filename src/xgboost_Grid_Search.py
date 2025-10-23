from itertools import product
from sklearn.metrics import precision_score
import pandas as pd
from xgboost import XGBClassifier

from xgboost_backtest import backtest

def grid_search_xgb(df, features, param_grid, start=1000, step=1000, threshold=0.6, alpha=0.7):
    """
    Grid search with a combined scoring metric:
    balanced_score = test_precision - alpha * (overfit_ratio - 1)
    """
    best_score = -1
    best_params = None
    results = []

    keys, values = zip(*param_grid.items())

    for combo in product(*values):
        params = dict(zip(keys, combo))
        print(f"\n🔍 Testing params: {params}")

        model = XGBClassifier(
            n_estimators=params.get('n_estimators', 200),
            max_depth=params.get('max_depth', 3),
            learning_rate=params.get('learning_rate', 0.05),
            subsample=params.get('subsample', 0.8),
            colsample_bytree=params.get('colsample_bytree', 0.8),
            gamma=params.get('gamma', 0.2),
            reg_alpha=params.get('reg_alpha', 1),
            reg_lambda=params.get('reg_lambda', 5),
            tree_method='hist',
            random_state=1
        )

        # Run backtest and collect precision metrics
        all_preds = []
        train_precisions, test_precisions = [], []

        for i, idx in enumerate(range(start, df.shape[0], step), 1):
            train = df.iloc[:idx].copy()
            test = df.iloc[idx:idx + step].copy()

            model.fit(train[features], train['target'])
            train_preds = (model.predict_proba(train[features])[:, 1] >= threshold).astype(int)
            test_preds = (model.predict_proba(test[features])[:, 1] >= threshold).astype(int)

            train_prec = precision_score(train['target'], train_preds)
            test_prec = precision_score(test['target'], test_preds)

            train_precisions.append(train_prec)
            test_precisions.append(test_prec)

            preds = pd.DataFrame({'target': test['target'], 'Predictions': test_preds}, index=test.index)
            all_preds.append(preds)

        avg_train = sum(train_precisions) / len(train_precisions)
        avg_test = sum(test_precisions) / len(test_precisions)
        overfit_ratio = avg_train / avg_test
        precision = precision_score(pd.concat(all_preds)['target'], pd.concat(all_preds)['Predictions'])

        # Compute balanced score
        balanced_score = avg_test - alpha * (overfit_ratio - 1)

        print(f"Average Train Precision: {avg_train:.3f}")
        print(f"Average Test Precision:  {avg_test:.3f}")
        print(f"Overfitting Ratio:       {overfit_ratio:.2f}")
        print(f"Balanced Score:          {balanced_score:.4f}")

        results.append({
            'params': params,
            'train_prec': avg_train,
            'test_prec': avg_test,
            'overfit': overfit_ratio,
            'precision': precision,
            'balanced_score': balanced_score
        })

        if balanced_score > best_score:
            best_score = balanced_score
            best_params = params

    print("\n✅ Best Params (balanced scoring):")
    print(best_params)
    print(f"Best Balanced Score: {best_score:.4f}")
    return best_params, results

best_depth = 3
best_lr = 0.5

param_grid = {
    'max_depth': [best_depth-1, best_depth, best_depth+1],  # [4, 5, 6]
    'learning_rate': [best_lr/2, best_lr, best_lr*1.5],    # [0.025, 0.05, 0.075]
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.5, 1],
    'reg_alpha': [0, 1, 2],
    'subsample': [0.6, 0.8],
    'colsample_bytree': [0.6, 0.8],

}
df = pd.read_csv("../data/2019-2023_stock_with_features_dif_tickers.csv")
features = ['quantity', 'volume', 'ibovespa_close', 'day_of_week', 'price_range', 'volume_per_quantity', 'rolling_std_5', 'rolling_return_5', 'momentum_5', 'rolling_volume_5', 'Trend_2', 'Close_Ratio_5', 'Trend_5', 'Close_Ratio_55', 'Trend_55', 'Close_Ratio_220']

best_params, results = grid_search_xgb(df, features, param_grid)
# -----------------------------
# 5️⃣ Run backtest
# -----------------------------
model = XGBClassifier(**best_params, tree_method='hist', random_state=1)
predictions = backtest(df, model, features)