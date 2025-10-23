"""
This file used to use the pipeline, but due to project evolution the pipeline was changed and this file was left as history to display our first approach
This file just displays our first model training and evaluation, giving us a direction as to which model to follow
"""


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from models.model_runner import ModelRunner
from models.evaluator import ModelEvaluator

# -----------------------------
# Preprocess / Load Data
# -----------------------------
df = pd.read_csv("../data/2023_stock_with_features.csv")

df = df.sort_values(by='date')

# -----------------------------
# Train/Test Split
# -----------------------------
split_date = "2023-08-01"
train_df = df[df['date'] < split_date]
test_df = df[df['date'] >= split_date]

# -----------------------------
# Feature Selection
# -----------------------------
features = [
    'open', 'close', 'min', 'max', 'avg', 'quantity', 'volume',
    'ibovespa_close', 'day_of_week', 'daily_return', 'price_range', 'volume_per_quantity',
    'rolling_close_5', 'rolling_std_5', 'rolling_return_5', 'momentum_5', 'rolling_volume_5'
]

# Check missing features
missing = [f for f in features if f not in df.columns]
if missing:
    raise ValueError(f"Missing features in dataframe: {missing}")

# Drop NaNs caused by rolling features
df = df.dropna(subset=features + ['target'])

X_train = train_df[features]
y_train = train_df['target']
X_test = test_df[features]
y_test = test_df['target']

# -----------------------------
# Scale numeric features
# -----------------------------
scaler = StandardScaler()
cols_to_scale = [col for col in features if col != 'day_of_week']

X_train_scaled = X_train.copy()
X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train_scaled[cols_to_scale])

X_test_scaled = X_test.copy()
X_test_scaled[cols_to_scale] = scaler.transform(X_test_scaled[cols_to_scale])

# -----------------------------
# Define Models
# -----------------------------
models = [
    ModelRunner(LogisticRegression(max_iter=1000), name="LogisticRegression"),
    ModelRunner(make_pipeline(PolynomialFeatures(2), LogisticRegression(max_iter=1000)),
                name="PolynomialLogistic_deg2"),
    ModelRunner(make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, probability=True)), name="SVC_rbf"),
    ModelRunner(RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42), name="RandomForest"),
    ModelRunner(GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42),
                name="GradientBoosting"),
    ModelRunner(XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42))
]

# -----------------------------
# Initialize Evaluator
# -----------------------------
evaluator = ModelEvaluator(class_labels=[0, 1])

# -----------------------------
# Evaluate Models
# -----------------------------
for model in models:
    model.fit(X_train_scaled, y_train)
    evaluator.evaluate(model, X_test_scaled, y_test)

# -----------------------------
# Display Results
# -----------------------------
results_df = evaluator.get_results_dataframe()
print("\n=== Model Performance Summary ===")
print(results_df)

for model in models:
    fi = model.get_feature_importances()
    if fi is not None:
        print(f"\n=== Feature importances for {model.name} ===")
        print(fi)

evaluator.plot_confusion_matrices()
evaluator.plot_metric_comparison()
