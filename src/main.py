import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from models.model_runner import ModelRunner

# === Load & preprocess data ===
df = pd.read_csv("../data/2023_stock_with_features.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date')

# Replace day names with integers: Monday=0, ..., Sunday=6
df['day_of_week'] = df['date'].dt.dayofweek

# === Create target: direction (up/down/neutral) based on next day's close ===
df['next_return'] = df.groupby('ticker')['close'].shift(-1) / df['close'] - 1

threshold = 0.0  # can adjust, e.g., 0.001 for 0.1% threshold
df['direction'] = df['next_return'].apply(
    lambda x: 'up' if x > threshold else ('down' if x < -threshold else 'neutral')
)

df = df.dropna(subset=['direction'])

# === Train/test split ===
split_date = "2023-08-01"
train_df = df[df['date'] < split_date]
test_df = df[df['date'] >= split_date]

features = [
    'open', 'close', 'min', 'max', 'avg', 'quantity', 'volume',
    'ibovespa_close', 'day_of_week', 'daily_return', 'price_range', 'volume_per_quantity'
]

X_train = train_df[features]
y_train = train_df['direction']
X_test = test_df[features]
y_test = test_df['direction']

# === Scale numeric features ===
scaler = StandardScaler()
cols_to_scale = ['open', 'close', 'min', 'max', 'avg', 'quantity', 'volume',
                 'ibovespa_close', 'daily_return', 'price_range', 'volume_per_quantity']

X_train_scaled = X_train.copy()
X_train_scaled.loc[:, cols_to_scale] = scaler.fit_transform(X_train_scaled[cols_to_scale])

X_test_scaled = X_test.copy()
X_test_scaled.loc[:, cols_to_scale] = scaler.transform(X_test_scaled[cols_to_scale])


# === Define models for classification ===
models = [
    ModelRunner(LogisticRegression(max_iter=1000), name="LogisticRegression"),
    ModelRunner(make_pipeline(PolynomialFeatures(2), LogisticRegression(multi_class='multinomial', max_iter=1000)),
                name="PolynomialLogistic_deg2"),
    ModelRunner(make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, probability=True)), name="SVC_rbf"),
    ModelRunner(RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42), name="RandomForestClassifier"),
    ModelRunner(GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42),
                name="GradientBoostingClassifier"),
]

# === Train, evaluate, collect results ===
results_table = []

for model in models:
    model.fit(X_train, y_train)
    results = model.evaluate(X_test, y_test)
    results["model"] = model.name
    results_table.append(results)

# === Print results ===
results_df = pd.DataFrame(results_table).set_index("model")
print(results_df.sort_values(by="accuracy", ascending=False))

# === Optional: confusion matrices for each model ===
for model in models:
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds, labels=['up', 'down', 'neutral'])
    print(f"\nConfusion matrix for {model.name}:\n{cm}")
