
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

from models.base_model import WrappedModel

# === Load & preprocess data ===
df = pd.read_csv("../data/2023_stock_with_features.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date')

# Replace day names with integers: Monday=0, ..., Sunday=6
df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek


# === Create target ===
df['target'] = df.groupby('ticker')['close'].shift(-1)
df = df.dropna()

# === Train/test split ===
split_date = "2023-08-01"
train_df = df[df['date'] < split_date]
test_df = df[df['date'] >= split_date]


features = [
    'open', 'close', 'min', 'max', 'avg', 'quantity', 'volume',
    'ibovespa_close', 'day_of_week', 'daily_return', 'price_range', 'volume_per_quantity'
]

X_train = train_df[features]
y_train = train_df['target']
X_test = test_df[features]
y_test = test_df['target']

# === Wrap your models ===
models = [
    WrappedModel(LinearRegression(), name="LinearRegression"),
    WrappedModel(make_pipeline(PolynomialFeatures(3), LinearRegression()), name="PolynomialRegression_deg3"),
    WrappedModel(make_pipeline(StandardScaler(), SVR(kernel='rbf', C=1.0, epsilon=0.1)), name="SVR_rbf")
]

# === Train, evaluate, print results ===
for model in models:
    model.fit(X_train, y_train)
    results = model.evaluate(X_test, y_test)
    print(f"{model.name} results:")
    print(f"  RMSE: {results['rmse']:.4f}")
    print(f"  MAE : {results['mae']:.4f}")
    print()
