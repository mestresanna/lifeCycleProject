# LifeCycleProject
# Predicting next day's close (positive / negative / neutral)

## Document Structure

The /data folder holds the base sheets for all the data used on this project.
The /notebooks folder holds different files well split and structured where the different steps for initial data manipulation, exploration, feature engineering and testing.
The /src holds two different folders:
 - models:
   - a model evaluator that not only output evaluations but also plots different graphs
   - a blueprint for every model defining their behaviour
 - main:
   - here is where the actual model training happens
 - preprocessing_pipeline:
   - here we compile all the manual code used on the notebooks creating a well organised pipeline. 

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
Only have data from January to November

## Leakage Guards

- To prevent data leakage, all features used are derived only from data available at the end of each trading day, and the target is the next day’s closing price, constructed using a one-day forward shift. No future data is used in training or feature engineering.

- In order to preserve the temporal structure of the data and prevent data leakage, the dataset was split chronologically. The model was trained on data from January to September 2023, and evaluated on unseen data from October to November 2023.


## Results and tuning 

- Using only the features: 
    'open', 'close', 'min', 'max', 'avg', 'quantity', 'volume',
    'ibovespa_close', 'day_of_week', 'daily_return', 'price_range', 'volume_per_quantity',
    'rolling_close_5', 'rolling_std_5', 'rolling_return_5', 'momentum_5', 'rolling_volume_5'

   We managed to achieve unsatisfying results:
      === Model Performance Summary ===
                              accuracy  precision    recall        f1
      model                                                           
      LogisticRegression       0.470270   0.315042  0.339877  0.315857
      RandomForest             0.464865   0.309830  0.335636  0.319629
      GradientBoosting         0.459459   0.308138  0.331851  0.315219
      SVC_rbf                  0.445946   0.294979  0.322253  0.300856
      PolynomialLogistic_deg2  0.437838   0.292787  0.315937  0.303380
      
      === Feature importances for RandomForest ===
      rolling_return_5       0.082515
      ibovespa_close         0.081356
      momentum_5             0.080674
      rolling_std_5          0.079976
      rolling_volume_5       0.072315
      daily_return           0.066063
      quantity               0.060589
      volume                 0.058483
      price_range            0.058077
      rolling_close_5        0.051682
      volume_per_quantity    0.049807
      open                   0.048894
      close                  0.047656
      max                    0.045024
      min                    0.044956
      avg                    0.041312
      day_of_week            0.030621
      dtype: float64
      
      === Feature importances for GradientBoosting ===
      rolling_return_5       0.120781
      momentum_5             0.118717
      rolling_std_5          0.106707
      ibovespa_close         0.103249
      rolling_volume_5       0.084319
      quantity               0.069908
      daily_return           0.064340
      price_range            0.057262
      volume                 0.048510
      rolling_close_5        0.046232
      close                  0.043478
      min                    0.030812
      open                   0.030097
      max                    0.025820
      day_of_week            0.021685
      volume_per_quantity    0.017094
      avg                    0.010989

   It's notable that we inserted certain rolling parameters for tuning results, which has in fact enhanced the model.
   For better results, more rolling features were created and instead of using 1100 lines (only 2023), we chose to use more data (since 2019).

   Based on the results above, we removed features with very little correlation with the target.


Using a RandomForest with backtesting and those parameters:
start=440, step=880, threshold=0.6

we managed to get those results:
=== Prediction Distribution ===
Predictions
0    16800
1     5147
Name: count, dtype: int64

=== Actual Distribution ===
target
0    0.507632
1    0.492368
Name: proportion, dtype: float64

=== Precision ===
0.6953565183602098

But then when start=220 and step=440 we get better results
=== Prediction Distribution ===
Predictions
0    16824
1     5343
Name: count, dtype: int64

=== Actual Distribution ===
target
0    0.508188
1    0.491812
Name: proportion, dtype: float64

=== Precision ===
0.7037244993449373

At first we suspected that this might've been an overfit situation since 70% is quite high.




- trying to use xgboost to improve random forest:
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    objective='binary:logistic',
    tree_method='gpu_hist',  # GPU acceleration
    predictor='gpu_predictor',
    random_state=1
)
those parameters result on this:
=== Prediction Distribution ===
Predictions
0    14647
1     7300
Name: count, dtype: int64

=== Actual Distribution ===
target
0    0.507632
1    0.492368
Name: proportion, dtype: float64

=== Precision ===
0.6363013698630137




- after trying to improve the xgboost parameters
model = XGBClassifier(
    n_estimators=800,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1,
    objective='binary:logistic',
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    random_state=1
)

those parameters result on this:
=== Prediction Distribution ===
Predictions
0    14242
1     7705
Name: count, dtype: int64

=== Actual Distribution ===
target
0    0.507632
1    0.492368
Name: proportion, dtype: float64

=== Precision ===
0.6321868916288125
