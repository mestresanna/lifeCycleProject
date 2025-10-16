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

    We checked the performance on our training data to see if we were overfitting.

      === Model Performance Summary Train Data ===
                                 accuracy  precision    recall        f1
        model                                                           
        GradientBoosting         1.000000   1.000000  1.000000  1.000000
        RandomForest             0.974286   0.982576  0.810145  0.858919
        PolynomialLogistic_deg2  0.658571   0.771107  0.498536  0.523959
        SVC_rbf                  0.624286   0.418479  0.433043  0.422695
        LogisticRegression       0.552857   0.369615  0.382475  0.370532
    
    The Gradient Boosting model achieved perfect accuracy and F1 score on the training set, indicating potential overfitting and the need for validation on unseen data. Random Forest also performed very well, suggesting it captures meaningful structure but might require regularization to prevent overfitting. Simpler models such as Logistic Regression and SVC performed substantially worse, implying that linear decision boundaries are insufficient to capture the complexity of the data. Overall, these results highlight the importance of testing model generalization on validation or test data before drawing conclusions about predictive power.

    We create a mutual information (MI) feature selection to find the most informative features to clean the dataset once more with proper features for our models.

        === Mutual Information Scores ===
                        Feature  MutualInformation
        7        ibovespa_close           0.157941
        13        rolling_std_5           0.059836
        12      rolling_close_5           0.042649
        2                   min           0.028554
        10          price_range           0.028293
        15           momentum_5           0.023795
        8           day_of_week           0.014554
        6                volume           0.012470
        11  volume_per_quantity           0.009892
        4                   avg           0.008255
        16     rolling_volume_5           0.006924
        9          daily_return           0.003451
        0                  open           0.000000
        5              quantity           0.000000
        3                   max           0.000000
        1                 close           0.000000
        14     rolling_return_5           0.000000

    We conclude that the features with 0.0 mutual information must be removed. And based on the Feature importance, model importance, we removed features that showed very little correlation with the model and zero or very low mutual information.
    We remove; day_of_week, avg, max, min, open, close, volume_per_quantity. 
    
    We decided to keep rolling_return_5 and quantity, besides they scored 0 in mutual information, because we can see that it is useful empirically. 
   Based on the results above, we removed features with very little correlation with the target.


## RANDOM FOREST

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









### XGBOOST
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





in order to reduce overfitting we changed our start=440 and step=880 to something like:
start=2500, step=220

but doing this previous one only enhanced overfitting to 2.6


now for start=1000 and step=1000 overfitting reduced to 1.16 (very good)



After analyzing the importance distribution

Close_Ratio_5 26.57 momentum_5 24.92 Trend_2 11.20 ibovespa_close 9.48 rolling_std_5 8.36 Close_Ratio_2 8.28 Trend_5 6.82 Close_Ratio_55 6.12 day_of_week 6.04 volume_per_quantity 5.99 max 5.27 Trend_55 5.27 daily_return 5.11 rolling_close_5 5.11 Close_Ratio_220 5.08 rolling_return_5 4.97 avg 4.87 close 4.81 price_range 4.76 quantity 4.69 min 4.57 rolling_volume_5 4.45 volume 4.43 Trend_220 4.42 open 4.33

we noticed that some features should be dropped for scoring to low and possibly causing the mode to overfit.

open
close
min
max
avg
daily_return
rolling_close_5
Trend_220
Close_Ratio_2


Those features were dropped and the results were not overfitting anymore
Average Train Precision: 0.811
Average Test Precision:  0.769
Overfitting Ratio: 1.05

=== Precision ===
0.6718297872340425


now with this:
model = XGBClassifier(
    n_estimators=200,
    max_depth=3,              # shallower trees
    learning_rate=0.05,
    subsample=0.6,            # less data per tree
    colsample_bytree=0.6,     # less features per tree
    gamma=0.3,                # more conservative split penalty
    reg_alpha=2,              # L1 penalty
    reg_lambda=8,             # L2 penalty
    tree_method='gpu_hist',
    random_state=1
) 

and start=1000 and step=1000

the overfitting ratio goes down to 1.17
and precision goes up to:
=== Precision ===
0.6959747727694305
