This repository contains a Python script for generating Renko bars, calculating various technical indicators (ATR, Hurst Exponent), and preparing a dataset for training an Artificial Intelligence (AI) model for forex trading. The script connects to MetaTrader 5 (MT5) to fetch historical data and includes functionality to train and evaluate a RandomForestClassifier for predicting potential long and short trade opportunities.

Features
MetaTrader 5 Integration: Connects to MT5 to download historical market data for a specified symbol and timeframe.

Heiken Ashi Calculation: Computes Heiken Ashi candlestick data and determines the Heiken Ashi trend direction.

ATR (Average True Range): Calculates the ATR, used for dynamic brick sizing in Renko charts.

Hurst Exponent: Measures the long-term memory of a time series, indicating whether the price series is trending or mean-reverting.

ATR-based Renko Bar Generation: Generates Renko bars where the brick size is dynamically determined by a multiple of the ATR, offering a more adaptive approach than fixed brick sizes.

Multi-Timeframe Analysis: Incorporates data and derived features (Heiken Ashi direction, Renko direction) from additional, higher timeframes to enrich the dataset.

AI Target Variable Creation: Defines target variables (target_long, target_short) based on future price movements relative to a calculated profit/loss threshold, suitable for supervised learning.

Random Forest Classifier: Trains and evaluates two RandomForestClassifier models: one for predicting profitable long trades and another for profitable short trades.

Feature Engineering: Creates a comprehensive set of features from raw price data, Heiken Ashi, Renko bars, ATR, Hurst Exponent, and multi-timeframe insights.

Data Persistence: Saves the prepared dataset to a CSV file for further analysis or direct use in AI model training. It also "locks" the initial ATR value for consistent Renko brick sizing across runs.

Output

Training data shape: (9895, 19)
Long target distribution:
target_long
1    0.699444
0    0.300556
Name: proportion, dtype: float64
Short target distribution:
target_short
1    0.669631
0    0.330369
Name: proportion, dtype: float64

--- Training Model for Long Trades ---
Long Trade Model Performance (Test Set):
              precision    recall  f1-score   support

           0       0.53      0.33      0.41       595
           1       0.75      0.87      0.81      1384

    accuracy                           0.71      1979
   macro avg       0.64      0.60      0.61      1979
weighted avg       0.68      0.71      0.69      1979

Accuracy: 0.7094492167761496
Confusion Matrix:
 [[ 199  396]
 [ 179 1205]]

Top 10 Feature Importances (Long Model):
 hurst                          0.328405
atr                            0.325744
renko_direction                0.032709
ha_direction                   0.030302
ha_direction_M30_shifted       0.026824
renko_direction_H4_shifted     0.026239
ha_direction_H1_shifted        0.025938
renko_direction_D1_shifted     0.023002
renko_direction_M30_shifted    0.022912
ha_direction_D1_shifted        0.022535
dtype: float64

--- Training Model for Short Trades ---
Short Trade Model Performance (Test Set):
              precision    recall  f1-score   support

           0       0.57      0.42      0.48       654
           1       0.75      0.85      0.79      1325

    accuracy                           0.70      1979
   macro avg       0.66      0.63      0.64      1979
weighted avg       0.69      0.70      0.69      1979

Accuracy: 0.7033855482566953
Confusion Matrix:
 [[ 272  382]
 [ 205 1120]]

Top 10 Feature Importances (Short Model):
 hurst                          0.334983
atr                            0.330246
renko_direction                0.033605
ha_direction                   0.027603
ha_direction_H1_shifted        0.025893
ha_direction_M30_shifted       0.025220
renko_direction_H4_shifted     0.024276
renko_direction_M30_shifted    0.022811
renko_direction_D1_shifted     0.022595
renko_direction_H1_shifted     0.021995
dtype: float64
