# momentum-predictor — Balanced Ensemble Model for Bybit Futures

This project implements a **momentum trading predictor** for Bybit Futures, focused on the **1h timeframe** (adaptable).  
It includes automated OHLCV data fetching, feature engineering, and an ensemble of ML models optimized for class balance.

## ✨ Features
- Fetches up to **4 years of historical OHLCV data** (adaptative) from Bybit (bypasses API limits by iterative requests).
- Works with **CSV or live API**.
- Rich **feature engineering**: RSI, ATR, ADX, MACD, Bollinger, volume ratios, momentum.
- Uses a **balanced ensemble** of:
  - Random Forest
  - Extra Trees
  - Gradient Boosting
- Includes **walk-forward validation** with `TimeSeriesSplit`.
- Saves trained models, selected features, and configs to `.pkl` files.
- Adjustable threshold accounting for **trading fees**.

## Examples Output, Results and Saved Models

Trains multiple models with walk-forward validation.

Prints fold accuracies, AUC, optimal thresholds.

Saves models and configs to /models.

📊 FINAL RESULTS FOR BTC/USDT:
  Average Fold Adjusted Accuracy: 0.5623
  Average Fold AUC: 0.6041
  Optimal Threshold: 0.51

💾 FILES SAVED:
  • models/BTCUSDT_random_forest_model.pkl
  • models/BTCUSDT_features.pkl
  • models/BTCUSDT_config.pkl
