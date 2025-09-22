import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
import time
from typing import Tuple, List, Dict, Any
import warnings
import ccxt
import argparse
from pathlib import Path
warnings.filterwarnings('ignore')

# -------------------- CONFIGURATION --------------------
api_key = os.getenv('BYBIT_API_KEY')
api_secret = os.getenv('BYBIT_API_SECRET')


parser = argparse.ArgumentParser()
parser.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT"])
parser.add_argument("--timeframe", default="1h")
args = parser.parse_args()

symbols = args.symbols
timeframe_df = args.timeframe
limit_bars = int(os.getenv("BYBIT_LIMIT_BARS", 4 * 365 * 24))  # 4 years by default

USE_CSV = False
CSV_FILENAMES = {s: f"{s.replace('/','')}_{timeframe_df}_data.csv" for s in symbols}

TRADING_FEES = 0.0012  # 0.12% total fees (entry + exit)

# Initialize exchange
exchange = ccxt.bybit({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
    'timeout': 60000,
    'options': {
        'defaultType': 'future',
        'adjustForTimeDifference': True,
        'recvWindow': 20000,
    },
})

# -------------------- TECHNICAL INDICATORS --------------------
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

def calculate_adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr = calculate_atr(high, low, close, period=1)
    plus_di = 100 * (plus_dm.rolling(period).mean() / tr.rolling(period).mean())
    minus_di = 100 * (minus_dm.rolling(period).mean() / tr.rolling(period).mean())
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    return adx

# -------------------- DATA LOADING --------------------
def fetch_ohlcv_from_csv(filename: str) -> pd.DataFrame:
    print(f"Loading data from CSV file: {filename}")
    try:
        df = pd.read_csv(filename)
        column_mappings = {
            'time': 'timestamp', 'date': 'timestamp', 'datetime': 'timestamp',
            'Time': 'timestamp', 'Date': 'timestamp', 'open_time': 'timestamp',
            'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume',
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
        }
        df.rename(columns=column_mappings, inplace=True)
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        if df['timestamp'].dtype == 'int64':
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df.set_index('timestamp', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep='last')]
        df = df[(df > 0).all(axis=1)]
        print(f"Loaded {len(df)} rows from CSV")
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        raise

def fetch_ohlcv(symbol: str) -> pd.DataFrame:
    all_ohlcv = []
    max_limit = 1000
    total_requests = limit_bars // max_limit + (1 if limit_bars % max_limit else 0)
    print(f"Fetching {limit_bars} bars for {symbol} in {total_requests} requests...")
    timeframe_ms = 60 * 60 * 1000  # 1 hour in milliseconds
    since = None
    for i in range(total_requests):
        try:
            if since is None:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe_df, limit=max_limit)
            else:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe_df, since=since, limit=max_limit)
            if not ohlcv:
                break
            all_ohlcv = ohlcv + all_ohlcv
            since = ohlcv[0][0] - (max_limit * timeframe_ms)
            print(f"Progress: {min((i + 1) * max_limit, limit_bars)}/{limit_bars} bars fetched")
            time.sleep(exchange.rateLimit / 1000)
            if len(all_ohlcv) >= limit_bars:
                all_ohlcv = all_ohlcv[-limit_bars:]
                break
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
    df = pd.DataFrame(all_ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[~df.index.duplicated(keep='last')]
    df.sort_index(inplace=True)
    df = df[(df > 0).all(axis=1)]
    return df

# -------------------- FEATURE ENGINEERING --------------------
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Creating features...")
    # Core price features
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['HL_Ratio'] = df['High'] / df['Low']
    # Moving averages
    for period in [5, 20, 50]:
        df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
        df[f'Close_to_SMA_{period}'] = df['Close'] / df[f'SMA_{period}']
    # RSI
    df['RSI_14'] = calculate_rsi(df['Close'], 14)
    # MACD
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    # Bollinger Bands
    sma_20 = df['Close'].rolling(20).mean()
    std_20 = df['Close'].rolling(20).std()
    df['BB_Position'] = (df['Close'] - (sma_20 - 2 * std_20)) / ((sma_20 + 2 * std_20) - (sma_20 - 2 * std_20))
    # Volume analysis
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    # ATR and ADX
    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'])
    df['ADX'] = calculate_adx(df['High'], df['Low'], df['Close'])
    # Momentum
    for period in [5, 20]:
        df[f'ROC_{period}'] = ((df['Close'] / df['Close'].shift(period)) - 1) * 100
    print(f"Created {len([col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']])} features")
    return df

# -------------------- DATA PREPARATION --------------------
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\nPreparing data...")
    df_features = create_features(df)
    df_features['Next_Return'] = df_features['Close'].shift(-1) / df_features['Close'] - 1
    df_features['Label'] = (df_features['Next_Return'] > 0.005).astype(int)  # 0.5% threshold
    df_features = df_features.dropna()
    print(f"Prepared {len(df_features)} samples")
    print(f"Class balance: {df_features['Label'].value_counts().to_dict()}")
    return df_features

# -------------------- ENSEMBLE MODEL --------------------
def create_balanced_ensemble(X_train: np.ndarray, y_train: np.ndarray, 
                           X_val: np.ndarray, y_val: np.ndarray) -> Dict:
    print("\nCreating balanced ensemble...")
    models = {}
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    sample_weights = np.array([class_weight_dict[int(y)] for y in y_train])
    
    # Random Forest
    rf_params = {
        'n_estimators': 300,  # Reduced to prevent overfitting
        'max_depth': 8,
        'min_samples_split': 50,
        'min_samples_leaf': 20,
        'max_features': 'sqrt',
        'class_weight': 'balanced_subsample',
        'random_state': 42,
        'n_jobs': -1,
        'oob_score': True
    }
    models['random_forest'] = RandomForestClassifier(**rf_params)
    models['random_forest'].fit(X_train, y_train, sample_weight=sample_weights)
    
    # Extra Trees
    et_params = {
        'n_estimators': 200,
        'max_depth': 6,
        'min_samples_split': 40,
        'min_samples_leaf': 15,
        'max_features': 0.5,
        'class_weight': 'balanced',
        'bootstrap': True,
        'random_state': 43,
        'n_jobs': -1
    }
    models['extra_trees'] = ExtraTreesClassifier(**et_params)
    models['extra_trees'].fit(X_train, y_train)
    
    # Gradient Boosting
    gb_params = {
        'n_estimators': 150,
        'max_depth': 3,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'min_samples_split': 50,
        'min_samples_leaf': 20,
        'random_state': 44
    }
    models['gradient_boost'] = GradientBoostingClassifier(**gb_params)
    models['gradient_boost'].fit(X_train, y_train, sample_weight=sample_weights)
    
    print("\nIndividual model validation performance:")
    for name, model in models.items():
        pred_proba = model.predict_proba(X_val)[:, 1]
        pred_binary = (pred_proba > 0.5).astype(int)
        acc = accuracy_score(y_val, pred_binary)
        auc = roc_auc_score(y_val, pred_proba) if len(np.unique(y_val)) > 1 else 0.5
        print(f"  {name}: Accuracy={acc:.4f}, AUC={auc:.4f}")
        if hasattr(model, 'oob_score_'):
            print(f"    OOB Score: {model.oob_score_:.4f}")
    
    return models

# -------------------- PREDICTION AND EVALUATION --------------------
def make_ensemble_predictions(models: Dict, X_test: np.ndarray) -> Tuple:
    predictions = {}
    for name, model in models.items():
        pred_proba = model.predict_proba(X_test)[:, 1]
        predictions[name] = pred_proba
    weights = {'random_forest': 0.4, 'extra_trees': 0.35, 'gradient_boost': 0.25}
    ensemble_pred = np.zeros(len(X_test))
    for name, pred in predictions.items():
        ensemble_pred += pred * weights[name]
    confidence = np.abs(ensemble_pred - 0.5)
    return ensemble_pred, predictions, weights, confidence

def comprehensive_evaluation(models: Dict, X_test: np.ndarray, y_test: np.ndarray, 
                           feature_names: List[str]) -> Dict:
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION")
    print("="*80)
    
    ensemble_pred, _, weights, confidence = make_ensemble_predictions(models, X_test)
    thresholds = np.arange(0.35, 0.65, 0.01)
    best_threshold = 0.5
    best_score = 0
    
    for threshold in thresholds:
        binary_pred = (ensemble_pred > threshold).astype(int)
        if len(np.unique(binary_pred)) == 2:
            acc = accuracy_score(y_test, binary_pred)
            # Adjust for fees
            adjusted_acc = acc - TRADING_FEES * len(y_test) / sum(abs(binary_pred - y_test))
            balance_penalty = abs(binary_pred.sum() / len(binary_pred) - 0.5) * 0.2
            adjusted_score = adjusted_acc - balance_penalty
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_threshold = threshold
    
    print(f"Optimal threshold: {best_threshold:.3f}")
    final_pred = (ensemble_pred > best_threshold).astype(int)
    accuracy = accuracy_score(y_test, final_pred)
    adjusted_accuracy = accuracy - TRADING_FEES
    auc = roc_auc_score(y_test, ensemble_pred) if len(np.unique(y_test)) > 1 else 0.5
    
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Adjusted Accuracy (with fees): {adjusted_accuracy:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(f"  Threshold: {best_threshold:.3f}")
    
    pred_dist = pd.Series(final_pred).value_counts()
    print(f"\n📊 PREDICTION DISTRIBUTION:")
    print(f"  Class 0: {pred_dist.get(0, 0)} ({pred_dist.get(0, 0)/len(final_pred)*100:.1f}%)")
    print(f"  Class 1: {pred_dist.get(1, 0)} ({pred_dist.get(1, 0)/len(final_pred)*100:.1f}%)")
    
    print(f"\n📋 CLASSIFICATION REPORT:")
    print(classification_report(y_test, final_pred))
    
    print(f"\n🎯 PERFORMANCE BY CONFIDENCE LEVEL:")
    conf_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]
    for conf_thresh in conf_thresholds:
        high_conf_mask = confidence > conf_thresh
        if high_conf_mask.sum() >= 10:
            conf_pred = final_pred[high_conf_mask]
            conf_true = y_test[high_conf_mask]
            conf_acc = accuracy_score(conf_true, conf_pred)
            print(f"  Confidence >{conf_thresh:.2f}: {high_conf_mask.sum():4d} samples "
                  f"({high_conf_mask.sum()/len(y_test)*100:5.1f}%), Accuracy: {conf_acc:.4f}")
    
    high_conf_threshold = 0.20
    high_conf_mask = confidence > high_conf_threshold
    if high_conf_mask.sum() > 0:
        print(f"\n🔥 HIGH CONFIDENCE ANALYSIS (confidence > {high_conf_threshold}):")
        high_conf_pred = final_pred[high_conf_mask]
        high_conf_true = y_test[high_conf_mask]
        high_conf_acc = accuracy_score(high_conf_true, high_conf_pred)
        print(f"  Samples: {high_conf_mask.sum()} ({high_conf_mask.sum()/len(y_test)*100:.1f}%)")
        print(f"  Accuracy: {high_conf_acc:.4f}")
        if len(np.unique(high_conf_pred)) == 2:
            conf_matrix = confusion_matrix(high_conf_true, high_conf_pred)
            print(f"  Confusion Matrix:")
            print(f"    TN={conf_matrix[0,0]:4d}, FP={conf_matrix[0,1]:4d}")
            print(f"    FN={conf_matrix[1,0]:4d}, TP={conf_matrix[1,1]:4d}")
    
    if 'random_forest' in models:
        rf_model = models['random_forest']
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(f"\n🔍 TOP 10 FEATURES:")
        for _, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']:<30}: {row['importance']:.4f}")
    
    return {
        'accuracy': accuracy,
        'adjusted_accuracy': adjusted_accuracy,
        'auc': auc,
        'optimal_threshold': best_threshold,
        'predictions': ensemble_pred,
        'binary_predictions': final_pred,
        'confidence': confidence,
        'weights': weights,
        'feature_importance': feature_importance
    }

# -------------------- MAIN EXECUTION --------------------
def main():
    print("="*80)
    print("MOMENTUM PREDICTOR - BALANCED APPROACH")
    print("="*80)
    print("Focus on robust predictions for 1h timeframe")
    
    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        # 1. Load data
        print("\n1. LOADING DATA...")
        if USE_CSV:
            df = fetch_ohlcv_from_csv(CSV_FILENAMES[symbol])
        else:
            df = fetch_ohlcv(symbol)
        print(f"Loaded {len(df)} rows")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        # 2. Prepare data
        print("\n2. PREPARING DATA...")
        df_features = prepare_data(df)
        
        if len(df_features) < 500:
            print(f"Warning: Low sample count ({len(df_features)})")
            continue
        
        # 3. Feature selection
        print("\n3. FEATURE SELECTION...")
        exclude_cols = ['Label', 'Next_Return']
        feature_cols = [col for col in df_features.columns if col not in exclude_cols]
        X_temp = df_features[feature_cols].fillna(0)
        y_temp = df_features['Label']
        mi_scores = mutual_info_classif(X_temp, y_temp, random_state=42)
        mi_df = pd.DataFrame({'feature': feature_cols, 'mi_score': mi_scores}).sort_values('mi_score', ascending=False)
        n_features = min(20, len(feature_cols))  # Reduced to 20 features
        top_features = mi_df.head(n_features)['feature'].tolist()
        print(f"Selected {len(top_features)} features based on mutual information")
        print(f"Top 10 features by MI score:")
        for _, row in mi_df.head(10).iterrows():
            print(f"  {row['feature']:<30}: {row['mi_score']:.4f}")
        
        # 4. Time-based split and walk-forward validation
        print("\n4. TRAINING AND VALIDATION...")
        X = df_features[top_features].copy()
        y = df_features['Label'].copy()
        X = X.fillna(method='ffill').fillna(0)
        print(f"Final dataset: {len(X)} samples with {len(top_features)} features")
        
        tscv = TimeSeriesSplit(n_splits=5)
        fold_accuracies = []
        fold_aucs = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            print(f"\nFold {fold}:")
            X_train, X_val = X.iloc[train_idx].values, X.iloc[val_idx].values
            y_train, y_val = y.iloc[train_idx].values, y.iloc[val_idx].values
            models = create_balanced_ensemble(X_train, y_train, X_val, y_val)
            results = comprehensive_evaluation(models, X_val, y_val, top_features)
            fold_accuracies.append(results['adjusted_accuracy'])
            fold_aucs.append(results['auc'])
            print(f"Fold {fold} - Adjusted Accuracy: {results['adjusted_accuracy']:.4f}, AUC: {results['auc']:.4f}")
        
        # Final model training on full data
        print("\n5. TRAINING FINAL MODELS...")
        X_full = X.values
        y_full = y.values
        final_models = create_balanced_ensemble(X_full, y_full, X_full, y_full)
        
        # 6. Save models and results
        print("\n6. SAVING MODELS...")
        output_dir = Path(os.getenv("OUTPUT_DIR", "models"))
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, model in final_models.items():
            joblib.dump(model, output_dir / f'{symbol.replace("/","")}_{name}_model.pkl')

        joblib.dump(top_features, output_dir / f'{symbol.replace("/","")}_features.pkl')
        joblib.dump({
            'optimal_threshold': results['optimal_threshold'],
            'n_features': len(top_features),
            'n_samples': len(df_features),
            'fold_accuracies': fold_accuracies,
            'fold_aucs': fold_aucs
        }, output_dir / f'{symbol.replace("/","")}_config.pkl')
        
        print(f"\n{'='*80}")
        print(f"MOMENTUM PREDICTOR COMPLETED FOR {symbol}!")
        print(f"{'='*80}")
        print(f"\n📊 FINAL RESULTS FOR {symbol}:")
        print(f"  Average Fold Adjusted Accuracy: {np.mean(fold_accuracies):.4f}")
        print(f"  Average Fold AUC: {np.mean(fold_aucs):.4f}")
        print(f"  Optimal Threshold: {results['optimal_threshold']:.3f}")
        print(f"\n💾 FILES SAVED:")
        print(f"  • Models: {symbol.replace('/','')}_*.pkl")
        print(f"  • Features: {symbol.replace('/','')}_features.pkl")
        print(f"  • Config: {symbol.replace('/','')}_config.pkl")

if __name__ == "__main__":
    main()