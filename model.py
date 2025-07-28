import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import datetime
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- Configuration ---
SYMBOL = "USTEC"
TIMEFRAME = mt5.TIMEFRAME_M15
ADDITIONAL_TIMEFRAMES = {
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "H6": mt5.TIMEFRAME_H6,
    "H8": mt5.TIMEFRAME_H8,
    "H12": mt5.TIMEFRAME_H12,
    "D1": mt5.TIMEFRAME_D1,
}
ATR_PERIOD = 14
RENKO_ATR_MULTIPLIER = 2
DATA_COUNT = 50000
HURST_WINDOW = 100
OUTPUT_FILE = "xauusd_ai_features.csv" # Renamed to reflect its purpose
LOCKED_ATR_FILE = "locked_atr.csv"

# --- Connect to MetaTrader 5 ---
def connect_mt5():
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return False
    print("MT5 initialized.")
    return True

# --- Fetch Historical Data ---
def get_historical_data(symbol, timeframe, count):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None:
        print(f"No data for {symbol} on timeframe {timeframe}. Error: {mt5.last_error()}")
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df = df[['open','high','low','close']]

    # Handle duplicate timestamps by taking the last entry for each timestamp
    if not df.index.is_unique:
        print(f"Warning: Duplicate timestamps found in {timeframe} data. Dropping duplicates, keeping the last.")
        df = df.loc[~df.index.duplicated(keep='last')]

    # Heiken Ashi
    ha_open = [(df.open.iloc[0]+df.close.iloc[0])/2]
    ha_close = [(df.open.iloc[0]+df.high.iloc[0]+df.low.iloc[0]+df.close.iloc[0])/4]
    for i in range(1,len(df)):
        ha_open.append((ha_open[i-1]+ha_close[i-1])/2)
        ha_close.append((df.open.iloc[i]+df.high.iloc[i]+df.low.iloc[i]+df.close.iloc[i])/4)
    df['ha_open'], df['ha_close'] = ha_open, ha_close
    df['ha_direction'] = np.where(df['ha_close']>df['ha_open'],1,0)

    print(f"Fetched and HA-calculated {len(df)} bars for timeframe {timeframe}.")
    return df

# --- Calculate ATR ---
def calculate_atr(df, period):
    df['tr'] = np.max([df.high-df.low, abs(df.high-df.close.shift()), abs(df.low-df.close.shift())], axis=0)
    df['atr'] = df['tr'].ewm(span=period, adjust=False).mean()
    return df['atr']

# --- Hurst Exponent ---
def calculate_hurst(ts, lags_max=20):
    lags = np.arange(2, lags_max)
    if len(lags) == 0:
        return np.nan
    tau = [np.sqrt(np.std(ts[lag:]-ts[:-lag])) for lag in lags]
    if len(tau) == 0 or np.all(np.isnan(tau)):
        return np.nan
    # Filter out NaN or infinite values from tau before polyfit
    valid_indices = np.isfinite(np.log(lags)) & np.isfinite(np.log(tau))
    if not np.any(valid_indices):
        return np.nan
    
    lags_filtered = lags[valid_indices]
    tau_filtered = np.array(tau)[valid_indices]
    
    if len(lags_filtered) < 2: # Need at least 2 points for polyfit
        return np.nan

    poly = np.polyfit(np.log(lags_filtered), np.log(tau_filtered), 1)
    return poly[0]*2.0

# --- Generate Renko Bars ---
def generate_atr_renko(df, brick_size_col='brick_size', ha_direction_col='ha_direction', hurst_col='hurst'):
    bars = []
    if df.empty:
        return pd.DataFrame()

    if brick_size_col not in df.columns:
        print(f"Error: '{brick_size_col}' column not found in DataFrame for Renko generation.")
        return pd.DataFrame()

    if not df.index.is_unique:
        print(f"Warning: Duplicate timestamps found in DataFrame for Renko generation. Dropping duplicates, keeping the last.")
        df = df.loc[~df.index.duplicated(keep='last')]
        if df.empty:
            return pd.DataFrame()

    first = df.iloc[0]
    bars.append({'open': first.open, 'close': first.open, 'direction': 0,
                  'brick_size': first[brick_size_col], 'real_close': first.close,
                  'ha_direction': first[ha_direction_col], 'hurst': first[hurst_col] if hurst_col in df.columns else np.nan, 'time': df.index[0]})

    for time, row in df.iterrows():
        last = bars[-1]
        size = row[brick_size_col]
        ref = last['close']

        current_renko_direction = last['direction']

        if row.close > ref + size:
            n = int((row.close - ref) / size)
            for _ in range(n):
                ref += size
                bars.append({'open': ref - size, 'close': ref, 'direction': 1, 'brick_size': size,
                              'real_close': row.close, 'ha_direction': row[ha_direction_col],
                              'hurst': row[hurst_col] if hurst_col in df.columns else np.nan, 'time': time})
            current_renko_direction = 1

        elif row.close < ref - size:
            n = int((ref - row.close) / size)
            for _ in range(n):
                ref -= size
                bars.append({'open': ref + size, 'close': ref, 'direction': -1, 'brick_size': size,
                              'real_close': row.close, 'ha_direction': row[ha_direction_col],
                              'hurst': row[hurst_col] if hurst_col in df.columns else np.nan, 'time': time})
            current_renko_direction = -1

        # Check for reversals
        if current_renko_direction == 1 and row.low < ref - size:
            ref -= size
            bars.append({'open': ref + size, 'close': ref, 'direction': -1, 'brick_size': size,
                          'real_close': row.close, 'ha_direction': row[ha_direction_col],
                          'hurst': row[hurst_col] if hurst_col in df.columns else np.nan, 'time': time})
        elif current_renko_direction == -1 and row.high > ref + size:
            ref += size
            bars.append({'open': ref - size, 'close': ref, 'direction': 1, 'brick_size': size,
                          'real_close': row.close, 'ha_direction': row[ha_direction_col],
                          'hurst': row[hurst_col] if hurst_col in df.columns else np.nan, 'time': time})

    renko_df = pd.DataFrame(bars).set_index('time')
    
    if not renko_df.index.is_unique:
        print(f"Warning: Duplicate timestamps found in generated Renko DataFrame. Dropping duplicates, keeping the last.")
        renko_df = renko_df.loc[~renko_df.index.duplicated(keep='last')]
    
    return renko_df

# --- AI Target Variable Creation ---
def create_target_variable(df_combined, look_ahead_bars=5, profit_multiplier=1.0, loss_multiplier=1.0):
    """
    Creates target variables for long and short trade profitability.
    
    Args:
        df_combined (pd.DataFrame): The DataFrame with all features and 'real_close'.
        look_ahead_bars (int): How many future bars to look for price movement.
        profit_multiplier (float): Multiplier of the average brick size for profit target.
        loss_multiplier (float): Multiplier of the average brick size for stop loss.

    Returns:
        pd.DataFrame: DataFrame with 'target_long' and 'target_short' columns.
    """
    df = df_combined.copy()
    
    # Calculate an average "move" size based on the average brick size.
    # This helps in defining a realistic profit/loss target.
    # If brick_size is mostly NaN or zero, use a default small value
    base_move_size = df['brick_size'].replace([0, np.inf, -np.inf], np.nan).dropna().mean()
    if pd.isna(base_move_size) or base_move_size == 0:
        # Fallback if brick_size calculation fails or is not available
        base_move_size = 0.0001 # A very small value to avoid division by zero

    profit_target = base_move_size * profit_multiplier
    stop_loss_target = base_move_size * loss_multiplier

    # Initialize target columns
    df['target_long'] = 0 # 1 if profitable long, 0 otherwise
    df['target_short'] = 0 # 1 if profitable short, 0 otherwise

    # Calculate future price dynamics
    # We need to look at the 'high' and 'low' within the look_ahead_bars for proper signal definition
    # This is a simplified approach. A more robust way would be to check if profit/loss is hit
    # within the next N bars before an opposing signal or the N bars are up.

    for i in range(len(df) - look_ahead_bars):
        current_close = df['real_close'].iloc[i]
        
        # Slice the future data
        future_data = df.iloc[i+1 : i+1+look_ahead_bars]
        
        if not future_data.empty:
            future_high = future_data['real_close'].max() # Simplistic: highest close in future
            future_low = future_data['real_close'].min()  # Simplistic: lowest close in future
            
            # Condition for a potentially profitable long trade
            # Price moved up by profit_target, and it didn't hit stop loss first (simplistic check)
            if (future_high >= current_close + profit_target): # and (future_low > current_close - stop_loss_target): # Add stop loss check for realism
                df.loc[df.index[i], 'target_long'] = 1
            
            # Condition for a potentially profitable short trade
            # Price moved down by profit_target, and it didn't hit stop loss first
            if (future_low <= current_close - profit_target): # and (future_high < current_close + stop_loss_target): # Add stop loss check for realism
                df.loc[df.index[i], 'target_short'] = 1
                
    # Drop rows where future data could not be calculated
    return df.iloc[:-look_ahead_bars].copy()


# --- AI Model Training and Evaluation ---
def train_and_evaluate_model(df_combined):
    """
    Trains and evaluates RandomForestClassifier models for long and short trade signals.
    """
    df_ml = create_target_variable(df_combined.copy())

    # Define features (X) and target (y)
    # Exclude original price columns, time, and directly target-related columns
    features = [col for col in df_ml.columns if col not in ['real_close', 'target_long', 'target_short']]
    X = df_ml[features]
    
    # Handle NaNs that might have been introduced by shifting or dropping earlier
    # Use median for numerical features, mode for categorical (like directions)
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].nunique() < 10: # Likely categorical or low cardinality numerical
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 0)
        else: # Numerical
            X[col] = X[col].fillna(X[col].median())

    y_long = df_ml['target_long']
    y_short = df_ml['target_short']

    print(f"\nTraining data shape: {X.shape}")
    print(f"Long target distribution:\n{y_long.value_counts(normalize=True)}")
    print(f"Short target distribution:\n{y_short.value_counts(normalize=True)}")

    # --- Train for Long Trades ---
    print("\n--- Training Model for Long Trades ---")
    # Stratify to maintain class distribution in train/test splits, important for imbalanced classes
    X_train_long, X_test_long, y_train_long, y_test_long = train_test_split(X, y_long, test_size=0.2, random_state=42, stratify=y_long)
    
    # class_weight='balanced' helps with imbalanced datasets
    model_long = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1) 
    model_long.fit(X_train_long, y_train_long)
    
    y_pred_long = model_long.predict(X_test_long)
    print("Long Trade Model Performance (Test Set):")
    print(classification_report(y_test_long, y_pred_long))
    print("Accuracy:", accuracy_score(y_test_long, y_pred_long))
    print("Confusion Matrix:\n", confusion_matrix(y_test_long, y_pred_long))
    
    # Feature Importances for Long Model
    if hasattr(model_long, 'feature_importances_'):
        importances_long = pd.Series(model_long.feature_importances_, index=features).sort_values(ascending=False)
        print("\nTop 10 Feature Importances (Long Model):\n", importances_long.head(10))

    # --- Train for Short Trades ---
    print("\n--- Training Model for Short Trades ---")
    X_train_short, X_test_short, y_train_short, y_test_short = train_test_split(X, y_short, test_size=0.2, random_state=42, stratify=y_short)
    
    model_short = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    model_short.fit(X_train_short, y_train_short)
    
    y_pred_short = model_short.predict(X_test_short)
    print("Short Trade Model Performance (Test Set):")
    print(classification_report(y_test_short, y_pred_short))
    print("Accuracy:", accuracy_score(y_test_short, y_pred_short))
    print("Confusion Matrix:\n", confusion_matrix(y_test_short, y_pred_short))

    # Feature Importances for Short Model
    if hasattr(model_short, 'feature_importances_'):
        importances_short = pd.Series(model_short.feature_importances_, index=features).sort_values(ascending=False)
        print("\nTop 10 Feature Importances (Short Model):\n", importances_short.head(10))
        
    return model_long, model_short, features # Return trained models and feature names

# --- Main ---
def main():
    if not connect_mt5(): return
    
    df = get_historical_data(SYMBOL, TIMEFRAME, DATA_COUNT)
    if df.empty:
        mt5.shutdown()
        return

    df['atr'] = calculate_atr(df, ATR_PERIOD)
    df['hurst'] = df['close'].rolling(window=HURST_WINDOW).apply(lambda x: calculate_hurst(x.values, lags_max=min(len(x)//2, 20)), raw=False)
    
    if os.path.exists(LOCKED_ATR_FILE):
        locked = pd.read_csv(LOCKED_ATR_FILE)
        brick_atr = locked['atr'].iloc[0]
        print(f"Using locked ATR for brick size: {brick_atr}")
    else:
        if len(df) <= HURST_WINDOW or df['atr'].isnull().all():
            print(f"Warning: Not enough valid ATR data ({len(df)} bars) to calculate ATR at HURST_WINDOW ({HURST_WINDOW}). Using default or last valid ATR value.")
            brick_atr = df['atr'].dropna().iloc[-1] if not df['atr'].dropna().empty else 0.0001
        else:
            # Take the ATR from the point where Hurst is first available, or last available if Hurst not fully calculated
            valid_atr_for_brick = df['atr'].iloc[HURST_WINDOW-1] if len(df) >= HURST_WINDOW else df['atr'].dropna().iloc[-1]
            brick_atr = valid_atr_for_brick if not pd.isna(valid_atr_for_brick) else df['atr'].dropna().iloc[-1] if not df['atr'].dropna().empty else 0.0001
        
        pd.DataFrame({'atr': [brick_atr]}).to_csv(LOCKED_ATR_FILE, index=False)
        print(f"Calculated and locked ATR for brick size: {brick_atr}")

    df['brick_size'] = brick_atr * RENKO_ATR_MULTIPLIER
    
    initial_len = len(df)
    df.dropna(subset=['atr', 'hurst'], inplace=True)
    if len(df) < initial_len:
        print(f"Dropped {initial_len - len(df)} rows from primary dataframe due to NaN in ATR or Hurst.")
    if df.empty:
        print("Primary DataFrame became empty after dropping NaNs. Exiting.")
        mt5.shutdown()
        return

    renko_m15 = generate_atr_renko(df)
    
    main_df = pd.DataFrame(index=df.index)
    main_df['real_close'] = df['close'] # Keep real_close for target calculation but not as a direct feature
    main_df['ha_direction'] = df['ha_direction']

    if not renko_m15.empty:
        # Renko direction based on the last Renko bar at each M15 timestamp
        main_df['renko_direction'] = renko_m15['direction'].reindex(main_df.index, method='ffill')
    else:
        main_df['renko_direction'] = np.nan 

    # Add original ATR, Hurst, and brick_size to main_df for features
    main_df['atr'] = df['atr']
    main_df['hurst'] = df['hurst']
    main_df['brick_size'] = df['brick_size'] # Important for target variable definition

    print("\n--- Processing Additional Timeframes ---")
    for tf_name, tf_mt5 in ADDITIONAL_TIMEFRAMES.items():
        print(f"Fetching data for {tf_name}...")
        tf_df = get_historical_data(SYMBOL, tf_mt5, DATA_COUNT)
        if tf_df.empty:
            print(f"Skipping {tf_name} due to empty data.")
            main_df[f'ha_direction_{tf_name}_shifted'] = np.nan
            main_df[f'renko_direction_{tf_name}_shifted'] = np.nan
            continue

        tf_df['atr'] = calculate_atr(tf_df, ATR_PERIOD)
        tf_df['hurst'] = tf_df['close'].rolling(window=HURST_WINDOW).apply(lambda x: calculate_hurst(x.values, lags_max=min(len(x)//2, 20)), raw=False)
        
        tf_initial_len = len(tf_df)
        tf_df.dropna(subset=['atr', 'hurst'], inplace=True)
        if len(tf_df) < tf_initial_len:
            print(f"Dropped {tf_initial_len - len(tf_df)} rows from {tf_name} due to NaN in ATR or Hurst.")
        if tf_df.empty:
            print(f"{tf_name} DataFrame became empty after dropping NaNs. Skipping.")
            main_df[f'ha_direction_{tf_name}_shifted'] = np.nan
            main_df[f'renko_direction_{tf_name}_shifted'] = np.nan
            continue

        tf_brick_atr = tf_df['atr'].iloc[HURST_WINDOW-1] if len(tf_df) >= HURST_WINDOW else tf_df['atr'].dropna().iloc[-1] if not tf_df['atr'].dropna().empty else 0.0001
        tf_df['brick_size'] = tf_brick_atr * RENKO_ATR_MULTIPLIER
        
        tf_renko = generate_atr_renko(tf_df, ha_direction_col='ha_direction', hurst_col='hurst')

        temp_ha = tf_df['ha_direction'].reindex(main_df.index, method='ffill')
        main_df[f'ha_direction_{tf_name}_shifted'] = temp_ha.shift(1)

        if not tf_renko.empty:
            temp_renko = tf_renko['direction'].reindex(main_df.index, method='ffill')
            main_df[f'renko_direction_{tf_name}_shifted'] = temp_renko.shift(1)
        else:
            main_df[f'renko_direction_{tf_name}_shifted'] = np.nan 

    initial_main_len = len(main_df)
    # Ensure real_close is not NaN for target creation later
    main_df.dropna(subset=['real_close'], inplace=True)
    # Only drop rows where shifted multi-timeframe data is missing, this is essential for AI features
    cols_to_check_for_nan = [col for col in main_df.columns if '_shifted' in col]
    if cols_to_check_for_nan: # Only if there are shifted columns
        main_df.dropna(subset=cols_to_check_for_nan, inplace=True)

    if len(main_df) < initial_main_len:
        print(f"Dropped {initial_main_len - len(main_df)} rows from combined main_df due to initial NaNs from shifting or missing higher timeframe data.")
    
    if main_df.empty:
        print("Final combined DataFrame is empty after dropping NaNs. Cannot train AI.")
        mt5.shutdown()
        return

    print("\nFinal DataFrame (features for AI) head:")
    print(main_df.head(10))
    print("\nFinal DataFrame (features for AI) tail:")
    print(main_df.tail(10))

    main_df.to_csv(OUTPUT_FILE)
    print(f"Features and target data saved to {OUTPUT_FILE}")

    # Train AI models based on the prepared data
    # The models and their features_used would typically be saved here using pickle or joblib
    # for later loading and prediction in a live trading script.
    model_long, model_short, features_used = train_and_evaluate_model(main_df.copy())
    
    # Example of how you might save the models
    # import joblib
    # joblib.dump(model_long, 'model_long.pkl')
    # joblib.dump(model_short, 'model_short.pkl')
    # joblib.dump(features_used, 'features_used.pkl')


    mt5.shutdown()

if __name__ == "__main__":
    main()
