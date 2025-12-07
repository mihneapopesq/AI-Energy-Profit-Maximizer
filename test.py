import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb 
import pulp
import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ==============================================================================
# 0. CONSTANTS ȘI PARAMETRI
# ==============================================================================

NUM_DAYS = 8
INTERVALS_PER_HOUR = 4
INTERVALS_PER_DAY = 24 * INTERVALS_PER_HOUR # 96
TOTAL_INTERVALS = NUM_DAYS * INTERVALS_PER_DAY # 768

C_MAX = 10.0 # Battery Capacity (MWh)
P_MAX = 10.0 # Max Power/Position
SOC_START = 5.0 # Initial SoC at 00:00
MIN_TRANZACTIE = 0.1 # Minimum transaction magnitude

# Retained the successful numerical safety margins for the MILP SoC bounds
SOC_MIN_MILP = 0.01
SOC_MAX_MILP = C_MAX - 0.01 # 9.99

# LightGBM Parameters
N_ESTIMATORS = 300 
LGBM_PARAMS = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'n_estimators': N_ESTIMATORS,
    'learning_rate': 0.03,
    'num_leaves': 128,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1.0,
    'reg_lambda': 2.0,
    'min_child_samples': 20,
    'n_jobs': -1,
    'verbose': -1,
    'seed': 42
}

# XGBoost Parameters for the ensemble
XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'n_estimators': N_ESTIMATORS,
    'learning_rate': 0.05,
    'max_depth': 6,
    'n_jobs': -1,
    'seed': 42
}

# ==============================================================================
# 1. PREGATIREA DATELOR ȘI FEATURE ENGINEERING
# ==============================================================================

def create_time_features(df):
    
    if 'Start' not in df.columns:
        df['Start'] = pd.to_datetime(df['Time interval (CET/CEST)'].str.split(' - ').str[0], format="%d.%m.%Y %H:%M")

    # Time Features 
    df['hour'] = df['Start'].dt.hour
    df['minute'] = df['Start'].dt.minute
    df['dow'] = df['Start'].dt.dayofweek
    df['dom'] = df['Start'].dt.day
    df['month'] = df['Start'].dt.month
    df['year'] = df['Start'].dt.year
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    # Sinusoidal features
    df['hour_sin'] = np.sin(2 * np.pi * (df['hour'] + df.minute/60) / 24)
    df['hour_cos'] = np.cos(2 * np.pi * (df['hour'] + df.minute/60) / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
    
    # Lag Features 
    if 'Price' in df.columns:
        df['price_lag_96'] = df['Price'].shift(96)     # 1 zi
        df['price_lag_48'] = df['Price'].shift(48)     # 12 ore
        df['price_lag_672'] = df['Price'].shift(672)   # 7 zile

    return df.dropna().reset_index(drop=True)

# Data Loading and Feature Preparation
try:
    df_raw = pd.read_csv("Dataset.csv")
    df_raw = df_raw.sort_values(by='Time interval (CET/CEST)', key=lambda x: pd.to_datetime(x.str.split(' - ').str[0], format="%d.%m.%Y %H:%M")).reset_index(drop=True)
    
    df_train = create_time_features(df_raw.copy())
    
    EXCLUDE_COLS = ['Time interval (CET/CEST)', 'Start', 'Price']
    FEATURE_COLS = [col for col in df_train.columns if col not in EXCLUDE_COLS]
    
    MAX_LAG = 672 

    print("1. Historical data loaded and preprocessed.")
    print(f"   Training rows count: {len(df_train)}")

except FileNotFoundError:
    print("ERROR: 'Dataset.csv' file not found. Cannot proceed.")
    exit()

# ==============================================================================
# 2. ENSEMBLE TRAINING AND ITERATIVE PREDICTION
# ==============================================================================

X = df_train[FEATURE_COLS]
y = df_train['Price']

X_train, X_val = X.iloc[:-96], X.iloc[-96:]
y_train, y_val = y.iloc[:-96], y.iloc[-96:]

# 1. Ensemble Training
print("\n2. Training LightGBM and XGBoost Ensemble...")
lgbm = lgb.LGBMRegressor(**LGBM_PARAMS)
xgb_model = xgb.XGBRegressor(**XGB_PARAMS)

# LightGBM (retains early stopping)
lgbm.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='mae',
    callbacks=[lgb.early_stopping(50, verbose=False)]
)

# 💥 FIX: Removed 'early_stopping_rounds' from XGBoost fit call
xgb_model.fit(X_train, y_train, 
              eval_set=[(X_val, y_val)], 
              verbose=False)

# Due to the fix, we rely on n_estimators for XGBoost completion
print(f"   Training finished. LGBM iterations: {lgbm.best_iteration_}, XGB completed {xgb_model.n_estimators} estimators.")


# 2. Prediction Data Preparation
last_time_str = df_raw['Time interval (CET/CEST)'].str.split(' - ').str[0].iloc[-1]
last_time = pd.to_datetime(last_time_str, format="%d.%m.%Y %H:%M")

future_timestamps = [last_time + pd.Timedelta(minutes=15 * (i + 1)) for i in range(TOTAL_INTERVALS)]

df_pred = pd.DataFrame({'Start': future_timestamps})
df_pred['Time interval (CET/CEST)'] = df_pred['Start'].apply(lambda x: x.strftime("%d.%m.%Y %H:%M"))

df_pred_template = create_time_features(df_pred.copy())

for col in ['price_lag_96', 'price_lag_48', 'price_lag_672']:
    if col in FEATURE_COLS:
        df_pred_template[col] = 0.0

X_pred_template = df_pred_template[FEATURE_COLS].copy() 


# 3. Iterative Prediction (Recursive Ensemble Average)
print("3. Iterative Ensemble Prediction (8 days)...")

full_history = df_train['Price'].iloc[-MAX_LAG:].tolist() 
predicted_prices = []

for i in tqdm(range(TOTAL_INTERVALS), desc="Predicting"):
    current_idx_in_history = len(full_history) - 1
    
    X_pred_row = X_pred_template.iloc[[i]].copy()
    
    # Update Lag Features
    X_pred_row.loc[X_pred_row.index[0], 'price_lag_96'] = full_history[current_idx_in_history - 95]
    X_pred_row.loc[X_pred_row.index[0], 'price_lag_48'] = full_history[current_idx_in_history - 47]
    X_pred_row.loc[X_pred_row.index[0], 'price_lag_672'] = full_history[current_idx_in_history - 671]
    
    # Make prediction using ensemble average
    pred_lgbm = lgbm.predict(X_pred_row)[0]
    pred_xgb = xgb_model.predict(X_pred_row)[0]
    pred_val = (pred_lgbm + pred_xgb) / 2
    
    predicted_prices.append(pred_val)
    
    # Update history
    full_history.append(pred_val)

df_pred['Predicted_Price'] = predicted_prices

# Price Smoothing (Generalization layer)
SMOOTHING_WINDOW = 4 
df_pred['Smoothed_Price'] = df_pred['Predicted_Price'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()

print("   Prediction finished and smoothed ✓")

# ==============================================================================
# 4. MIXED-INTEGER LINEAR PROGRAMMING (MILP) OPTIMIZATION
# ==============================================================================

def solve_daily_milp(day_prices):
    T = INTERVALS_PER_DAY
    prob = pulp.LpProblem("Battery_Trading_Optimization_MILP", pulp.LpMaximize)
    
    # Decision Variables
    position = pulp.LpVariable.dicts("Position", range(1, T + 1), lowBound=-P_MAX, upBound=P_MAX)
    soc = pulp.LpVariable.dicts("SoC", range(1, T + 1), lowBound=SOC_MIN_MILP, upBound=SOC_MAX_MILP)
    surplus = pulp.LpVariable("Surplus", lowBound=0, upBound=C_MAX)
    
    # Binary Variables
    charge = pulp.LpVariable.dicts("Charge", range(1, T + 1), cat='Binary')
    discharge = pulp.LpVariable.dicts("Discharge", range(1, T + 1), cat='Binary')

    pret_minim_zi = min(day_prices)

    # Objective Function (Retained the successful form: Maximize (Surplus * P_min) - Sum(P*Q))
    cost_tranzactii = pulp.lpSum([position[t] * day_prices[t-1] for t in range(1, T + 1)])
    venit_surplus = surplus * pret_minim_zi
    
    prob += venit_surplus - cost_tranzactii, "Profit_Total"

    # Battery & State of Charge Constraints
    prob += soc[1] == SOC_START + position[1], "SoC_init"
    
    for t in range(2, T + 1):
        prob += soc[t] == soc[t-1] + position[t], f"SoC_t_{t}"
        
    prob += soc[T] == surplus, "SoC_Final_Surplus"

    # MILP Constraints for Non-Zero and Minimum Transaction (>= 0.1 MWh)
    for t in range(1, T + 1):
        prob += charge[t] + discharge[t] == 1, f"Action_Mandatory_{t}"
        
        prob += position[t] >= MIN_TRANZACTIE * charge[t] - P_MAX * discharge[t], f"Position_Lower_{t}"
        prob += position[t] <= P_MAX * charge[t] - MIN_TRANZACTIE * discharge[t], f"Position_Upper_{t}"

    prob.solve(pulp.PULP_CBC_CMD(msg=0)) 
    
    if pulp.LpStatus[prob.status] == "Optimal":
        actions = [position[t].varValue if position[t].varValue is not None else 0.0 for t in range(1, T + 1)]
        return actions
    else:
        print(f"   Warning: Optimization failed. Status: {pulp.LPStatus[prob.status]}. Using fallback.")
        return [MIN_TRANZACTIE] * T

# Run MILP over the 8 Days
all_actions = []
print("\n4. MILP Optimization of actions (daily)...")

for day in tqdm(range(NUM_DAYS), desc="MILP Solver"):
    start_idx = day * INTERVALS_PER_DAY
    end_idx = (day + 1) * INTERVALS_PER_DAY
    
    # Use the Smoothed Prices for generalization
    day_prices = df_pred['Smoothed_Price'][start_idx:end_idx].tolist()
    
    actions_day = solve_daily_milp(day_prices)
    all_actions.extend(actions_day)

print("   Optimization finished. Total actions: ", len(all_actions))

# ==============================================================================
# 5. FINAL EXPORT
# ==============================================================================

# Rounding to 4 decimal places for submission format
final_actions = [round(action, 4) if action is not None else 0.0 for action in all_actions]

def format_time_interval_strict(timestamp):
    """Generates the required time interval format: DD.MM.YYYY HH:MM - DD.MM.YYYY HH:MM"""
    start_time = timestamp
    end_time = timestamp + pd.Timedelta(minutes=15)
    
    fmt = '%d.%m.%Y %H:%M'
    return f"{start_time.strftime(fmt)} - {end_time.strftime(fmt)}"

submission_df = pd.DataFrame()
submission_df['Time interval (CET/CEST)'] = df_pred['Start'].apply(format_time_interval_strict)
submission_df['Position'] = final_actions

# Export
submission_df.to_csv('submission.csv', index=False)
print("\n5. File 'submission.csv' has been generated.")
print("\n--- First 5 actions (Final Format) ---")
print(submission_df.head())