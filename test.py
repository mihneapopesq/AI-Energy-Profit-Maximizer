import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb 
import pulp
import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 0. CONSTANTS ȘI PARAMETRI
# ==============================================================================

NUM_DAYS = 8
INTERVALS_PER_HOUR = 4
INTERVALS_PER_DAY = 24 * INTERVALS_PER_HOUR # 96
TOTAL_INTERVALS = NUM_DAYS * INTERVALS_PER_DAY # 768

C_MAX = 10.0 
P_MAX = 10.0 
SOC_START = 5.0 
MIN_TRANZACTIE = 0.1 

# ROBUST MILP BOUNDS
SOC_MIN_MILP = 0.01
SOC_MAX_MILP = C_MAX - 0.01 

# Ensemble weighting (Balanced 50/50 for maximum generalization)
ALPHA_LGBM = 0.5          
ALPHA_XGB = 0.5

# Smoothing window (Reduced to 3 for sharpness)
SMOOTHING_WINDOW = 3 # 45 minutes

# LightGBM Parameters (High Precision)
N_ESTIMATORS = 600 # Increased for detail
LGBM_PARAMS = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'n_estimators': N_ESTIMATORS,
    'learning_rate': 0.015, # Very slow, precise learning
    'num_leaves': 128,
    'max_depth': 10,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1.0,
    'reg_lambda': 2.0,
    'min_child_samples': 20,
    'n_jobs': -1,
    'verbose': -1,
    'seed': 42
}

# XGBoost Parameters
XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'n_estimators': N_ESTIMATORS,
    'learning_rate': 0.03, # Tuned for ensemble
    'max_depth': 8,        
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
    df['hour_sin'] = np.sin(2 * np.pi * (df['hour'] + df['minute'] / 60) / 24)
    df['hour_cos'] = np.cos(2 * np.pi * (df['hour'] + df['minute'] / 60) / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
    
    # Lag Features (The Stable Core)
    if 'Price' in df.columns:
        df['price_lag_96'] = df['Price'].shift(96)     # 24 hours
        df['price_lag_48'] = df['Price'].shift(48)     # 12 hours
        df['price_lag_672'] = df['Price'].shift(672)   # 7 days
        
        # 💥 NEW SAFE FEATURE: Rolling Stats on the 24h LAG
        # "What was the average price 24 hours ago (over a 1-hour window)?"
        # This adds trend context WITHOUT using recursive predictions. It is safe.
        df['lag96_roll_mean_4'] = df['price_lag_96'].rolling(window=4, min_periods=1).mean()
        df['lag96_roll_std_4'] = df['price_lag_96'].rolling(window=4, min_periods=1).std()

    return df.dropna().reset_index(drop=True)

# Data Loading
try:
    df_raw = pd.read_csv("Dataset.csv")
    df_raw = df_raw.sort_values(by='Time interval (CET/CEST)', key=lambda x: pd.to_datetime(x.str.split(' - ').str[0], format="%d.%m.%Y %H:%M")).reset_index(drop=True)
    
    df_train = create_time_features(df_raw.copy())
    
    EXCLUDE_COLS = ['Time interval (CET/CEST)', 'Start', 'Price']
    FEATURE_COLS = [col for col in df_train.columns if col not in EXCLUDE_COLS]
    
    MAX_LAG = 672
    
    # Relaxed Clipping (0.001 - 0.999) to catch more profit
    PRICE_Q_LOW = df_train['Price'].quantile(0.001)
    PRICE_Q_HIGH = df_train['Price'].quantile(0.999)

    print("1. Historical data loaded.")
    print(f"   Training rows: {len(df_train)}")

except FileNotFoundError:
    print("ERROR: 'Dataset.csv' file not found.")
    exit()

# ==============================================================================
# 2. ENSEMBLE TRAINING
# ==============================================================================

X = df_train[FEATURE_COLS]
y = df_train['Price']

X_train, X_val = X.iloc[:-96], X.iloc[-96:]
y_train, y_val = y.iloc[:-96], y.iloc[-96:]

print("\n2. Training Ensemble...")
lgbm = lgb.LGBMRegressor(**LGBM_PARAMS)
xgb_model = xgb.XGBRegressor(**XGB_PARAMS)

lgbm.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='mae',
    callbacks=[lgb.early_stopping(50, verbose=False)]
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print(f"   Training finished.")

# ==============================================================================
# 3. ITERATIVE PREDICTION
# ==============================================================================

# Prediction Prep
last_time_str = df_raw['Time interval (CET/CEST)'].str.split(' - ').str[0].iloc[-1]
last_time = pd.to_datetime(last_time_str, format="%d.%m.%Y %H:%M")
future_timestamps = [last_time + pd.Timedelta(minutes=15 * (i + 1)) for i in range(TOTAL_INTERVALS)]

df_pred = pd.DataFrame({'Start': future_timestamps})
df_pred['Time interval (CET/CEST)'] = df_pred['Start'].dt.strftime("%d.%m.%Y %H:%M")

df_pred_template = create_time_features(df_pred.copy())
# Reset dynamic columns
for col in ['price_lag_96', 'price_lag_48', 'price_lag_672', 'lag96_roll_mean_4', 'lag96_roll_std_4']:
    if col in FEATURE_COLS:
        df_pred_template[col] = 0.0

X_pred_template = df_pred_template[FEATURE_COLS].copy()

print("3. Iterative Ensemble Prediction...")

full_history = df_train['Price'].iloc[-MAX_LAG:].tolist()
predicted_prices = []

for i in tqdm(range(TOTAL_INTERVALS), desc="Predicting"):
    current_idx_in_history = len(full_history) - 1
    
    X_pred_row = X_pred_template.iloc[[i]].copy()
    
    # Update Lags (Known history)
    lag96_val = full_history[current_idx_in_history - 95]
    X_pred_row.loc[X_pred_row.index[0], 'price_lag_96'] = lag96_val
    X_pred_row.loc[X_pred_row.index[0], 'price_lag_48'] = full_history[current_idx_in_history - 47]
    X_pred_row.loc[X_pred_row.index[0], 'price_lag_672'] = full_history[current_idx_in_history - 671]
    
    # Update Rolling on Lags (Safe)
    # Get the slice of history around t-96 to calculate roll stats
    # Indices: t-96-3, t-96-2, t-96-1, t-96 (Window=4)
    idx_center = current_idx_in_history - 95
    hist_slice = full_history[idx_center - 3 : idx_center + 1]
    
    X_pred_row.loc[X_pred_row.index[0], 'lag96_roll_mean_4'] = np.mean(hist_slice)
    X_pred_row.loc[X_pred_row.index[0], 'lag96_roll_std_4'] = np.std(hist_slice)

    # Ensemble
    pred_lgbm = float(lgbm.predict(X_pred_row)[0])
    pred_xgb = float(xgb_model.predict(X_pred_row)[0])
    pred_val = ALPHA_LGBM * pred_lgbm + ALPHA_XGB * pred_xgb

    # Relaxed Clipping
    pred_val = float(np.clip(pred_val, PRICE_Q_LOW, PRICE_Q_HIGH))
    
    predicted_prices.append(pred_val)
    full_history.append(pred_val)

df_pred['Predicted_Price'] = predicted_prices

# Sharper Smoothing (Window=3)
df_pred['Smoothed_Price'] = df_pred['Predicted_Price'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()

print("   Prediction finished.")

# ==============================================================================
# 4. MILP OPTIMIZATION
# ==============================================================================

def solve_daily_milp(day_prices):
    T = INTERVALS_PER_DAY
    prob = pulp.LpProblem("Battery_Trading", pulp.LpMaximize)
    
    # Variables
    position = pulp.LpVariable.dicts("Position", range(1, T + 1), lowBound=-P_MAX, upBound=P_MAX)
    soc = pulp.LpVariable.dicts("SoC", range(1, T + 1), lowBound=SOC_MIN_MILP, upBound=SOC_MAX_MILP)
    surplus = pulp.LpVariable("Surplus", lowBound=0, upBound=C_MAX)
    
    # Binaries
    charge = pulp.LpVariable.dicts("Charge", range(1, T + 1), cat='Binary')
    discharge = pulp.LpVariable.dicts("Discharge", range(1, T + 1), cat='Binary')

    pret_minim_zi = min(day_prices)

    # Objective
    cost_tranzactii = pulp.lpSum(position[t] * day_prices[t - 1] for t in range(1, T + 1))
    venit_surplus = surplus * pret_minim_zi
    
    prob += venit_surplus - cost_tranzactii, "Profit_Total"

    # Constraints
    prob += soc[1] == SOC_START + position[1], "SoC_init"
    for t in range(2, T + 1):
        prob += soc[t] == soc[t - 1] + position[t], f"SoC_t_{t}"
    prob += soc[T] == surplus, "SoC_Final_Surplus"

    for t in range(1, T + 1):
        prob += charge[t] + discharge[t] == 1, f"Action_Mandatory_{t}"
        prob += position[t] >= MIN_TRANZACTIE * charge[t] - P_MAX * discharge[t], f"LB_{t}"
        prob += position[t] <= P_MAX * charge[t] - MIN_TRANZACTIE * discharge[t], f"UB_{t}"

    prob.solve(pulp.PULP_CBC_CMD(msg=0)) 
    
    if pulp.LpStatus[prob.status] == "Optimal":
        return [position[t].varValue if position[t].varValue is not None else 0.0 for t in range(1, T + 1)]
    else:
        return [MIN_TRANZACTIE] * T

# Run MILP
all_actions = []
print("\n4. MILP Optimization...")

for day in tqdm(range(NUM_DAYS), desc="MILP Solver"):
    start_idx = day * INTERVALS_PER_DAY
    end_idx = (day + 1) * INTERVALS_PER_DAY
    day_prices = df_pred['Smoothed_Price'][start_idx:end_idx].tolist()
    all_actions.extend(solve_daily_milp(day_prices))

print("   Optimization finished.")

# ==============================================================================
# 5. EXPORT
# ==============================================================================

final_actions = [round(action, 4) if action is not None else 0.0 for action in all_actions]

def format_time_interval_strict(timestamp):
    end = timestamp + pd.Timedelta(minutes=15)
    return f"{timestamp.strftime('%d.%m.%Y %H:%M')} - {end.strftime('%d.%m.%Y %H:%M')}"

submission_df = pd.DataFrame()
submission_df['Time interval (CET/CEST)'] = df_pred['Start'].apply(format_time_interval_strict)
submission_df['Position'] = final_actions

submission_df.to_csv('submission.csv', index=False)
print("\n5. File 'submission.csv' generated.")
