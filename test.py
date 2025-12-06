import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb  # <--- Librăria nouă
import pulp
from tqdm import tqdm

# ==========================================
# 0. CONFIGURĂRI
# ==========================================
NUM_DAYS = 8
INTERVALS_PER_DAY = 96
TOTAL_INTERVALS = NUM_DAYS * INTERVALS_PER_DAY
C_MAX = 10.0 
P_MAX = 10.0 
SOC_START = 5.0 
MIN_TRANZACTIE = 0.1 
SOC_MIN_MILP = 0.01
SOC_MAX_MILP = C_MAX - 0.01

# --- PARAMETRI LIGHTGBM (Cei buni) ---
LGBM_PARAMS = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'n_estimators': 800,
    'learning_rate': 0.03,
    'num_leaves': 80,
    'max_depth': 10,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5,
    'min_child_samples': 20,
    'n_jobs': -1,
    'verbose': -1,
    'seed': 42
}

# --- PARAMETRI XGBOOST (Modelul Secundar) ---
XGB_PARAMS = {
    'objective': 'reg:absoluteerror', # Echivalentul MAE
    'n_estimators': 800,
    'learning_rate': 0.03,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5,
    'n_jobs': -1,
    'random_state': 42,
    'tree_method': 'hist' # Rapid pentru seturi mari
}

# ==========================================
# 1. FEATURE ENGINEERING
# ==========================================
def create_time_features(df):
    if 'Start' not in df.columns:
        try:
            df['Start'] = pd.to_datetime(df['Time interval (CET/CEST)'].str.split(' - ').str[0], format="%d.%m.%Y %H:%M")
        except:
            df['Start'] = pd.to_datetime(df['Time interval (CET/CEST)'].str.split(' - ').str[0], dayfirst=True)

    df['hour'] = df['Start'].dt.hour
    df['minute'] = df['Start'].dt.minute
    df['dow'] = df['Start'].dt.dayofweek
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    
    # Feature-uri ciclice
    df['hour_sin'] = np.sin(2 * np.pi * (df['hour'] + df.minute/60) / 24)
    df['hour_cos'] = np.cos(2 * np.pi * (df['hour'] + df.minute/60) / 24)
    
    # Lags
    if 'Price' in df.columns:
        df['lag_96'] = df['Price'].shift(96)     # 24h
        df['lag_48'] = df['Price'].shift(48)     # 12h
        df['lag_672'] = df['Price'].shift(672)   # 1 săptămână
        
        # --- FEATURE NOU: Volatilitate (Rolling Std) ---
        # Ajută modelul să știe dacă piața e agitată
        df['rolling_std_24h'] = df['Price'].rolling(window=96).std()

    return df.dropna().reset_index(drop=True)

# ==========================================
# 2. ÎNCĂRCARE
# ==========================================
print(">>> Încărcare date...")
try:
    df_raw = pd.read_csv("Dataset.csv")
    df_raw = df_raw.sort_values(by='Time interval (CET/CEST)', key=lambda x: pd.to_datetime(x.str.split(' - ').str[0], dayfirst=True)).reset_index(drop=True)
except FileNotFoundError:
    print("EROARE: Lipsă Dataset.csv")
    exit()

df_train = create_time_features(df_raw.copy())

EXCLUDE = ['Time interval (CET/CEST)', 'Start', 'Price']
FEATURE_COLS = [c for c in df_train.columns if c not in EXCLUDE]
print(f"Features: {FEATURE_COLS}")

# ==========================================
# 3. ANTRENARE HIBRIDĂ (LGBM + XGB)
# ==========================================
X = df_train[FEATURE_COLS]
y = df_train['Price']

# Validare 14 zile
VAL_SIZE = 14 * 96 
X_train, X_val = X.iloc[:-VAL_SIZE], X.iloc[-VAL_SIZE:]
y_train, y_val = y.iloc[:-VAL_SIZE], y.iloc[-VAL_SIZE:]

print("\n>>> Antrenare Model 1: LightGBM...")
model_lgb = lgb.LGBMRegressor(**LGBM_PARAMS)
model_lgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='mae',
    callbacks=[lgb.early_stopping(100, verbose=False)]
)

print("\n>>> Antrenare Model 2: XGBoost...")
model_xgb = xgb.XGBRegressor(**XGB_PARAMS)
model_xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
print(">>> Ambele modele antrenate.")

# ==========================================
# 4. PREDICȚIE RECURSIVĂ HIBRIDĂ
# ==========================================
print("\n>>> Predicție Recursivă Hibridă...")

# Setup viitor
last_time_str = df_raw['Time interval (CET/CEST)'].str.split(' - ').str[0].iloc[-1]
try:
    last_time = pd.to_datetime(last_time_str, format="%d.%m.%Y %H:%M")
except:
    last_time = pd.to_datetime(last_time_str, dayfirst=True)

future_times = [last_time + pd.Timedelta(minutes=15 * (i + 1)) for i in range(TOTAL_INTERVALS)]
df_pred = pd.DataFrame({'Start': future_times})

# Template
df_pred_tmpl = create_time_features(df_pred.copy())
for col in ['lag_96', 'lag_48', 'lag_672', 'rolling_std_24h']:
    if col in FEATURE_COLS:
        df_pred_tmpl[col] = 0.0

X_pred_tmpl = df_pred_tmpl[FEATURE_COLS].copy()
full_history = df_train['Price'].tolist()
predicted_prices = []

for i in tqdm(range(TOTAL_INTERVALS), desc="Ensemble Prediction"):
    curr_idx = len(full_history)
    row = X_pred_tmpl.iloc[[i]].copy()
    
    # 1. Update Lags
    row.loc[row.index[0], 'lag_96'] = full_history[curr_idx - 96]
    row.loc[row.index[0], 'lag_48'] = full_history[curr_idx - 48]
    row.loc[row.index[0], 'lag_672'] = full_history[curr_idx - 672]
    
    # 2. Update Rolling Std (aproximativ, calculat pe ultimele 96 valori din istoric)
    recent_window = full_history[-96:]
    if len(recent_window) == 96:
        row.loc[row.index[0], 'rolling_std_24h'] = np.std(recent_window)
    else:
        row.loc[row.index[0], 'rolling_std_24h'] = 0.0 # Fallback

    # 3. Predicție Hibridă
    pred_lgb = model_lgb.predict(row)[0]
    pred_xgb = model_xgb.predict(row)[0]
    
    # BLENDING: 50% LGBM + 50% XGB
    # Poți ajusta ponderea (ex: 0.6 LGBM + 0.4 XGB)
    final_pred = (0.5 * pred_lgb) + (0.5 * pred_xgb)
    
    predicted_prices.append(final_pred)
    full_history.append(final_pred)

df_pred['Predicted_Price'] = predicted_prices

# ==========================================
# 5. OPTIMIZARE MILP
# ==========================================
print("\n>>> Optimizare MILP...")

def solve_milp(prices):
    T = 96
    prob = pulp.LpProblem("BatOpt", pulp.LpMaximize)
    pos = pulp.LpVariable.dicts("P", range(T), -P_MAX, P_MAX)
    soc = pulp.LpVariable.dicts("S", range(T+1), 0, C_MAX)
    chg = pulp.LpVariable.dicts("C", range(T), cat='Binary')
    dis = pulp.LpVariable.dicts("D", range(T), cat='Binary')
    surplus = pulp.LpVariable("Sur", 0, C_MAX)

    min_p = min(prices)
    # Logica profitului cu incentivare surplus
    obj = pulp.lpSum([-1 * pos[t] * prices[t] for t in range(T)]) + (surplus * (min_p - 0.01))
    prob += obj

    prob += soc[0] == SOC_START
    for t in range(T):
        prob += soc[t+1] == soc[t] + pos[t]
        prob += chg[t] + dis[t] == 1
        prob += pos[t] >= MIN_TRANZACTIE * chg[t] - P_MAX * dis[t]
        prob += pos[t] <= P_MAX * chg[t] - MIN_TRANZACTIE * dis[t]
    prob += soc[T] == surplus
    
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] == "Optimal":
        return [pulp.value(pos[t]) for t in range(T)]
    return [0.0]*T

final_actions = []
for d in range(NUM_DAYS):
    day_prices = predicted_prices[d*96 : (d+1)*96]
    final_actions.extend(solve_milp(day_prices))

final_actions = [round(x, 4) for x in final_actions]

# ==========================================
# 6. EXPORT
# ==========================================
def format_kaggle_interval(start_time):
    end_time = start_time + pd.Timedelta(minutes=15)
    fmt = "%d.%m.%Y %H:%M"
    return f"{start_time.strftime(fmt)} - {end_time.strftime(fmt)}"

if not pd.api.types.is_datetime64_any_dtype(df_pred['Start']):
    try:
        df_pred['Start'] = pd.to_datetime(df_pred['Start'], format="%d.%m.%Y %H:%M")
    except:
        df_pred['Start'] = pd.to_datetime(df_pred['Start'])

submission_df = pd.DataFrame()
submission_df['Time interval (CET/CEST)'] = df_pred['Start'].apply(format_kaggle_interval)
submission_df['Position'] = final_actions

submission_df.to_csv('submission_ensemble.csv', index=False)
print("\n✅ submission_ensemble.csv generat (LGBM + XGB).")
print(submission_df.head())