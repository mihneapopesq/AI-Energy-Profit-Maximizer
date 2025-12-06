import pandas as pd
import numpy as np
import lightgbm as lgb
import pulp
import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ==============================================================================
# 0. CONSTANTE ȘI PARAMETRI
# ==============================================================================

NUM_DAYS = 8
INTERVALS_PER_HOUR = 4
INTERVALS_PER_DAY = 24 * INTERVALS_PER_HOUR # 96
TOTAL_INTERVALS = NUM_DAYS * INTERVALS_PER_DAY # 768

C_MAX = 10.0 
P_MAX = 10.0 
SOC_START = 5.0 
MIN_TRANZACTIE = 0.1 

# Marjă de siguranță pentru capacitatea MILP
SOC_MIN_MILP = 0.01
SOC_MAX_MILP = C_MAX - 0.01


# Parametri LightGBM ajustați pentru performanță maximă
N_ESTIMATORS = 300 # Crescut pentru learning rate mic
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

# ==============================================================================
# 1. PREGATIREA DATELOR ȘI FEATURE ENGINEERING (Noul Lag 48)
# ==============================================================================

def create_time_features(df):
    
    if 'Start' not in df.columns:
        df['Start'] = pd.to_datetime(df['Time interval (CET/CEST)'].str.split(' - ').str[0], format="%d.%m.%Y %H:%M")

    # Caracteristici de Timp
    df['hour'] = df['Start'].dt.hour
    df['minute'] = df['Start'].dt.minute
    df['dow'] = df['Start'].dt.dayofweek
    df['dom'] = df['Start'].dt.day
    df['month'] = df['Start'].dt.month
    df['year'] = df['Start'].dt.year
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * (df['hour'] + df.minute/60) / 24)
    df['hour_cos'] = np.cos(2 * np.pi * (df['hour'] + df.minute/60) / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
    
    # Lag-uri de preț (Lag 48 adăugat)
    if 'Price' in df.columns:
        df['price_lag_96'] = df['Price'].shift(96)     # 1 zi
        df['price_lag_48'] = df['Price'].shift(48)     # 12 ore
        df['price_lag_672'] = df['Price'].shift(672)   # 7 zile

    return df.dropna().reset_index(drop=True)

# 🛑 AICI ÎNCĂRCĂM DATELE TALE
try:
    df_raw = pd.read_csv("Dataset.csv")
    df_raw = df_raw.sort_values(by='Time interval (CET/CEST)', key=lambda x: pd.to_datetime(x.str.split(' - ').str[0], format="%d.%m.%Y %H:%M")).reset_index(drop=True)
    
    df_train = create_time_features(df_raw.copy())
    
    EXCLUDE_COLS = ['Time interval (CET/CEST)', 'Start', 'Price']
    FEATURE_COLS = [col for col in df_train.columns if col not in EXCLUDE_COLS]
    
    # MAX_LAG devine 672
    MAX_LAG = 672 

    print("1. Date istorice încărcate și preprocesate.")
    print(f"   Număr de rânduri de antrenare: {len(df_train)}")

except FileNotFoundError:
    print("EROARE: Fișierul 'Dataset.csv' nu a fost găsit. Nu se poate continua.")
    exit()

# ==============================================================================
# 2. ANTRENAMENT LIGHTGBM ȘI PREDICTIE ITERATIVĂ
# ==============================================================================

X = df_train[FEATURE_COLS]
y = df_train['Price']

X_train, X_val = X.iloc[:-96], X.iloc[-96:]
y_train, y_val = y.iloc[:-96], y.iloc[-96:]

# 1. Antrenarea Modelului
print("\n2. Antrenarea LightGBM (Final Fine-Tuning)...")
lgbm = lgb.LGBMRegressor(**LGBM_PARAMS)

lgbm.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='mae',
    callbacks=[lgb.early_stopping(50, verbose=False)]
)
print(f"   Antrenare finalizată. Număr de iteratii: {lgbm.best_iteration_}")


# 2. Pregătirea datelor pentru predicție
last_time_str = df_raw['Time interval (CET/CEST)'].str.split(' - ').str[0].iloc[-1]
last_time = pd.to_datetime(last_time_str, format="%d.%m.%Y %H:%M")

future_timestamps = [last_time + pd.Timedelta(minutes=15 * (i + 1)) for i in range(TOTAL_INTERVALS)]

df_pred = pd.DataFrame({'Start': future_timestamps})
df_pred['Time interval (CET/CEST)'] = df_pred['Start'].apply(lambda x: x.strftime("%d.%m.%Y %H:%M"))

df_pred_template = create_time_features(df_pred.copy())

# Reinițializăm coloanele de lag
for col in ['price_lag_96', 'price_lag_48', 'price_lag_672']:
    if col in FEATURE_COLS:
        df_pred_template[col] = 0.0

X_pred_template = df_pred_template[FEATURE_COLS].copy() 


# 3. Predicția (Iterativ)
print("3. Predictie iterativă (8 zile)...")

full_history = df_train['Price'].iloc[-MAX_LAG:].tolist() 
predicted_prices = []

for i in tqdm(range(TOTAL_INTERVALS), desc="Predicting"):
    current_idx = len(full_history) - 1
    
    X_pred_row = X_pred_template.iloc[[i]].copy()
    
    # 24h Lag (96 de pași în urmă)
    lag_96_val = full_history[current_idx - 95] 
    X_pred_row.loc[X_pred_row.index[0], 'price_lag_96'] = lag_96_val
    
    # 12h Lag (48 de pași în urmă)
    lag_48_val = full_history[current_idx - 47] 
    X_pred_row.loc[X_pred_row.index[0], 'price_lag_48'] = lag_48_val

    # 7 Day Lag (672 de pași în urmă)
    lag_672_val = full_history[current_idx - 671] 
    X_pred_row.loc[X_pred_row.index[0], 'price_lag_672'] = lag_672_val
    
    # Realizăm predicția
    pred_val = lgbm.predict(X_pred_row)[0]
    predicted_prices.append(pred_val)
    
    # Actualizăm istoricul
    full_history.append(pred_val)

df_pred['Predicted_Price'] = predicted_prices
print("   Predictie finalizată ✓")

# ==============================================================================
# 4. OPTIMIZARE LINIARA CU NUMERE INTREGI MIXTE (MILP)
# ==============================================================================

def solve_daily_milp(day_prices):
    T = INTERVALS_PER_DAY
    prob = pulp.LpProblem("Battery_Trading_Optimization_MILP", pulp.LpMaximize)
    
    # Variabile de decizie (SoC limite ajustate)
    position = pulp.LpVariable.dicts("Position", range(1, T + 1), lowBound=-P_MAX, upBound=P_MAX)
    soc = pulp.LpVariable.dicts("SoC", range(1, T + 1), lowBound=SOC_MIN_MILP, upBound=SOC_MAX_MILP)
    surplus = pulp.LpVariable("Surplus", lowBound=0, upBound=C_MAX)
    
    charge = pulp.LpVariable.dicts("Charge", range(1, T + 1), cat='Binary')
    discharge = pulp.LpVariable.dicts("Discharge", range(1, T + 1), cat='Binary')

    pret_minim_zi = min(day_prices)

    # Functia Obiectiv
    cost_tranzactii = pulp.lpSum([position[t] * day_prices[t-1] for t in range(1, T + 1)])
    venit_surplus = surplus * pret_minim_zi
    
    prob += venit_surplus - cost_tranzactii, "Profit_Total"

    # Constrangeri Baterie & Logica
    prob += soc[1] == SOC_START + position[1], "SoC_init"
    for t in range(2, T + 1):
        prob += soc[t] == soc[t-1] + position[t], f"SoC_t_{t}"
    prob += soc[T] == surplus, "SoC_Final_Surplus"

    # Constrangeri MILP pentru Tranzactie Non-Zero si Minima
    for t in range(1, T + 1):
        prob += charge[t] + discharge[t] == 1, f"Action_Mandatory_{t}"
        prob += position[t] >= MIN_TRANZACTIE * charge[t] - P_MAX * discharge[t], f"Position_Lower_{t}"
        prob += position[t] <= P_MAX * charge[t] - MIN_TRANZACTIE * discharge[t], f"Position_Upper_{t}"

    prob.solve(pulp.PULP_CBC_CMD(msg=0)) 
    
    if pulp.LpStatus[prob.status] == "Optimal":
        actions = [position[t].varValue if position[t].varValue is not None else 0.0 for t in range(1, T + 1)]
        return actions
    else:
        print(f"   Avertisment: Optimizare nereușită. Status: {pulp.LpStatus[prob.status]}. Folosind fallback.")
        return [MIN_TRANZACTIE] * T

# Rularea MILP pe cele 8 Zile
all_actions = []
print("\n4. Optimizarea MILP a acțiunilor (pe zi)...")

for day in tqdm(range(NUM_DAYS), desc="MILP Solver"):
    start_idx = day * INTERVALS_PER_DAY
    end_idx = (day + 1) * INTERVALS_PER_DAY
    day_prices = df_pred['Predicted_Price'][start_idx:end_idx].tolist()
    
    actions_day = solve_daily_milp(day_prices)
    all_actions.extend(actions_day)

print("   Optimizarea finalizată. Total acțiuni: ", len(all_actions))

# ==============================================================================
# 5. EXPORTUL FINAL
# ==============================================================================

final_actions = [round(action, 4) if action is not None else 0.0 for action in all_actions]

def format_time_interval_strict(timestamp):
    """Genereaza formatul de interval cerut: DD.MM.YYYY HH:MM - DD.MM.YYYY HH:MM"""
    start_time = timestamp
    end_time = timestamp + pd.Timedelta(minutes=15)
    
    fmt = '%d.%m.%Y %H:%M'
    return f"{start_time.strftime(fmt)} - {end_time.strftime(fmt)}"

submission_df = pd.DataFrame()
submission_df['Time interval (CET/CEST)'] = df_pred['Start'].apply(format_time_interval_strict)
submission_df['Position'] = final_actions

# Export
submission_df.to_csv('submission.csv', index=False)
print("\n5. Fișierul 'submission.csv' a fost generat.")
print("\n--- Primele 5 acțiuni (Formatul final) ---")
print(submission_df.head())