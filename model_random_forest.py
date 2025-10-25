import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold # <-- New import
from sklearn.metrics import mean_squared_error
import optuna
from optuna.samplers import TPESampler
from optuna.visualization.matplotlib import plot_optimization_history
import matplotlib.pyplot as plt
import warnings

# --- HIDE NON-CRITICAL WARNINGS ---
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=Warning)


# --- PREPROCESSING FUNCTION ---
def preprocess_data(df, is_train=True, train_stats=None):
    """
    Your winning v3 preprocessing function.
    """
    
    # === 1. Handle Missingness ===
    df['Supplier_Reliability_is_Missing'] = df['Supplier_Reliability'].isna().astype(int)
    
    if is_train:
        stats = {
            'Supplier_Reliability_fill': -1, 
            'Equipment_Height_median': df['Equipment_Height'].median(),
            'Equipment_Width_median': df['Equipment_Width'].median(),
            'Equipment_Weight_median': df['Equipment_Weight'].median(),
            'Equipment_Value_median': df['Equipment_Value'].median()
        }
    else:
        stats = train_stats

    for k, v in stats.items():
        if pd.isna(v): stats[k] = 0

    df['Supplier_Reliability'] = df['Supplier_Reliability'].fillna(stats['Supplier_Reliability_fill'])
    df['Equipment_Height'] = df['Equipment_Height'].fillna(stats['Equipment_Height_median'])
    df['Equipment_Width'] = df['Equipment_Width'].fillna(stats['Equipment_Width_median'])
    df['Equipment_Weight'] = df['Equipment_Weight'].fillna(stats['Equipment_Weight_median'])
    df['Equipment_Value'] = df['Equipment_Value'].fillna(stats['Equipment_Value_median'])
    
    df['Equipment_Type'] = df['Equipment_Type'].fillna('Unknown')
    df['Transport_Method'] = df['Transport_Method'].fillna('Unknown')
    df['Rural_Hospital'] = df['Rural_Hospital'].fillna('Unknown')

    # === 2. Date Feature Engineering ===
    df['Order_Placed_Date'] = pd.to_datetime(df['Order_Placed_Date'], errors='coerce')
    df['Delivery_Date'] = pd.to_datetime(df['Delivery_Date'], errors='coerce')
    df['Delivery_Time_Days'] = (df['Delivery_Date'] - df['Order_Placed_Date']).dt.days
    df['Delivery_Time_Days'] = df['Delivery_Time_Days'].apply(lambda x: 0 if (x < 0 or pd.isna(x)) else x)
    df['Order_Year'] = df['Order_Placed_Date'].dt.year
    df['Order_Month'] = df['Order_Placed_Date'].dt.month
    df['Order_Day_of_Week'] = df['Order_Placed_Date'].dt.dayofweek
    df['Order_Day_of_Year'] = df['Order_Placed_Date'].dt.dayofyear
    df['Order_Quarter'] = df['Order_Placed_Date'].dt.quarter
    df['is_month_start'] = (df['Order_Placed_Date'].dt.day.isin([1, 2])).astype(int)
    df['is_month_end'] = (df['Order_Placed_Date'].dt.is_month_end).astype(int)

    # Fill date NaNs
    if is_train:
        stats['Order_Year_mode'] = df['Order_Year'].mode()[0] if not df['Order_Year'].mode().empty else 2020
        stats['Order_Month_mode'] = df['Order_Month'].mode()[0] if not df['Order_Month'].mode().empty else 1
        stats['Order_Day_of_Week_mode'] = df['Order_Day_of_Week'].mode()[0] if not df['Order_Day_of_Week'].mode().empty else 0
        stats['Order_Day_of_Year_mode'] = df['Order_Day_of_Year'].mode()[0] if not df['Order_Day_of_Year'].mode().empty else 1
        stats['Order_Quarter_mode'] = df['Order_Quarter'].mode()[0] if not df['Order_Quarter'].mode().empty else 1
    
    df['Order_Year'] = df['Order_Year'].fillna(stats['Order_Year_mode'])
    df['Order_Month'] = df['Order_Month'].fillna(stats['Order_Month_mode'])
    df['Order_Day_of_Week'] = df['Order_Day_of_Week'].fillna(stats['Order_Day_of_Week_mode'])
    df['Order_Day_of_Year'] = df['Order_Day_of_Year'].fillna(stats['Order_Day_of_Year_mode'])
    df['Order_Quarter'] = df['Order_Quarter'].fillna(stats['Order_Quarter_mode'])

    # === 3. Physical & Value Feature Engineering ===
    df['Equipment_Area'] = df['Equipment_Height'] * df['Equipment_Width']
    df['Value_per_Weight'] = df['Equipment_Value'] / (df['Equipment_Weight'] + 1e-6)
    df['Equipment_Density'] = df['Equipment_Weight'] / (df['Equipment_Area'] + 1e-6)
    df['Value_per_Area'] = df['Equipment_Value'] / (df['Equipment_Area'] + 1e-6)

    # === 4. Encoding (Binaries and Dummies) ===
    binary_cols = ['CrossBorder_Shipping', 'Urgent_Shipping', 'Installation_Service', 
                   'Fragile_Equipment', 'Rural_Hospital']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(0) 

    if 'Hospital_Info' in df.columns:
        df['Hospital_Info'] = df['Hospital_Info'].map({'Wealthy': 1, 'Working Class': 0}).fillna(0)
        
    df = pd.get_dummies(df, columns=['Equipment_Type', 'Transport_Method'], drop_first=True)
    
    # === 5. NEW Interaction Feature Engineering ===
    df['Urgent_x_CrossBorder'] = df['Urgent_Shipping'] * df['CrossBorder_Shipping']
    df['Fragile_x_Rural'] = df['Fragile_Equipment'] * df['Rural_Hospital']
    df['Fragile_x_Urgent'] = df['Fragile_Equipment'] * df['Urgent_Shipping']
    
    # === 6. Drop Useless Columns ===
    cols_to_drop = ['Hospital_Id', 'Supplier_Name', 'Hospital_Location', 
                    'Order_Placed_Date', 'Delivery_Date']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    if is_train:
        return df, stats
    else:
        return df

# --- 1. LOAD AND PREPROCESS DATA ---
print("Loading and preprocessing data with ADVANCED (v3) features...")
df_train_orig = pd.read_csv("train.csv")
df_train_clean, train_stats = preprocess_data(df_train_orig, is_train=True)

# --- 2. PREPARE FULL TRAINING DATA ---
y_full = df_train_clean['Transport_Cost'].clip(lower=0)
y_full_log = np.log1p(y_full) # Log-transformed target
X_full = df_train_clean.drop(columns=['Transport_Cost'])

# --- 3. K-FOLD VALIDATION ---
print("Using K-Fold Cross-Validation for hyperparameter tuning...")

# --- 4. OPTUNA HYPERPARAMETER TUNING (WITH K-FOLD) ---
def objective(trial):
    """Define the objective function for Optuna to optimize."""
    params = {
        # --- ADJUSTED FOCUSED SEARCH SPACE ---
        # Based on your best params: {'n_estimators': 623, 'max_depth': 25, 'min_samples_split': 4, 'min_samples_leaf': 2, 'max_features': 0.98...}
        'n_estimators': trial.suggest_int('n_estimators', 400, 800),       # <-- ADJUSTED: Centered around 623
        'max_depth': trial.suggest_int('max_depth', 15, 30),              # Centered around 25 (Good)
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 8), # <-- ADJUSTED: Centered around 4
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),    # <-- ADJUSTED: Centered around 2
        'max_features': trial.suggest_float('max_features', 0.8, 1.0),      # Focus on high values (Good)
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = RandomForestRegressor(**params)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmses = []

    for train_index, val_index in kf.split(X_full):
        X_train_k, X_val_k = X_full.iloc[train_index], X_full.iloc[val_index]
        y_train_k = y_full_log.iloc[train_index]
        y_val_k_dollar = y_full.iloc[val_index]
        
        model.fit(X_train_k, y_train_k)
        
        preds_log = model.predict(X_val_k)
        preds_dollar = np.expm1(preds_log)
        
        rmse = np.sqrt(mean_squared_error(y_val_k_dollar, preds_dollar))
        rmses.append(rmse)
    
    return np.mean(rmses)

print("\nStarting FINE-TUNED hyperparameter search (300 trials, 5-Fold CV)...")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("!!! WARNING: This will take some time to run. !!!")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

sampler = TPESampler(seed=42)
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=300)

best_params = study.best_params
print("\n--- OPTUNA FINE-TUNING COMPLETE ---")
print(f"Best 5-Fold CV RMSE found: ${study.best_value:,.2f}")
print("Best parameters found:")
print(best_params)
print("----------------------------\n")

# --- 5. VISUALIZE THE OPTIMIZATION HISTORY WITH MATPLOTLIB ---
print("Displaying optimization history plot...")
plot_optimization_history(study)
plt.show()

# --- 6. PREPARE FINAL TEST DATA FOR SUBMISSION ---
print("\nLoading and preprocessing test.csv for submission...")
df_test_orig = pd.read_csv("test.csv")
submission_ids = df_test_orig['Hospital_Id'].copy()
df_test_clean = preprocess_data(df_test_orig, is_train=False, train_stats=train_stats)

X_test_final = df_test_clean.reindex(columns=X_full.columns, fill_value=0)

# --- 7. RE-TRAIN MODEL ON 100% OF DATA WITH BEST PARAMS ---
print("Re-training Random Forest model on 100% of data using best parameters...")
y_full_log = np.log1p(y_full)
final_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
final_model.fit(X_full, y_full_log) 

# --- 8. MAKE FINAL PREDICTIONS ---
print("Making final predictions on test.csv...")
preds_log = final_model.predict(X_test_final)
preds_dollar = np.expm1(preds_log)
preds_dollar = np.maximum(preds_dollar, 0)

# --- 9. CREATE SUBMISSION FILE ---
submission = pd.DataFrame({'Hospital_Id': submission_ids, 'Transport_Cost': preds_dollar})
submission.to_csv("submission_rf_v3_fine_tuned.csv", index=False)

print(f"\n✓ submission_rf_v3_fine_tuned.csv created successfully!")
print("First few predictions:")
print(submission.head())

