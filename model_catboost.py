import pandas as pd
import numpy as np
from catboost import CatBoostRegressor # <-- New import
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import optuna
from optuna.samplers import TPESampler
from optuna.visualization.matplotlib import plot_optimization_history
import matplotlib.pyplot as plt

# --- PREPROCESSING FUNCTION ---
def preprocess_data(df, is_train=True, train_stats=None):
    
    # 1. Fill Missing Numerical
    if is_train:
        stats = {
            'Supplier_Reliability_median': df['Supplier_Reliability'].median(),
            'Equipment_Height_median': df['Equipment_Height'].median(),
            'Equipment_Width_median': df['Equipment_Width'].median(),
            'Equipment_Weight_median': df['Equipment_Weight'].median(),
            'Equipment_Value_median': df['Equipment_Value'].median()
        }
    else:
        stats = train_stats

    # Handle potential NaNs in stats dictionary
    for k, v in stats.items():
        if pd.isna(v):
            stats[k] = 0

    df['Supplier_Reliability'] = df['Supplier_Reliability'].fillna(stats['Supplier_Reliability_median'])
    df['Equipment_Height'] = df['Equipment_Height'].fillna(stats['Equipment_Height_median'])
    df['Equipment_Width'] = df['Equipment_Width'].fillna(stats['Equipment_Width_median'])
    df['Equipment_Weight'] = df['Equipment_Weight'].fillna(stats['Equipment_Weight_median'])
    df['Equipment_Value'] = df['Equipment_Value'].fillna(stats['Equipment_Value_median'])
    
    # 2. Fill Missing Categorical
    if is_train:
        # Handle potential empty mode
        stats['Equipment_Type_mode'] = df['Equipment_Type'].mode()[0] if not df['Equipment_Type'].mode().empty else 'Type A'
        stats['Transport_Method_mode'] = df['Transport_Method'].mode()[0] if not df['Transport_Method'].mode().empty else 'Road'
        stats['Rural_Hospital_mode'] = df['Rural_Hospital'].mode()[0] if not df['Rural_Hospital'].mode().empty else 'No'
        
    df['Equipment_Type'] = df['Equipment_Type'].fillna(stats['Equipment_Type_mode'])
    df['Transport_Method'] = df['Transport_Method'].fillna(stats['Transport_Method_mode'])
    df['Rural_Hospital'] = df['Rural_Hospital'].fillna(stats['Rural_Hospital_mode'])
    
    # 3. Feature Engineering (Dates) 
    df['Order_Placed_Date'] = pd.to_datetime(df['Order_Placed_Date'], errors='coerce')
    df['Delivery_Date'] = pd.to_datetime(df['Delivery_Date'], errors='coerce')
    df['Delivery_Time_Days'] = (df['Delivery_Date'] - df['Order_Placed_Date']).dt.days
    df['Delivery_Time_Days'] = df['Delivery_Time_Days'].apply(lambda x: 0 if (x < 0 or pd.isna(x)) else x)
    df['Order_Year'] = df['Order_Placed_Date'].dt.year
    df['Order_Month'] = df['Order_Placed_Date'].dt.month
    df['Order_Day_of_Week'] = df['Order_Placed_Date'].dt.dayofweek
    
    if is_train:
        stats['Order_Year_mode'] = df['Order_Year'].mode()[0] if not df['Order_Year'].mode().empty else 2020
        stats['Order_Month_mode'] = df['Order_Month'].mode()[0] if not df['Order_Month'].mode().empty else 1
        stats['Order_Day_of_Week_mode'] = df['Order_Day_of_Week'].mode()[0] if not df['Order_Day_of_Week'].mode().empty else 0
    
    df['Order_Year'] = df['Order_Year'].fillna(stats['Order_Year_mode'])
    df['Order_Month'] = df['Order_Month'].fillna(stats['Order_Month_mode'])
    df['Order_Day_of_Week'] = df['Order_Day_of_Week'].fillna(stats['Order_Day_of_Week_mode'])
    
    # 4. Additional Feature Engineering 
    df['Value_per_Weight'] = df['Equipment_Value'] / (df['Equipment_Weight'] + 1)
    df['Equipment_Area'] = df['Equipment_Height'] * df['Equipment_Width']
    df['Is_Weekend'] = df['Order_Day_of_Week'].isin([5, 6]).astype(int)

    # 5. Encoding (Binaries) (
    binary_cols = ['CrossBorder_Shipping', 'Urgent_Shipping', 'Installation_Service', 
                   'Fragile_Equipment', 'Rural_Hospital']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(0)

    if 'Hospital_Info' in df.columns:
        df['Hospital_Info'] = df['Hospital_Info'].map({'Wealthy': 1, 'Working Class': 0}).fillna(0)
        
    # --- THIS IS THE KEY CHANGE ---
    # We NO LONGER one-hot encode. CatBoost will handle these columns.
    # df = pd.get_dummies(df, columns=['Equipment_Type', 'Transport_Method'], drop_first=True)
    
    # 6. Drop Useless Columns 
    cols_to_drop = ['Hospital_Id', 'Supplier_Name', 'Hospital_Location', 
                    'Order_Placed_Date', 'Delivery_Date']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    if is_train:
        return df, stats
    else:
        return df

# --- 1. LOAD AND PREPROCESS DATA ---
print("Loading and preprocessing data...")
df_train_orig = pd.read_csv("train.csv")
df_train_clean, train_stats = preprocess_data(df_train_orig, is_train=True)

# --- 2. PREPARE FULL TRAINING DATA ---
y_full = df_train_clean['Transport_Cost'].clip(lower=0)
X_full = df_train_clean.drop(columns=['Transport_Cost'])

# --- 3. CREATE VALIDATION SET FOR TUNING ---
print("Creating validation set for hyperparameter tuning...")
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)
y_train_split_log = np.log1p(y_train_split)

# --- 4. OPTUNA HYPERPARAMETER TUNING FOR CATBOOST ---

# Define the list of categorical feature names
categorical_features = ['Equipment_Type', 'Transport_Method']

def objective(trial):
    """Define the objective function for Optuna to optimize for CatBoost."""
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 100.0, log=True),
        'random_seed': 42,
        'verbose': False # Suppress training output
    }
    
    model = CatBoostRegressor(**params)
    
    model.fit(X_train_split, y_train_split_log, 
              cat_features=categorical_features # <-- Telling CatBoost which columns are categorical
             )
    
    preds_log = model.predict(X_val)
    preds_dollar = np.expm1(preds_log)
    
    rmse = np.sqrt(mean_squared_error(y_val, preds_dollar))
    return rmse

print("\nStarting hyperparameter tuning for CatBoost with Optuna...")
# Create a sampler with a fixed seed for reproducibility
sampler = TPESampler(seed=42)
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=150)

best_params = study.best_params
print("\n--- OPTUNA TUNING COMPLETE ---")
print(f"Best RMSE found: ${study.best_value:,.2f}")
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
print("Re-training CatBoost model on 100% of data using best parameters...")
y_full_log = np.log1p(y_full)
final_model = CatBoostRegressor(**best_params, random_seed=42, verbose=False) # Use best params

final_model.fit(X_full, y_full_log, 
                cat_features=categorical_features # <-- Tell CatBoost which columns are categorical
               )

# --- 8. MAKE FINAL PREDICTIONS ---
print("Making final predictions on test.csv...")
preds_log = final_model.predict(X_test_final)
preds_dollar = np.expm1(preds_log)
preds_dollar = np.maximum(preds_dollar, 0)

# --- 9. CREATE SUBMISSION FILE ---
submission = pd.DataFrame({'Hospital_Id': submission_ids, 'Transport_Cost': preds_dollar})
submission.to_csv("submission_catboost_optuna.csv", index=False)

print("\n✓ submission_catboost_optuna.csv created successfully!")
print("First few predictions:")
print(submission.head())
