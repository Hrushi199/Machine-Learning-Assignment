import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import optuna
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

    df['Supplier_Reliability'] = df['Supplier_Reliability'].fillna(stats['Supplier_Reliability_median'])
    df['Equipment_Height'] = df['Equipment_Height'].fillna(stats['Equipment_Height_median'])
    df['Equipment_Width'] = df['Equipment_Width'].fillna(stats['Equipment_Width_median'])
    df['Equipment_Weight'] = df['Equipment_Weight'].fillna(stats['Equipment_Weight_median'])
    df['Equipment_Value'] = df['Equipment_Value'].fillna(stats['Equipment_Value_median'])
    
    # 2. Fill Missing Categorical
    if is_train:
        stats['Equipment_Type_mode'] = df['Equipment_Type'].mode()[0]
        stats['Transport_Method_mode'] = df['Transport_Method'].mode()[0]
        stats['Rural_Hospital_mode'] = df['Rural_Hospital'].mode()[0]
        
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
        stats['Order_Year_mode'] = df['Order_Year'].mode()[0]
        stats['Order_Month_mode'] = df['Order_Month'].mode()[0]
        stats['Order_Day_of_Week_mode'] = df['Order_Day_of_Week'].mode()[0]
    
    df['Order_Year'] = df['Order_Year'].fillna(stats['Order_Year_mode'])
    df['Order_Month'] = df['Order_Month'].fillna(stats['Order_Month_mode'])
    df['Order_Day_of_Week'] = df['Order_Day_of_Week'].fillna(stats['Order_Day_of_Week_mode'])
    
    # 4. Additional Feature Engineering
    df['Value_per_Weight'] = df['Equipment_Value'] / (df['Equipment_Weight'] + 1)
    df['Equipment_Area'] = df['Equipment_Height'] * df['Equipment_Width']
    df['Is_Weekend'] = df['Order_Day_of_Week'].isin([5, 6]).astype(int)

    # 5. Encoding
    binary_cols = ['CrossBorder_Shipping', 'Urgent_Shipping', 'Installation_Service', 
                   'Fragile_Equipment', 'Rural_Hospital']
    for col in binary_cols:
        if 'Yes' in df[col].unique():
            df[col] = df[col].map({'Yes': 1, 'No': 0})
        else: 
            df[col] = df[col].map({'No': 0}).fillna(0)

    if 'Wealthy' in df['Hospital_Info'].unique():
        df['Hospital_Info'] = df['Hospital_Info'].map({'Wealthy': 1, 'Working Class': 0})
    else:
        df['Hospital_Info'] = df['Hospital_Info'].map({'Working Class': 0}).fillna(0)
        
    df = pd.get_dummies(df, columns=['Equipment_Type', 'Transport_Method'], drop_first=True)
    
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

# --- 4. OPTUNA HYPERPARAMETER TUNING FOR ADABOOST ---
def objective(trial):
    """Define the objective function for Optuna to optimize for AdaBoost."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 1.0, log=True),
        'loss': trial.suggest_categorical('loss', ['linear', 'square', 'exponential']),
        'random_state': 42
    }
    
    model = AdaBoostRegressor(**params)
    model.fit(X_train_split, y_train_split_log)
    
    preds_log = model.predict(X_val)
    preds_dollar = np.expm1(preds_log)
    
    rmse = np.sqrt(mean_squared_error(y_val, preds_dollar))
    return rmse

print("\nStarting hyperparameter tuning for AdaBoost with Optuna...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=300)

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
print("Re-training AdaBoost model on 100% of data using best parameters...")
y_full_log = np.log1p(y_full)
final_model = AdaBoostRegressor(**best_params, random_state=42)
final_model.fit(X_full, y_full_log)

# --- 8. MAKE FINAL PREDICTIONS ---
print("Making final predictions on test.csv...")
preds_log = final_model.predict(X_test_final)
preds_dollar = np.expm1(preds_log)
preds_dollar = np.maximum(preds_dollar, 0)

# --- 9. CREATE SUBMISSION FILE ---
submission = pd.DataFrame({'Hospital_Id': submission_ids, 'Transport_Cost': preds_dollar})
submission.to_csv("submission_ada_hyp_300.csv", index=False)

print("\n✓ submission.csv created successfully!")
print("First few predictions:")
print(submission.head())

