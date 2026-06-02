import pandas as pd
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb

print("Starting baseline model training pipeline...")
start_time = time.time()

def find_data_file(filename):
    if os.path.exists(filename):
        return filename
    csvs_path = os.path.join("csvs", filename)
    if os.path.exists(csvs_path):
        return csvs_path
    raise FileNotFoundError(f"Could not find {filename} in current directory or in 'csvs/'")

# 1. Load installments_payments.csv & Engineer features
print("Step 1: Loading and engineering features...")
features_path = os.path.join("processed_data", "installments_features.csv")
if os.path.exists(features_path):
    print(f"Loading pre-calculated features from {features_path}...")
    features = pd.read_csv(features_path)
else:
    print("Pre-calculated features not found. Engineering from scratch...")
    df = pd.read_csv(find_data_file("installments_payments.csv"))
    installments_collapsed = df.groupby(['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_NUMBER']).agg({
        'NUM_INSTALMENT_VERSION': 'first',
        'DAYS_INSTALMENT': 'first',
        'DAYS_ENTRY_PAYMENT': 'max',
        'AMT_INSTALMENT': 'first',
        'AMT_PAYMENT': 'sum'
    }).reset_index()

    installments_collapsed['DPD'] = installments_collapsed['DAYS_ENTRY_PAYMENT'] - installments_collapsed['DAYS_INSTALMENT']
    installments_collapsed['LATE_DAYS'] = installments_collapsed['DPD'].clip(lower=0)
    installments_collapsed['EARLY_DAYS'] = (-installments_collapsed['DPD']).clip(lower=0)
    installments_collapsed['UNDERPAYMENT'] = (installments_collapsed['AMT_INSTALMENT'] - installments_collapsed['AMT_PAYMENT']).clip(lower=0)
    installments_collapsed['IS_LATE'] = (installments_collapsed['DPD'] > 0).astype(int)
    installments_collapsed['IS_UNDERPAID'] = (installments_collapsed['AMT_PAYMENT'] < installments_collapsed['AMT_INSTALMENT']).astype(int)
    installments_collapsed['IS_MISSING_PAYMENT'] = installments_collapsed['DAYS_ENTRY_PAYMENT'].isnull().astype(int)

    features = installments_collapsed.groupby('SK_ID_CURR').agg({
        'NUM_INSTALMENT_NUMBER': 'count',
        'NUM_INSTALMENT_VERSION': 'nunique',
        'DPD': ['mean', 'max', 'std'],
        'LATE_DAYS': ['mean', 'max', 'sum'],
        'EARLY_DAYS': ['mean', 'max', 'std'],
        'UNDERPAYMENT': ['mean', 'max', 'sum'],
        'IS_LATE': ['mean', 'sum'],
        'IS_UNDERPAID': ['mean', 'sum'],
        'IS_MISSING_PAYMENT': ['mean', 'sum']
    })
    features.columns = ['_'.join(col).strip() for col in features.columns.values]
    features = features.reset_index()
    features.rename(columns={'NUM_INSTALMENT_NUMBER_count': 'INSTALMENT_COUNT'}, inplace=True)

    # Time-windowed aggregations
    for days in [90, 180, 360]:
        window_df = installments_collapsed[installments_collapsed['DAYS_INSTALMENT'] >= -days]
        if len(window_df) > 0:
            window_agg = window_df.groupby('SK_ID_CURR').agg({
                'DPD': ['mean', 'max'],
                'LATE_DAYS': ['mean', 'max', 'sum'],
                'UNDERPAYMENT': ['mean', 'sum'],
                'IS_LATE': ['mean', 'sum'],
                'IS_UNDERPAID': ['mean', 'sum']
            })
            window_agg.columns = [f'{col[0]}_{col[1]}_LAST_{days}D' for col in window_agg.columns.values]
            window_agg = window_agg.reset_index()
            features = pd.merge(features, window_agg, on='SK_ID_CURR', how='left')

print(f"Features ready. Shape: {features.shape}. Time: {time.time() - start_time:.2f}s")

# 2. Load application_train.csv
print("Step 2: Loading application_train.csv...")
app_train = pd.read_csv(find_data_file("application_train.csv"))
app_train_merged = pd.merge(app_train, features, on='SK_ID_CURR', how='left')
print(f"Merged Train Shape: {app_train_merged.shape}")

# 3. Data Prep for LightGBM
print("Step 3: Preparing data for training...")
# Define target and ID
target_col = 'TARGET'
id_col = 'SK_ID_CURR'

# Let's drop columns that shouldn't be features
ignore_cols = [id_col, target_col]
feature_cols = [col for col in app_train_merged.columns if col not in ignore_cols]

# Handle categorical features (convert to category type for LightGBM)
categorical_cols = []
for col in feature_cols:
    if not pd.api.types.is_numeric_dtype(app_train_merged[col]) or app_train_merged[col].dtype.name == 'category':
        app_train_merged[col] = app_train_merged[col].astype('category')
        categorical_cols.append(col)

# 4. Split Train / Validation (80/20 Stratified)
X = app_train_merged[feature_cols]
y = app_train_merged[target_col]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train size: {X_train.shape[0]} | Validation size: {X_val.shape[0]}")
print(f"Target distribution in Train: {y_train.mean():.4f} | Validation: {y_val.mean():.4f}")

# 5. Train LightGBM model
print("Step 4: Training LightGBM classifier...")
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

# We use standard lightgbm parameters suited for binary classification with imbalance
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 6,
    'min_data_in_leaf': 30,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'scale_pos_weight': 11.0,  # 1:11 ratio of defaults (appx 8% target rate)
    'random_state': 42,
    'verbose': -1
}

# Run LightGBM training
evals_result = {}
model = lgb.train(
    params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train, lgb_val],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50)
    ]
)

# 6. Evaluation on Validation Set
print("\nStep 5: Evaluating model on validation set...")
y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)

roc_auc = roc_auc_score(y_val, y_pred_proba)
pr_auc = average_precision_score(y_val, y_pred_proba)

print(f"\n======================================")
print(f"Validation ROC-AUC: {roc_auc:.5f}")
print(f"Validation PR-AUC (Average Precision): {pr_auc:.5f}")
print(f"======================================")

# 7. Feature Importance
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values(by='importance', ascending=False)

print("\nTop 15 Feature Importances (by Gain):")
print(importance.head(15))

# Save the top 100 features list to a CSV file
importance.to_csv("feature_importances_baseline.csv", index=False)
print("Saved feature_importances_baseline.csv")

print(f"\nPipeline completed in {time.time() - start_time:.2f} seconds.")
