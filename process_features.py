import pandas as pd
import numpy as np
import time
import os

print("Starting Feature Engineering pipeline...")
start_time = time.time()

def find_data_file(filename):
    if os.path.exists(filename):
        return filename
    csvs_path = os.path.join("csvs", filename)
    if os.path.exists(csvs_path):
        return csvs_path
    raise FileNotFoundError(f"Could not find {filename} in current directory or in 'csvs/'")

# 1. Load installments_payments.csv
print("Loading installments_payments.csv...")
df = pd.read_csv(find_data_file("installments_payments.csv"))
print(f"Loaded {len(df)} rows in {time.time() - start_time:.2f} seconds.")

# 2. Collapse split payments to installment level (SK_ID_PREV + NUM_INSTALMENT_NUMBER)
# To handle duplicates correctly without double counting AMT_INSTALMENT
print("Collapsing split payments to installment level...")
collapse_start = time.time()
installments_collapsed = df.groupby(['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_NUMBER']).agg({
    'NUM_INSTALMENT_VERSION': 'first',
    'DAYS_INSTALMENT': 'first',
    'DAYS_ENTRY_PAYMENT': 'max',  # Date of the last partial payment
    'AMT_INSTALMENT': 'first',     # The expected amount is the same across duplicates
    'AMT_PAYMENT': 'sum'          # Sum of all partial payments
}).reset_index()
print(f"Collapsed to {len(installments_collapsed)} installments in {time.time() - collapse_start:.2f} seconds.")

# 3. Calculate metrics at the installment level
print("Calculating installment-level metrics...")
# DPD (Days Past Due): positive means late, negative means early
installments_collapsed['DPD'] = installments_collapsed['DAYS_ENTRY_PAYMENT'] - installments_collapsed['DAYS_INSTALMENT']
installments_collapsed['LATE_DAYS'] = installments_collapsed['DPD'].clip(lower=0)
installments_collapsed['EARLY_DAYS'] = (-installments_collapsed['DPD']).clip(lower=0)

# Underpayment
installments_collapsed['UNDERPAYMENT'] = (installments_collapsed['AMT_INSTALMENT'] - installments_collapsed['AMT_PAYMENT']).clip(lower=0)
installments_collapsed['IS_LATE'] = (installments_collapsed['DPD'] > 0).astype(int)
installments_collapsed['IS_UNDERPAID'] = (installments_collapsed['AMT_PAYMENT'] < installments_collapsed['AMT_INSTALMENT']).astype(int)

# Handle cases where DAYS_ENTRY_PAYMENT is null (missing payment)
# In these cases, DPD and LATE_DAYS should represent a default/late state or stay null. 
# We'll let tree-based models handle the nulls, but it's good to set flag:
installments_collapsed['IS_MISSING_PAYMENT'] = installments_collapsed['DAYS_ENTRY_PAYMENT'].isnull().astype(int)

# 4. Generate aggregates at the client level (SK_ID_CURR)
print("Creating client-level aggregations...")
agg_start = time.time()

# General aggregations
features = installments_collapsed.groupby('SK_ID_CURR').agg({
    'NUM_INSTALMENT_NUMBER': 'count', # Total historical installments
    'NUM_INSTALMENT_VERSION': 'nunique', # Number of calendar revisions
    'DPD': ['mean', 'max', 'std'],
    'LATE_DAYS': ['mean', 'max', 'sum'],
    'EARLY_DAYS': ['mean', 'max', 'std'],
    'UNDERPAYMENT': ['mean', 'max', 'sum'],
    'IS_LATE': ['mean', 'sum'],
    'IS_UNDERPAID': ['mean', 'sum'],
    'IS_MISSING_PAYMENT': ['mean', 'sum']
})

# Flatten the multi-index columns
features.columns = ['_'.join(col).strip() for col in features.columns.values]
features = features.reset_index()
features.rename(columns={'NUM_INSTALMENT_NUMBER_count': 'INSTALMENT_COUNT'}, inplace=True)

# Time-windowed aggregations
print("Creating time-windowed aggregations...")
# Last 3 months (90 days), Last 6 months (180 days), Last 12 months (360 days)
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

print(f"Features created. Shape: {features.shape}. Time: {time.time() - agg_start:.2f} seconds.")

# 5. Export results to a new folder
out_dir = "processed_data"
print(f"Creating output directory '{out_dir}' if it doesn't exist...")
os.makedirs(out_dir, exist_ok=True)

# Save standalone client features
features_path = os.path.join(out_dir, "installments_features.csv")
print(f"Saving client-level features to {features_path}...")
features.to_csv(features_path, index=False)
print("Saved installments_features.csv")

# Load and join with application_train
print("Loading application_train.csv...")
app_train = pd.read_csv(find_data_file("application_train.csv"))
print(f"Loaded train. Shape: {app_train.shape}")

print("Performing Left Join with application_train...")
app_train_merged = pd.merge(app_train, features, on='SK_ID_CURR', how='left')
train_merged_path = os.path.join(out_dir, "application_train_merged.csv")
print(f"Saving merged train data to {train_merged_path}...")
app_train_merged.to_csv(train_merged_path, index=False)
print("Saved application_train_merged.csv")

# Load and join with application_test (if it exists)
try:
    test_path = find_data_file("application_test.csv")
    print(f"Loading {test_path}...")
    app_test = pd.read_csv(test_path)
    print(f"Loaded test. Shape: {app_test.shape}")
    print("Performing Left Join with application_test...")
    app_test_merged = pd.merge(app_test, features, on='SK_ID_CURR', how='left')
    test_merged_path = os.path.join(out_dir, "application_test_merged.csv")
    print(f"Saving merged test data to {test_merged_path}...")
    app_test_merged.to_csv(test_merged_path, index=False)
    print("Saved application_test_merged.csv")
except FileNotFoundError:
    print("application_test.csv not found, skipping merge with test data.")

# Verification of merged data
print("\n--- Verification of merged data ---")
check_cols = ['INSTALMENT_COUNT', 'DPD_mean', 'DPD_max', 'LATE_DAYS_mean_LAST_90D', 'UNDERPAYMENT_sum_LAST_90D']
for col in check_cols:
    if col in app_train_merged.columns:
        null_cnt = app_train_merged[col].isnull().sum()
        pct = 100 * null_cnt / len(app_train_merged)
        print(f"Column '{col}': {null_cnt} nulls ({pct:.2f}%)")

print(f"Total time elapsed: {time.time() - start_time:.2f} seconds.")
