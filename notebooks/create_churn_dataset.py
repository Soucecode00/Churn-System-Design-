import sys
from pathlib import Path

# Add parent directory to path (works regardless of where script is run from)
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

import pandas as pd
import numpy as np
from data_generator import raw_data
import json

# Debug logging setup
DEBUG_LOG_PATH = r'c:\Users\acer\Desktop\Churn System Design\.cursor\debug.log'

def log_debug(location, message, data, hypothesis_id):
    """Write debug log entry"""
    try:
        with open(DEBUG_LOG_PATH, 'a', encoding='utf-8') as f:
            log_entry = {
                'id': f'log_{pd.Timestamp.now().value}',
                'timestamp': pd.Timestamp.now().value // 1000000,
                'location': location,
                'message': message,
                'data': data,
                'hypothesisId': hypothesis_id
            }
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        pass  # Silent fail for logging

print("Starting churn dataset creation...")
print(f"Raw data shape: {raw_data.shape}")

# #region agent log
log_debug('create_churn_dataset.py:31', 'Raw data loaded', {
    'total_rows': len(raw_data),
    'unique_users': raw_data['user_id'].nunique(),
    'date_range': {
        'min': str(raw_data['timestamp'].min()),
        'max': str(raw_data['timestamp'].max())
    }
}, 'H1')
# #endregion

# Calculate the observation period
min_date = raw_data['timestamp'].min()
max_date = raw_data['timestamp'].max()
total_days = (max_date - min_date).days

# Define churn cutoff: last 7 days of the period
churn_window_start = max_date - pd.Timedelta(days=7)

print(f"\nObservation period: {min_date.date()} to {max_date.date()} ({total_days} days)")
print(f"Churn definition: No activity after {churn_window_start.date()}")

# #region agent log
log_debug('create_churn_dataset.py:54', 'Churn window calculated', {
    'min_date': str(min_date),
    'max_date': str(max_date),
    'churn_window_start': str(churn_window_start),
    'total_days': total_days
}, 'H2')
# #endregion

# Aggregate user-level features
print("\nAggregating user-level features...")

# Group by user
user_features = raw_data.groupby('user_id').agg({
    'timestamp': ['min', 'max', 'count'],
    'session_duration_sec': ['mean', 'sum', 'std'],
    'pages_viewed': ['mean', 'sum'],
    'purchase_amount': ['sum', 'count']
}).reset_index()

# Flatten column names
user_features.columns = ['user_id', 'first_session', 'last_session', 'total_sessions',
                          'avg_session_duration', 'total_session_duration', 'std_session_duration',
                          'avg_pages_viewed', 'total_pages_viewed',
                          'total_purchase_amount', 'purchase_count']

# #region agent log
log_debug('create_churn_dataset.py:80', 'User features aggregated', {
    'total_users': len(user_features),
    'sample_user': {
        'user_id': int(user_features.iloc[0]['user_id']),
        'total_sessions': int(user_features.iloc[0]['total_sessions']),
        'last_session': str(user_features.iloc[0]['last_session'])
    }
}, 'H3')
# #endregion

# Calculate additional features
user_features['days_since_first_session'] = (max_date - user_features['first_session']).dt.days
user_features['days_since_last_activity'] = (max_date - user_features['last_session']).dt.days
user_features['purchase_frequency'] = user_features['purchase_count'] / user_features['total_sessions']
user_features['avg_purchase_value'] = user_features['total_purchase_amount'] / user_features['purchase_count'].replace(0, np.nan)
user_features['avg_purchase_value'] = user_features['avg_purchase_value'].fillna(0)

# Add device type distribution
device_dist = raw_data.groupby(['user_id', 'device_type']).size().unstack(fill_value=0)
device_dist.columns = [f'sessions_{col}' for col in device_dist.columns]
user_features = user_features.merge(device_dist, left_on='user_id', right_index=True, how='left')

# Fill any missing device columns
for device in ['mobile', 'desktop', 'tablet']:
    col = f'sessions_{device}'
    if col not in user_features.columns:
        user_features[col] = 0

# #region agent log
log_debug('create_churn_dataset.py:112', 'Additional features calculated', {
    'feature_count': len(user_features.columns),
    'sample_recency': int(user_features.iloc[0]['days_since_last_activity']),
    'features': list(user_features.columns)
}, 'H4')
# #endregion

# **CHURN LABEL CREATION**
# Churn = 1 if user's last session was BEFORE the churn window (last 7 days)
# Churn = 0 if user had activity in the last 7 days
user_features['churn'] = (user_features['last_session'] < churn_window_start).astype(int)

# #region agent log
churned_users = user_features[user_features['churn'] == 1]
active_users = user_features[user_features['churn'] == 0]
log_debug('create_churn_dataset.py:129', 'Churn labels created', {
    'total_users': len(user_features),
    'churned_count': int(user_features['churn'].sum()),
    'active_count': int((user_features['churn'] == 0).sum()),
    'churn_rate': float(user_features['churn'].mean()),
    'churned_sample': {
        'user_id': int(churned_users.iloc[0]['user_id']) if len(churned_users) > 0 else None,
        'last_session': str(churned_users.iloc[0]['last_session']) if len(churned_users) > 0 else None,
        'days_since_last': int(churned_users.iloc[0]['days_since_last_activity']) if len(churned_users) > 0 else None
    },
    'active_sample': {
        'user_id': int(active_users.iloc[0]['user_id']) if len(active_users) > 0 else None,
        'last_session': str(active_users.iloc[0]['last_session']) if len(active_users) > 0 else None,
        'days_since_last': int(active_users.iloc[0]['days_since_last_activity']) if len(active_users) > 0 else None
    }
}, 'H5')
# #endregion

# Display results
print("\n" + "="*80)
print("CHURN DATASET CREATED")
print("="*80)
print(f"\nTotal users: {len(user_features)}")
print(f"Churned users: {user_features['churn'].sum()} ({user_features['churn'].mean()*100:.1f}%)")
print(f"Active users: {(user_features['churn'] == 0).sum()} ({(1-user_features['churn'].mean())*100:.1f}%)")

print("\n" + "-"*80)
print("Feature Summary:")
print("-"*80)
print(user_features.describe())

print("\n" + "-"*80)
print("Sample Churned Users:")
print("-"*80)
print(user_features[user_features['churn'] == 1].head())

print("\n" + "-"*80)
print("Sample Active Users:")
print("-"*80)
print(user_features[user_features['churn'] == 0].head())

# Save the dataset
output_path = r'c:\Users\acer\Desktop\Churn System Design\notebooks\churn_dataset.csv'
user_features.to_csv(output_path, index=False)
print(f"\n✓ Dataset saved to: {output_path}")

# #region agent log
log_debug('create_churn_dataset.py:180', 'Dataset saved', {
    'output_path': output_path,
    'final_shape': list(user_features.shape),
    'columns': list(user_features.columns)
}, 'H6')
# #endregion

print("\n" + "="*80)
print("Process completed successfully!")
print("="*80)
