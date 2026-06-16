# -----------------------------
# Define feature and target columns
# -----------------------------
# FEATURES contains all available input variables in the original dataset.
# TARGET is the variable to be predicted (customer churn status).
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATA_PATH = 'data/telco.csv'
assert Path(DATA_PATH).exists(), f"{DATA_PATH} does not exist."

df = pd.read_csv(DATA_PATH)

FEATURES = ['Customer ID', 'Gender', 'Age', 'Under 30', 'Senior Citizen', 'Married', 'Dependents', 
            'Number of Dependents', 'Country', 'State', 'City', 'Zip Code', 'Latitude', 'Longitude',
            'Population', 'Quarter', 'Referred a Friend', 'Number of Referrals', 'Tenure in Months',
            'Offer', 'Phone Service', 'Avg Monthly Long Distance Charges', 'Multiple Lines', 
            'Internet Service', 'Internet Type', 'Avg Monthly GB Download', 'Online Security', 
            'Online Backup', 'Device Protection Plan', 'Premium Tech Support', 'Streaming TV',
            'Streaming Movies', 'Streaming Music', 'Unlimited Data', 'Contract', 'Paperless Billing',
            'Payment Method', 'Monthly Charge', 'Total Charges', 'Total Refunds',
            'Total Extra Data Charges', 'Total Long Distance Charges', 'Total Revenue', 
            'Satisfaction Score', 'Customer Status', 'Churn Score', 'CLTV', 
            'Churn Category', 'Churn Reason']

TARGET = 'Churn Label'

assert set(FEATURES).issubset(df.columns), "Some features are missing from the dataset."
assert TARGET in df.columns, "Target variable is missing from the dataset."

train_df,test_df = train_test_split(df, test_size=0.2, random_state=1, stratify=df[TARGET])

# -----------------------------
# Encode the target variable
# -----------------------------
# Convert the churn labels from categorical values ("Yes"/"No")
# into binary numeric values (1/0) for machine learning models.

mappings = {'No': 0, 'Yes': 1}
reverse_mappings = {0: 'No', 1: 'Yes'}

# Check that all labels can be mapped
assert set(df[TARGET].dropna().unique()).issubset(mappings.keys()), \
    f"Unexpected labels found: {set(df[TARGET].dropna().unique()) - set(mappings.keys())}"

# Apply mapping
train_df[TARGET] = train_df[TARGET].map(mappings)
test_df[TARGET] = test_df[TARGET].map(mappings)

# -----------------------------
# Remove identifier column
# -----------------------------
# Customer ID is a unique identifier and does not provide predictive
# information, so it is removed from both datasets.

train_df = train_df.drop('Customer ID', axis=1, errors='ignore')
test_df = test_df.drop('Customer ID', axis=1, errors='ignore')

# -----------------------------
# Handle missing values
# -----------------------------
# Replace missing values in "Offer" with "None", indicating that
# the customer did not receive a promotional offer.

train_df['Offer'] = train_df['Offer'].fillna('None')
test_df['Offer'] = test_df['Offer'].fillna('None')

# Replace missing values in "Internet Type" with "No",
# representing customers without an internet service type.

train_df['Internet Type'] = train_df['Internet Type'].fillna('No')
test_df['Internet Type'] = test_df['Internet Type'].fillna('No')

# -----------------------------
# Remove data leakage features
# -----------------------------
# These variables contain information generated after churn occurs
# or are highly correlated with the target, so they are excluded
# to prevent data leakage.

train_df = train_df.drop(
    ['Churn Score', 'CLTV', 'Churn Category', 'Churn Reason'],
    axis=1,
    errors='ignore'
)

test_df = test_df.drop(
    ['Churn Score', 'CLTV', 'Churn Category', 'Churn Reason'],
    axis=1,
    errors='ignore'
)

# -----------------------------
# Remove low-value location/time features
# -----------------------------
# Country, State, and Quarter are removed because they provide
# limited predictive value or redundant information. ( value only collect in one country, State and quarter)

train_df = train_df.drop(['Country', 'State', 'Quarter'], axis=1, errors='ignore')
test_df = test_df.drop(['Country', 'State', 'Quarter'], axis=1, errors='ignore')

# -----------------------------
# Remove post-outcome/customer status features
# -----------------------------
# Customer Status and Satisfaction Score reveal information (see picture_1 and picture_2)
# closely related to churn and are excluded from modeling.

train_df = train_df.drop(
    ['Customer Status', 'Satisfaction Score'],
    axis=1,
    errors='ignore'
)

test_df = test_df.drop(
    ['Customer Status', 'Satisfaction Score'],
    axis=1,
    errors='ignore'
)

# -----------------------------
# Define numerical features
# -----------------------------
# These columns contain continuous or discrete numeric values and
# will typically be used for scaling or numerical preprocessing.

NUM_COLS = [
    'Age',
    'Number of Dependents',
    'Zip Code',
    'Latitude',
    'Longitude',
    'Population',
    'Number of Referrals',
    'Tenure in Months',
    'Avg Monthly Long Distance Charges',
    'Avg Monthly GB Download',
    'Monthly Charge',
    'Total Charges',
    'Total Refunds',
    'Total Extra Data Charges',
    'Total Long Distance Charges',
    'Total Revenue'
]

# -----------------------------
# Define categorical features
# -----------------------------
# These columns represent categorical attributes and will typically
# require encoding (e.g., one-hot encoding or label encoding).

CAT_COLS = [
    'Gender',
    'Under 30',
    'Senior Citizen',
    'Married',
    'Dependents',
    'City',
    'Referred a Friend',
    'Offer',
    'Phone Service',
    'Multiple Lines',
    'Internet Service',
    'Internet Type',
    'Online Security',
    'Online Backup',
    'Device Protection Plan',
    'Premium Tech Support',
    'Streaming TV',
    'Streaming Movies',
    'Streaming Music',
    'Unlimited Data',
    'Contract',
    'Paperless Billing',
    'Payment Method'
]

assert Counter(train_df.columns.tolist()) == Counter(NUM_COLS + CAT_COLS + [TARGET]), "Train DataFrame columns do not match expected features and target."
assert Counter(test_df.columns.tolist()) == Counter(NUM_COLS + CAT_COLS + [TARGET]), "Test DataFrame columns do not match expected features and target."

train_df.to_csv('data/train_data.csv', index=False)
test_df.to_csv('data/test_data.csv', index=False)

assert Path('data/train_data.csv').exists(), "Train data file was not created successfully."
assert Path('data/test_data.csv').exists(), "Test data file was not created successfully."

print("Preprocessing completed successfully. Train and test datasets are saved to 'data/train_data.csv' and 'data/test_data.csv'.")