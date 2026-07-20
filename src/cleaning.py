from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.feature_config import (
    CLEANED_CAT_COLS,
    CLEANED_NUM_COLS,
    RAW_FEATURES,
    TARGET,
)

def clean_original_data():
    DATA_PATH = 'data/telco.csv'
    assert Path(DATA_PATH).exists(), f"{DATA_PATH} does not exist."

    df = pd.read_csv(DATA_PATH)

    TARGET = 'Churn Label'

    assert set(RAW_FEATURES).issubset(df.columns), "Some features are missing from the dataset."
    assert TARGET in df.columns, "Target variable is missing from the dataset."

    train_df,test_df = train_test_split(df, test_size=0.2, random_state=1, stratify=df[TARGET])

    # -----------------------------
    # Encode the target variable
    # -----------------------------
    # Convert the churn labels from categorical values ("Yes"/"No")
    # into binary numeric values (1/0) for machine learning models.

    mappings = {'No': 0, 'Yes': 1}

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

    # drop 'Under 30' column as it is redundant with 'Age' and may introduce multicollinearity.
    train_df = train_df.drop(['Under 30'], axis = 1, errors = 'ignore')
    test_df = test_df.drop(['Under 30'], axis = 1 , errors = 'ignore')
    # -----------------------------
    # Handle missing values
    # -----------------------------
    # Replace missing values in "Offer" with "None", indicating that
    # the customer did not receive a promotional offer.

    train_df['Offer'] = train_df['Offer'].fillna('None')
    test_df['Offer'] = test_df['Offer'].fillna('None')

    # Replace missing values in "Internet Type" with "No_Internet_Service", indicating that
    # the customer does not have an internet service.

    train_df['Internet_Service_detail'] = train_df['Internet Type'].copy()
    train_df['Internet_Service_detail'] = train_df['Internet_Service_detail'].fillna('No_Internet_Service')
    test_df['Internet_Service_detail'] = test_df['Internet Type'].copy()
    test_df['Internet_Service_detail'] = test_df['Internet_Service_detail'].fillna('No_Internet_Service')

    train_df = train_df.drop(['Internet Service', 'Internet Type'], axis=1, errors='ignore')
    test_df = test_df.drop(['Internet Service', 'Internet Type'], axis=1, errors='ignore')

    # Convert Zip Code to string type as it is a categorical variable rather than a numeric one.
    train_df['Zip Code'] = train_df['Zip Code'].astype(str)
    test_df['Zip Code'] = test_df['Zip Code'].astype(str)

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
    # redundant information. ( value only collect in one country, State and quarter)

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

    assert Counter(train_df.columns.tolist()) == Counter(CLEANED_NUM_COLS + CLEANED_CAT_COLS + [TARGET]), "Train DataFrame columns do not match expected features and target."
    assert Counter(test_df.columns.tolist()) == Counter(CLEANED_NUM_COLS + CLEANED_CAT_COLS + [TARGET]), "Test DataFrame columns do not match expected features and target."

    train_df.to_csv('data/train_data.csv', index=False)
    test_df.to_csv('data/test_data.csv', index=False)

    assert Path('data/train_data.csv').exists(), "Train data file was not created successfully."
    assert Path('data/test_data.csv').exists(), "Test data file was not created successfully."

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["data_source"] = "train"
    test_df["data_source"] = "test"

    full_df = pd.concat(
        [train_df, test_df],
        axis=0,
        ignore_index=True,
    )

    full_df.to_csv('data/cleaned_data.csv', index=False)
    
    assert Path('data/cleaned_data.csv').exists(), "Cleaned data file was not created successfully."

    print(" create train_data.csv and test_data.csv and cleaned_data.csv successfully.")
################################################################

def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    missing_features = set(RAW_FEATURES) - set(df.columns)
    assert not missing_features, (
        f"Some required features are missing from the dataset: "
        f"{sorted(missing_features)}"
    )

    cleaned_df = df.copy()

    # Encode target when it is present.
    if TARGET in cleaned_df.columns:
        mappings = {"No": 0, "Yes": 1}

        target_values = set(cleaned_df[TARGET].dropna().unique())
        valid_values = set(mappings) | {0, 1}

        unexpected_labels = target_values - valid_values
        assert not unexpected_labels, (
            f"Unexpected target labels found: {unexpected_labels}"
        )

        cleaned_df[TARGET] = cleaned_df[TARGET].replace(mappings)

    # Remove identifier and redundant columns.
    cleaned_df = cleaned_df.drop(
        columns=["Customer ID", "Under 30"],
        errors="ignore",
    )

    # Handle missing Offer values.
    if "Offer" in cleaned_df.columns:
        cleaned_df["Offer"] = cleaned_df["Offer"].fillna("None")

    # Combine internet service information into one categorical feature.
    if "Internet Type" in cleaned_df.columns:
        cleaned_df["Internet_Service_detail"] = (
            cleaned_df["Internet Type"]
            .fillna("No_Internet_Service")
        )

    cleaned_df = cleaned_df.drop(
        columns=["Internet Service", "Internet Type"],
        errors="ignore",
    )

    # Treat Zip Code as a categorical feature while preserving missing values.
    if "Zip Code" in cleaned_df.columns:
        cleaned_df["Zip Code"] = cleaned_df["Zip Code"].astype("string")

    # Remove leakage and low-value features.
    columns_to_drop = [
        "Churn Score",
        "CLTV",
        "Churn Category",
        "Churn Reason",
        "Country",
        "State",
        "Quarter",
        "Customer Status",
        "Satisfaction Score",
    ]

    cleaned_df = cleaned_df.drop(
        columns=columns_to_drop,
        errors="ignore",
    )

    expected_columns = CLEANED_NUM_COLS + CLEANED_CAT_COLS

    if TARGET in cleaned_df.columns:
        expected_columns = expected_columns + [TARGET]

    actual_columns = cleaned_df.columns.tolist()

    assert Counter(actual_columns) == Counter(expected_columns), (
        "Cleaned DataFrame columns do not match the expected columns. "
        f"Missing: {sorted(set(expected_columns) - set(actual_columns))}. "
        f"Unexpected: {sorted(set(actual_columns) - set(expected_columns))}."
    )

    # Return columns in a stable, configured order.
    return cleaned_df.loc[:, expected_columns].copy()