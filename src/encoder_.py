import numpy as np
import pandas as pd
from typing import Any

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder
from src.feature_config import (
    CLEANED_CAT_COLS,
    CLEANED_NUM_COLS,
    FEATURE_ENGINEERING_REQUIRED_COLS,
    ENGINEERED_CAT_COLS,
    ENGINEERED_NUM_COLS,
    PAIRWISE_INTERACTIONS,
    STREAMING_COLS,
    STREAMING_MAP,
    BASIC_ENGINEERED_CAT_COLS,
    INTERACTION_CAT_COLS,
    TRIPLE_INTERACTIONS,
)


def create_feature_engineering(
    X: pd.DataFrame,
    lat_center: float | None = None,
    lon_center: float | None = None,
) -> tuple[pd.DataFrame, float, float]:
    """
    Create engineered features without fitting or applying encoders.

    Training:
        X_train_fe, lat_center, lon_center = (
            create_feature_engineering(X_train)
        )

    Validation or prediction:
        X_new_fe, _, _ = create_feature_engineering(
            X_new,
            lat_center=lat_center,
            lon_center=lon_center,
        )
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    X = X.copy()

    missing_columns = (
        set(FEATURE_ENGINEERING_REQUIRED_COLS) - set(X.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing columns required for feature engineering: "
            f"{sorted(missing_columns)}"
        )

    if lat_center is not None and not isinstance(lat_center, float):
        raise TypeError("lat_center must be numeric or None.")

    if lon_center is not None and not isinstance(lon_center, float):
        raise TypeError("lon_center must be numeric or None.")

    # =====================================================
    # Geographic features
    # =====================================================

    X["Latitude_2f"] = X["Latitude"].round(2)
    X["Longitude_2f"] = X["Longitude"].round(2)

    X["lat_lon_grid"] = (
        X["Latitude_2f"].astype(str)
        + "_"
        + X["Longitude_2f"].astype(str)
    )

    X["lat_x_lon"] = (
        X["Latitude_2f"] * X["Longitude_2f"]
    )

    X["lat_plus_lon"] = (
        X["Latitude_2f"] + X["Longitude_2f"]
    )

    X["lat_minus_lon"] = (
        X["Latitude_2f"] - X["Longitude_2f"]
    )

    if lat_center is None:
        lat_center = float(X["Latitude"].median())
    else:
        lat_center = float(lat_center)

    if lon_center is None:
        lon_center = float(X["Longitude"].median())
    else:
        lon_center = float(lon_center)

    X["dist_center"] = np.sqrt(
        (X["Latitude"] - lat_center) ** 2
        + (X["Longitude"] - lon_center) ** 2
    )

    # =====================================================
    # Population interactions
    # =====================================================

    X["log_population"] = np.log1p(X["Population"])

    X["pop_x_lat"] = (
        X["Population"] * X["Latitude"]
    )

    X["pop_x_lon"] = (
        X["Population"] * X["Longitude"]
    )

    X["logpop_x_lat"] = (
        X["log_population"] * X["Latitude"]
    )

    X["logpop_x_lon"] = (
        X["log_population"] * X["Longitude"]
    )

    X["pop_x_lat2f"] = (
        X["Population"] * X["Latitude_2f"]
    )

    X["pop_x_lon2f"] = (
        X["Population"] * X["Longitude_2f"]
    )

    X["pop_per_dist"] = (
        X["Population"] / (X["dist_center"] + 1e-6)
    )

    X["pop_geo"] = (
        (X["Population"] // 1000)
        .astype(int)
        .astype(str)
        + "_"
        + X["lat_lon_grid"]
    )

    # =====================================================
    # Referrals and dependents
    # =====================================================

    referrals = X["Number of Referrals"]
    dependents = X["Number of Dependents"]

    X["referrals_x_dependents"] = referrals * dependents

    X["referrals_plus_dependents"] = referrals + dependents

    X["referrals_minus_dependents"] = referrals - dependents

    X["referrals_per_dependent"] = (
        referrals / (dependents + 1)
    )

    X["has_referrals"] = (
        referrals > 0
    ).astype(int)

    X["has_dependents"] = (
        dependents > 0
    ).astype(int)

    X["has_referrals_and_dependents"] = (
        (referrals > 0)
        & (dependents > 0)
    ).astype(int)

    # =====================================================
    # Financial and usage features
    # =====================================================

    tenure = X["Tenure in Months"]
    total_revenue = X["Total Revenue"]
    total_charges = X["Total Charges"]
    total_refunds = X["Total Refunds"]
    monthly_charge = X["Monthly Charge"]
    total_ld_charges = X["Total Long Distance Charges"]
    avg_ld_charges = X["Avg Monthly Long Distance Charges"]
    avg_gb = X["Avg Monthly GB Download"]
    extra_data_charges = X["Total Extra Data Charges"]

    X["Revenue_per_Month"] = (
        total_revenue / (tenure + 1)
    )

    X["Charges_per_Month"] = (
        total_charges / (tenure + 1)
    )

    X["Refund_per_Month"] = (
        total_refunds / (tenure + 1)
    )

    X["Expected_Charges"] = monthly_charge * tenure

    X["Charge_Gap"] = (
        total_charges - X["Expected_Charges"]
    )

    X["Charge_Ratio"] = (
        total_charges / (X["Expected_Charges"] + 1)
    )

    X["Refund_Rate"] = (
        total_refunds / (total_charges + 1)
    )

    X["Net_Revenue_Ratio"] = (
        total_revenue / (total_charges + 1)
    )

    X["Refunds_x_Tenure"] = (
        total_refunds * np.log1p(tenure)
    )

    X["LD_Ratio"] = (
        total_ld_charges / (total_charges + 1)
    )

    X["AvgLD_x_Tenure"] = (
        avg_ld_charges * tenure
    )

    X["LD_Gap"] = (
        total_ld_charges - avg_ld_charges * tenure
    )

    X["GB_per_Dollar"] = (
        avg_gb / (monthly_charge + 1)
    )

    X["GB_x_MonthlyCharge"] = (
        avg_gb * monthly_charge
    )

    X["ExtraData_Rate"] = (
        extra_data_charges / (total_revenue + 1)
    )

    X["ExtraData_per_Month"] = (
        extra_data_charges / (tenure + 1)
    )

    X["LD_Revenue_Share"] = (
        total_ld_charges / (total_revenue + 1)
    )

    X["ExtraData_Revenue_Share"] = (
        extra_data_charges / (total_revenue + 1)
    )

    X["Refund_Revenue_Share"] = (
        total_refunds / (total_revenue + 1)
    )

    X["Tenure_x_MonthlyCharge"] = (
        tenure * monthly_charge
    )

    X["Tenure_x_GB"] = tenure * avg_gb

    X["MonthlyCharge_x_GB"] = (
        monthly_charge * avg_gb
    )

    X["Refunds_x_MonthlyCharge"] = (
        total_refunds * monthly_charge
    )

    # =====================================================
    # Pairwise categorical interactions
    # =====================================================

    for col_1, col_2 in PAIRWISE_INTERACTIONS:
        feature_name = f"{col_1}_{col_2}_FE"

        X[feature_name] = (
            X[col_1].astype(str)
            + "__"
            + X[col_2].astype(str)
        )

    # =====================================================
    # Streaming interactions
    # =====================================================

    X["Streaming_Bundle_FE"] = (
        X[STREAMING_COLS]
        .fillna("missing")
        .astype(str)
        .agg("__".join, axis=1)
    )

    streaming_count = (
        X[STREAMING_COLS]
        .apply(lambda column: column.map(STREAMING_MAP))
        .fillna(0)
        .sum(axis=1)
        .astype(int)
    )

    X["Streaming_Count_FE"] = streaming_count.astype(str)

    X["Contract_StreamCount_FE"] = (
        X["Contract"].astype(str)
        + "__"
        + X["Streaming_Count_FE"]
    )

    X["City_StreamingBundle_FE"] = (
        X["City"].astype(str)
        + "__"
        + X["Streaming_Bundle_FE"]
    )

    # =====================================================
    # Triple categorical interactions
    # =====================================================

    for feature_name, columns in TRIPLE_INTERACTIONS.items():
        X[feature_name] = (
            X[columns]
            .fillna("missing")
            .astype(str)
            .agg("__".join, axis=1)
        )

    # =====================================================
    # Validate generated output
    # =====================================================

    expected_engineered_columns = (
        ENGINEERED_NUM_COLS + ENGINEERED_CAT_COLS
    )

    missing_engineered_columns = (
        set(expected_engineered_columns) - set(X.columns)
    )

    if missing_engineered_columns:
        raise RuntimeError(
            "Feature engineering failed to create columns: "
            f"{sorted(missing_engineered_columns)}"
        )

    return X, lat_center, lon_center


def FeatureEncoder(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    dict[str, Any],
]:
    """
    Fit encoders using training data and transform train/validation data.

    Expected input:
        X_train and X_val already passed through
        create_feature_engineering().

    Returns:
        X_train_encoded
        X_val_encoded
        feature_names
        fitted_encoders
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")

    if not isinstance(X_val, pd.DataFrame):
        raise TypeError("X_val must be a pandas DataFrame.")

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train and y_train must contain the same number of rows."
        )

    X_train = X_train.copy()
    X_val = X_val.copy()
    y_train = pd.Series(y_train, index=X_train.index).copy()

    feature_names: list[str] = []
    encoders: dict[str, Any] = {
        "ordinal": {},
        "target": {},
    }

    # =====================================================
    # Validate required columns
    # =====================================================

    ordinal_columns = (
        CLEANED_CAT_COLS
        + BASIC_ENGINEERED_CAT_COLS
    )

    target_columns = (
        CLEANED_CAT_COLS
        + INTERACTION_CAT_COLS
    )

    required_columns = set(
        CLEANED_NUM_COLS
        + ENGINEERED_NUM_COLS
        + ordinal_columns
        + target_columns
    )

    missing_train_columns = required_columns - set(X_train.columns)
    missing_val_columns = required_columns - set(X_val.columns)

    if missing_train_columns:
        raise ValueError(
            "X_train is missing columns: "
            f"{sorted(missing_train_columns)}"
        )

    if missing_val_columns:
        raise ValueError(
            "X_val is missing columns: "
            f"{sorted(missing_val_columns)}"
        )

    # =====================================================
    # Numerical features
    # =====================================================

    numerical_features = (
        CLEANED_NUM_COLS
        + ENGINEERED_NUM_COLS
    )

    feature_names.extend(numerical_features)

    # =====================================================
    # Ordinal encoding
    # =====================================================

    for column in ordinal_columns:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )

        encoded_column = f"oe_{column}"

        X_train[encoded_column] = (
            encoder.fit_transform(X_train[[column]]).ravel()
        )

        X_val[encoded_column] = (
            encoder.transform(X_val[[column]]).ravel()
        )

        encoders["ordinal"][column] = encoder
        feature_names.append(encoded_column)

    # =====================================================
    # Target encoding
    # =====================================================

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for column in target_columns:
        encoder = TargetEncoder(
            cv=cv,
            target_type="binary",
            smooth="auto",
        )

        encoded_column = f"te_{column}"

        X_train[encoded_column] = encoder.fit_transform(
            X_train[[column]],
            y_train,
        )

        X_val[encoded_column] = encoder.transform(
            X_val[[column]]
        )

        encoders["target"][column] = encoder
        feature_names.append(encoded_column)

    # Remove duplicate names while preserving order.
    feature_names = list(dict.fromkeys(feature_names))

    missing_encoded_train = (
        set(feature_names) - set(X_train.columns)
    )

    missing_encoded_val = (
        set(feature_names) - set(X_val.columns)
    )

    if missing_encoded_train:
        raise RuntimeError(
            "Failed to create training features: "
            f"{sorted(missing_encoded_train)}"
        )

    if missing_encoded_val:
        raise RuntimeError(
            "Failed to create validation features: "
            f"{sorted(missing_encoded_val)}"
        )

    return X_train, X_val, feature_names, encoders


def TransformFeatureEncoder(
    X: pd.DataFrame,
    encoders: dict[str, Any],
) -> pd.DataFrame:
    """
    Transform new data using previously fitted encoders.

    X must already have passed through create_feature_engineering().
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    X = X.copy()

    ordinal_encoders = encoders.get("ordinal", {})
    target_encoders = encoders.get("target", {})

    for column, encoder in ordinal_encoders.items():
        if column not in X.columns:
            raise ValueError(
                f"Missing ordinal-encoding column: {column}"
            )

        X[f"oe_{column}"] = (
            encoder.transform(X[[column]]).ravel()
        )

    for column, encoder in target_encoders.items():
        if column not in X.columns:
            raise ValueError(
                f"Missing target-encoding column: {column}"
            )

        X[f"te_{column}"] = encoder.transform(
            X[[column]]
        )

    return X