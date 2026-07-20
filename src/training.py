import os
import joblib
import pandas as pd
import numpy as np
from typing import Any
from collections import Counter

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import clone
from sklearn.metrics import auc, roc_auc_score
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from src.feature_config import CLEANED_CAT_COLS, CLEANED_NUM_COLS, SELECTED_FEATURES
from src.feature_engineering import FeatureEngineer
from src.encoder_ import create_feature_engineering, FeatureEncoder
import warnings
from pandas.errors import PerformanceWarning

warnings.simplefilter("ignore", PerformanceWarning)

# config
N_SPLITS = 5
RANDOM_STATE = 1
skf = StratifiedKFold(n_splits=N_SPLITS,shuffle=True,
                      random_state=RANDOM_STATE)
params = {
    "subsample": 0.8,
    "reg_lambda": 0.1,
    "reg_alpha": 0.1,
    "num_leaves": 63,
    "n_estimators": 1000,
    "min_child_samples": 50,
    "max_depth": 3,
    "learning_rate": 0.01,
    "colsample_bytree": 1.0,
}
base_model = LGBMClassifier(
        objective="binary",
        random_state=RANDOM_STATE,
        verbose=-1,
        # GPU
        # device="gpu",
        # gpu_platform_id=0,
        # gpu_device_id=0,
        # max_bin=255,
        **params
    )

def train_simple_model():
    train_path = "data/train_data.csv"
    test_path = "data/test_data.csv"
    output_path = "result/customer_churn_oof.csv"

    assert os.path.exists(train_path), f"{train_path} not found"
    assert os.path.exists(test_path), f"{test_path} not found"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    FEATURES = CLEANED_CAT_COLS + CLEANED_NUM_COLS
    TARGET = "Churn Label"

    expected_columns = FEATURES + [TARGET]

    assert Counter(train_df.columns) == Counter(expected_columns), (
        "Train DataFrame columns do not match expected features and target."
    )
    assert Counter(test_df.columns) == Counter(expected_columns), (
        "Test DataFrame columns do not match expected features and target."
    )
    model = clone(base_model)
    transformer = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CLEANED_CAT_COLS),
            ("num", "passthrough", CLEANED_NUM_COLS),
        ]
    )
    pipeline = Pipeline([
        ("transformer", transformer),
        ("model", model)
    ])
    pipeline.fit(train_df[FEATURES], train_df[TARGET])
    y_pred = pipeline.predict(test_df[FEATURES])
    score = roc_auc_score(test_df[TARGET], y_pred)
    print(f"ROC AUC score: {score:.4f}")

    os.makedirs("model", exist_ok=True)
    joblib.dump(pipeline, "model/simple_churn_model.joblib")
    
def train_model():
    train_path = "data/train_data.csv"
    test_path = "data/test_data.csv"

    assert os.path.exists(train_path), f"{train_path} not found"
    assert os.path.exists(test_path), f"{test_path} not found"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    FEATURES = CLEANED_CAT_COLS + CLEANED_NUM_COLS
    TARGET = "Churn Label"

    expected_columns = FEATURES + [TARGET]

    assert Counter(train_df.columns) == Counter(expected_columns), (
        "Train DataFrame columns do not match expected features and target."
    )
    assert Counter(test_df.columns) == Counter(expected_columns), (
        "Test DataFrame columns do not match expected features and target."
    )
    X_train,X_val = train_df[FEATURES].copy(), test_df[FEATURES].copy()
    y_train = train_df[TARGET]

    # Step 1: create engineered features
    X_train_fe, lat_center, lon_center = create_feature_engineering(
        X_train
    )

    X_val_fe, _, _ = create_feature_engineering(
        X_val,
        lat_center=lat_center,
        lon_center=lon_center,
    )

    # Step 2: encode features
    X_train_encoded, X_val_encoded, feature_names, encoders = (
        FeatureEncoder(
            X_train_fe,
            X_val_fe,
            y_train,
        )
    )

    missing_features = (
        set(SELECTED_FEATURES) - set(X_train_encoded.columns)
    )
    if missing_features:
        raise ValueError(
            f"Missing selected features: {sorted(missing_features)}"
        )

    model = clone(base_model)

    model.fit(
        X_train_encoded[SELECTED_FEATURES],
        y_train,
    )

    deployment_bundle = {
        "model": model,
        "encoders": encoders,
        "feature_names": SELECTED_FEATURES,
        "lat_center": lat_center,
        "lon_center": lon_center,
    }

    os.makedirs("model", exist_ok=True)

    joblib.dump(
        deployment_bundle,
        "model/churn_model.joblib",
    )


if __name__ == "__main__":
    train_model()