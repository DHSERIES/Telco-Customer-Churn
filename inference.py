import joblib
import pandas as pd

from src.encoder_ import create_feature_engineering, TransformFeatureEncoder
from src.feature_config import CLEANED_CAT_COLS, CLEANED_NUM_COLS


MODEL_PATH = "model/churn_model.joblib"
FEATURES = CLEANED_CAT_COLS + CLEANED_NUM_COLS


def load_deployment_bundle(
    model_path: str = MODEL_PATH,
) -> dict:
    bundle = joblib.load(model_path)

    required_keys = {
        "model",
        "encoders",
        "feature_names",
        "lat_center",
        "lon_center",
    }

    missing_keys = required_keys - set(bundle)

    if missing_keys:
        raise ValueError(
            f"Deployment bundle is missing keys: {sorted(missing_keys)}"
        )

    return bundle


def predict_new_data(
    new_data: pd.DataFrame,
    model_path: str = MODEL_PATH,
) -> pd.DataFrame:
    """
    Predict churn for new raw customer data.

    new_data must contain the original cleaned input columns:
        CLEANED_CAT_COLS + CLEANED_NUM_COLS
    """
    if not isinstance(new_data, pd.DataFrame):
        raise TypeError("new_data must be a pandas DataFrame.")

    if new_data.empty:
        raise ValueError("new_data is empty.")

    duplicate_columns = new_data.columns[
        new_data.columns.duplicated()
    ].tolist()

    if duplicate_columns:
        raise ValueError(
            f"Duplicate columns found: {duplicate_columns}"
        )

    missing_columns = set(FEATURES) - set(new_data.columns)

    if missing_columns:
        raise ValueError(
            "New data is missing required raw columns: "
            f"{sorted(missing_columns)}"
        )
    
    bundle = load_deployment_bundle(model_path)

    model = bundle["model"]
    encoders = bundle["encoders"]
    selected_features = bundle["feature_names"]
    lat_center = bundle["lat_center"]
    lon_center = bundle["lon_center"]

    # Deployment accept only the cleaned columns.
    X_new = new_data[FEATURES].copy()

    # Step 1: create the same engineered features used in training.
    X_new_fe, _, _ = create_feature_engineering(
        X_new,
        lat_center=lat_center,
        lon_center=lon_center,
    )

    # Step 2: apply the encoders fitted during training.
    X_new_encoded = TransformFeatureEncoder(
        X_new_fe,
        encoders,
    )

    missing_model_features = (
        set(selected_features) - set(X_new_encoded.columns)
    )

    if missing_model_features:
        raise ValueError(
            "Prediction data is missing model features: "
            f"{sorted(missing_model_features)}"
        )

    # Preserve the exact training feature order.
    X_model = X_new_encoded[selected_features].copy()

    predictions = model.predict_proba(X_model)

    result = new_data.copy()
    reverse_mappings = {0: "No", 1: "Yes"}
    result["Churn Prediction"] = predictions[:, 1]
    result["Yes_probability"] = predictions[:, 1]
    result["No_probability"] = predictions[:, 0]
    return result

