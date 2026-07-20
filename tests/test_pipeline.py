import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cleaning import clean_data
from src.feature_config import CLEANED_CAT_COLS, CLEANED_NUM_COLS, TARGET
from src.encoder_ import create_feature_engineering, FeatureEncoder
from src.training import train_model


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "telco.csv"


def test_clean_data_returns_expected_columns_and_target() -> None:
    df = pd.read_csv(DATA_PATH)

    cleaned = clean_data(df)

    expected_columns = CLEANED_NUM_COLS + CLEANED_CAT_COLS + [TARGET]

    assert list(cleaned.columns) == expected_columns
    assert cleaned[TARGET].isin([0, 1]).all()


def test_feature_engineering_and_encoder_pipeline_produce_features() -> None:
    df = pd.read_csv(DATA_PATH)
    cleaned = clean_data(df)

    X = cleaned.drop(columns=[TARGET]).copy()
    y = cleaned[TARGET].astype(int).copy()

    X_train = X.iloc[:200].copy()
    X_val = X.iloc[200:400].copy()
    y_train = y.iloc[:200].copy()

    X_train_fe, _, _ = create_feature_engineering(X_train)
    X_val_fe, _, _ = create_feature_engineering(X_val)

    X_train_encoded, X_val_encoded, feature_names, encoders = FeatureEncoder(
        X_train_fe,
        X_val_fe,
        y_train,
    )

    assert len(feature_names) > 0
    assert set(feature_names).issubset(set(X_train_encoded.columns))
    assert set(feature_names).issubset(set(X_val_encoded.columns))
    assert "ordinal" in encoders and "target" in encoders


def test_train_model_creates_bundle(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(DATA_PATH)
    cleaned = clean_data(raw_df)

    train_df = cleaned.iloc[: int(0.8 * len(cleaned))].copy()
    test_df = cleaned.iloc[int(0.8 * len(cleaned)) :].copy()

    train_df.to_csv(data_dir / "train_data.csv", index=False)
    test_df.to_csv(data_dir / "test_data.csv", index=False)

    train_model()

    bundle_path = tmp_path / "model" / "churn_model.joblib"
    assert bundle_path.exists()
