import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

CAT_COLS = [
    'Gender','Under 30','Senior Citizen','Married',
    'Dependents','City','Referred a Friend',
    'Offer','Phone Service','Multiple Lines',
    'Internet Service','Internet Type',
    'Online Security','Online Backup',
    'Device Protection Plan',
    'Premium Tech Support',
    'Streaming TV','Streaming Movies',
    'Streaming Music','Unlimited Data',
    'Contract','Paperless Billing',
    'Payment Method'
]

NUM_COLS = [
    'Age','Number of Dependents','Zip Code','Latitude',
    'Longitude','Population','Number of Referrals',
    'Tenure in Months','Avg Monthly Long Distance Charges',
    'Avg Monthly GB Download','Monthly Charge',
    'Total Charges','Total Refunds',
    'Total Extra Data Charges',
    'Total Long Distance Charges',
    'Total Revenue'
]


class DummyTransformer:
    def __init__(self, prefix):
        self.prefix = prefix

    def transform(self, X):
        values = X.iloc[:, 0].astype(str)
        return values.apply(lambda v: f"{self.prefix}_{v}").to_list()


class DummyModel:
    def predict(self, X):
        return ["dummy_prediction"] * len(X)


def reload_inference(monkeypatch):
    import joblib

    def fake_load(path):
        encoder = {}
        for col in CAT_COLS:
            encoder[f"oe_{col}"] = DummyTransformer("oe")
            encoder[f"te_{col}"] = DummyTransformer("te")

        return {
            "model": DummyModel(),
            "encoder": encoder,
            "FEATURES": [f"oe_{col}" for col in CAT_COLS] + [f"te_{col}" for col in CAT_COLS]
        }

    monkeypatch.setattr(joblib, "load", fake_load)
    # ensure repository root is on sys.path so `import inference` resolves
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    if "inference" in sys.modules:
        del sys.modules["inference"]
    return importlib.import_module("inference")


def make_valid_input():
    data = {col: 1.0 for col in NUM_COLS}
    data.update({col: "test" for col in CAT_COLS})
    return data


def test_transform_new_data_returns_features_in_FE(monkeypatch):
    inference = reload_inference(monkeypatch)
    input_df = pd.DataFrame([{col: "value" for col in CAT_COLS}])
    transformed = inference.transform_new_data(input_df, inference.encoder)

    assert list(transformed.columns) == inference.FEATURES
    assert transformed.iloc[0, 0] == "oe_value"
    assert transformed.iloc[0, len(CAT_COLS)] == "te_value"


def test_predict_success(monkeypatch):
    inference = reload_inference(monkeypatch)
    output = inference.predict(make_valid_input())

    assert output == ["dummy_prediction"]
    assert len(output) == 1


def test_predict_missing_required_column_raises(monkeypatch):
    inference = reload_inference(monkeypatch)
    data = make_valid_input()
    data.pop("Age")

    with pytest.raises(AssertionError, match="Missing required columns"):
        inference.predict(data)
