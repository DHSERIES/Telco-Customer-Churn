import importlib
import sys
from pathlib import Path

from fastapi.testclient import TestClient


def reload_main(monkeypatch):
    import joblib

    class DummyTransformer:
        def __init__(self, prefix):
            self.prefix = prefix

        def transform(self, X):
            return [f"{self.prefix}_{v}" for v in X.iloc[:, 0].astype(str)]

    class DummyModel:
        def predict(self, X):
            return ["dummy_prediction"]

    def fake_load(path):
        encoder = {}
        for col in [
            'Gender','Under 30','Senior Citizen','Married',
            'Dependents','City','Referred a Friend',
            'Offer','Phone Service','Multiple Lines',
            'Internet Service','Internet Type','Online Security',
            'Online Backup','Device Protection Plan',
            'Premium Tech Support','Streaming TV','Streaming Movies',
            'Streaming Music','Unlimited Data','Contract',
            'Paperless Billing','Payment Method'
        ]:
            encoder[f"oe_{col}"] = DummyTransformer("oe")
            encoder[f"te_{col}"] = DummyTransformer("te")

        return {
            "model": DummyModel(),
            "encoder": encoder,
            "FEATURES": [f"oe_{col}" for col in [
                'Gender','Under 30','Senior Citizen','Married',
                'Dependents','City','Referred a Friend',
                'Offer','Phone Service','Multiple Lines',
                'Internet Service','Internet Type','Online Security',
                'Online Backup','Device Protection Plan',
                'Premium Tech Support','Streaming TV','Streaming Movies',
                'Streaming Music','Unlimited Data','Contract',
                'Paperless Billing','Payment Method'
            ]] + [f"te_{col}" for col in [
                'Gender','Under 30','Senior Citizen','Married',
                'Dependents','City','Referred a Friend',
                'Offer','Phone Service','Multiple Lines',
                'Internet Service','Internet Type','Online Security',
                'Online Backup','Device Protection Plan',
                'Premium Tech Support','Streaming TV','Streaming Movies',
                'Streaming Music','Unlimited Data','Contract',
                'Paperless Billing','Payment Method'
            ]]
        }

    monkeypatch.setattr(joblib, "load", fake_load)
    # ensure repository root is on sys.path so imports resolve
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    if "inference" in sys.modules:
        del sys.modules["inference"]
    if "main" in sys.modules:
        del sys.modules["main"]
    return importlib.import_module("main")


def make_customer_payload():
    payload = {
        "Age": 35.0,
        "Number_of_Dependents": 2.0,
        "Zip_Code": 12345.0,
        "Latitude": 37.0,
        "Longitude": -122.0,
        "Population": 100000.0,
        "Number_of_Referrals": 1.0,
        "Tenure_in_Months": 10.0,
        "Avg_Monthly_Long_Distance_Charges": 20.0,
        "Avg_Monthly_GB_Download": 5.0,
        "Monthly_Charge": 80.0,
        "Total_Charges": 800.0,
        "Total_Refunds": 0.0,
        "Total_Extra_Data_Charges": 0.0,
        "Total_Long_Distance_Charges": 20.0,
        "Total_Revenue": 820.0,
        "Gender": "Female",
        "Under_30": "No",
        "Senior_Citizen": "No",
        "Married": "Yes",
        "Dependents": "No",
        "City": "Sample City",
        "Referred_a_Friend": "No",
        "Offer": "No",
        "Phone_Service": "Yes",
        "Multiple_Lines": "No",
        "Internet_Service": "Fiber optic",
        "Internet_Type": "DSL",
        "Online_Security": "No",
        "Online_Backup": "No",
        "Device_Protection_Plan": "No",
        "Premium_Tech_Support": "No",
        "Streaming_TV": "No",
        "Streaming_Movies": "No",
        "Streaming_Music": "No",
        "Unlimited_Data": "No",
        "Contract": "Month-to-month",
        "Paperless_Billing": "Yes",
        "Payment_Method": "Electronic check"
    }
    return payload


def test_home_returns_message(monkeypatch):
    main = reload_main(monkeypatch)
    client = TestClient(main.app)

    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Customer Churn Prediction API"}


def test_predict_customer_returns_prediction(monkeypatch):
    main = reload_main(monkeypatch)
    client = TestClient(main.app)

    response = client.post("/predict", json=make_customer_payload())

    assert response.status_code == 200
    assert response.json() == ["dummy_prediction"]


def test_predict_customer_validation_error(monkeypatch):
    main = reload_main(monkeypatch)
    client = TestClient(main.app)
    payload = make_customer_payload()
    payload.pop("Age")

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert response.json()["detail"]
