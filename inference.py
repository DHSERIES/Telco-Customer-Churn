import joblib
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder

# -----------------------------
# Feature configuration
# -----------------------------
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

obj = joblib.load("model/xgb_model.pkl")
model, encoder, FEATURES = obj["model"], obj["encoder"], obj["FEATURES"]

def transform_new_data(new_df, encoder):
    X = new_df.copy()

    for col in CAT_COLS:
        oe = encoder[f"oe_{col}"]
        X[f"oe_{col}"] = oe.transform(X[[col]])

    for col in CAT_COLS:
        te = encoder[f"te_{col}"]
        X[f"te_{col}"] = te.transform(X[[col]])

    return X[FEATURES]
def predict(data):

    df = pd.DataFrame([data])
    assert all(col in df.columns for col in NUM_COLS + CAT_COLS), "Missing required columns"
    
    X = df[NUM_COLS + CAT_COLS]
    X = transform_new_data(X, encoder)
    pred = model.predict(X)

    return pred