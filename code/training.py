import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder,TargetEncoder
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

NUM_COLS = [
    'Age','Number of Dependents','Zip Code','Latitude','Longitude',
    'Population','Number of Referrals','Tenure in Months','Avg Monthly Long Distance Charges',
    'Avg Monthly GB Download','Monthly Charge','Total Charges','Total Refunds',
    'Total Extra Data Charges','Total Long Distance Charges','Total Revenue'
]

CAT_COLS = [
    'Gender','Under 30','Senior Citizen','Married','Dependents',
    'City','Referred a Friend','Offer','Phone Service','Multiple Lines',
    'Internet Service','Internet Type','Online Security',
    'Online Backup','Device Protection Plan','Premium Tech Support',
    'Streaming TV','Streaming Movies','Streaming Music','Unlimited Data',
    'Contract','Paperless Billing','Payment Method'
]

def feature_engineering(
    X_train,X_val,
    y_train,
    num_cols = NUM_COLS,cat_cols = CAT_COLS
):  
    X_train = X_train.copy()
    X_val = X_val.copy()

    FE = []
    encoder = {}
    
    # Numerical features
    for col in num_cols:
        FE.append(col)

    # Ordinal Encoding
    for col in cat_cols:
        oe = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )

        X_train[f"oe_{col}"] = oe.fit_transform(X_train[[col]]).ravel()
        X_val[f"oe_{col}"] = oe.transform(X_val[[col]]).ravel()
        
        encoder[f"oe_{col}"] = oe
        FE.append(f"oe_{col}")

    # Target Encoding
    for col in cat_cols:
        te = TargetEncoder()

        X_train[f"te_{col}"] = te.fit_transform(
            X_train[[col]], y_train).ravel()

        X_val[f"te_{col}"] = te.transform(
            X_val[[col]]).ravel()

        encoder[f"te_{col}"] = te
        FE.append(f"te_{col}")

    return X_train, X_val, FE, encoder

train_path = 'data/train_data.csv'
test_path = 'data/test_data.csv'

assert os.path.exists(train_path), f"{train_path} not found"
assert os.path.exists(test_path), f"{test_path} not found"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Configure the features and target variable
FEATURES = train_df.columns.drop('Churn Label').tolist()
TARGET = 'Churn Label' 

assert Counter(train_df.columns.tolist()) == Counter(NUM_COLS + CAT_COLS + [TARGET]), "Train DataFrame columns do not match expected features and target."
assert Counter(test_df.columns.tolist()) == Counter(NUM_COLS + CAT_COLS + [TARGET]), "Test DataFrame columns do not match expected features and target."

# config
N_SPLITS = 3
random_state = 42
skf = StratifiedKFold(n_splits=N_SPLITS,shuffle=True,
                      random_state=random_state)

xgb_params = {  
    'n_estimators': 2000,
    'max_depth': 6,
    'learning_rate': 0.05,
    'random_state': random_state,
    'eval_metric': 'logloss'
}
model = XGBClassifier(**xgb_params) 

X = train_df[NUM_COLS + CAT_COLS].copy()
y = train_df[TARGET]
acc = []
for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):

    X_train = X.iloc[train_idx].copy()
    X_val = X.iloc[valid_idx].copy()
    y_train = y.iloc[train_idx]
    y_val = y.iloc[valid_idx]

    X_train, X_val, FE, encoder = feature_engineering(X_train, X_val, y_train)
    model.fit(X_train[FE], y_train)

    y_pred = model.predict(X_val[FE])
    fold_accuracy = accuracy_score(y_val, y_pred)
    acc.append(fold_accuracy)
    print(f"Fold {fold}: Accuracy = {fold_accuracy:.4f}")
print(f"Average eval Accuracy across {N_SPLITS} folds: {np.mean(acc):.4f}")

print("Training complete. Final model trained on all data.")
# Train final model on all data
X_prep, test_prep, FE, encoder = feature_engineering(X, test_df[NUM_COLS + CAT_COLS], y)
model.fit(X_prep[FE], y)
test_pred = model.predict(test_prep[FE])
test_acc = accuracy_score(test_df[TARGET], test_pred)
print(f"Test Accuracy: {test_acc:.4f}")

# Save model
model_dir = "model"

os.makedirs(model_dir, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "encoder": encoder,
        "FEATURES": FE
    },
    f"{model_dir}/xgb_model.pkl"
)
print(f"Model and encoders saved to {model_dir} path")
