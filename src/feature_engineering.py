from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder
import numpy as np
from src.feature_config import (
    CLEANED_NUM_COLS,
    CLEANED_CAT_COLS,
)

def FeatureEngineer(
    X_train,
    X_val,
    y_train,
    num_cols = CLEANED_NUM_COLS,
    cat_cols = CLEANED_CAT_COLS
):
    X_train = X_train.copy()
    X_val = X_val.copy()
    FE = []
    ###_Latitude_X_Longitude FE###
    
    # Round coordinates
    for df in [X_train, X_val]:
        df["Latitude_2f"] = df["Latitude"].round(2)
        df["Longitude_2f"] = df["Longitude"].round(2)
    
        # Geographic bucket
        df["lat_lon_grid"] = (df["Latitude_2f"].astype(str) + "_" + df["Longitude_2f"].astype(str))
    
        # Numeric interactions
        df["lat_x_lon"] = (df["Latitude_2f"] * df["Longitude_2f"])
    
        df["lat_plus_lon"] = (df["Latitude_2f"] + df["Longitude_2f"])
    
        df["lat_minus_lon"] = (df["Latitude_2f"] - df["Longitude_2f"])
    
    # Distance from train center 
    lat_center = X_train["Latitude"].median()
    lon_center = X_train["Longitude"].median()
    
    for df in [X_train, X_val]:
        df["dist_center"] = np.sqrt(
            (df["Latitude"] - lat_center) ** 2 +
            (df["Longitude"] - lon_center) ** 2
        )
    # =========================
    # Population Interactions
    # =========================
    
    for df in [X_train, X_val]:
    
        df["log_population"] = np.log1p(df["Population"])
    
        # Population × Coordinates
        df["pop_x_lat"] = (df["Population"] * df["Latitude"])
    
        df["pop_x_lon"] = (df["Population"] * df["Longitude"])
    
        # Log-population × Coordinates
        df["logpop_x_lat"] = (df["log_population"] * df["Latitude"])
    
        df["logpop_x_lon"] = (df["log_population"] * df["Longitude"])
    
        # Population × Rounded Coordinates
        df["pop_x_lat2f"] = (df["Population"] * df["Latitude_2f"])
    
        df["pop_x_lon2f"] = (df["Population"] * df["Longitude_2f"])
    
        # Population relative to center
        df["pop_per_dist"] = (df["Population"] / (df["dist_center"] + 1e-6))
    
        # Population + Geo Bucket (for CatBoost)
        df["pop_geo"] = ((df["Population"] // 1000).astype(int).astype(str) + "_" + df["lat_lon_grid"])
    
    new_features = [
        "Latitude_2f","Longitude_2f",
        "lat_lon_grid","lat_x_lon",
        "lat_plus_lon","lat_minus_lon",
        "dist_center",
        "log_population",
        "pop_x_lat","pop_x_lon",
        "logpop_x_lat","logpop_x_lon",
        "pop_x_lat2f","pop_x_lon2f",
        "pop_per_dist","pop_geo",
    ]
    # orinal enc
    cat_features_new = [
        "lat_lon_grid",
        "pop_geo",
    ]
    
    oe = OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1)
    
    X_train[cat_features_new] = oe.fit_transform(
        X_train[cat_features_new]
    )
    
    X_val[cat_features_new] = oe.transform(
        X_val[cat_features_new]
    )
    
    FE += new_features
    ### Referrals and Dependents
    
    new_features = [
    'referrals_x_dependents',
    'referrals_plus_dependents',
    'referrals_minus_dependents',
    'referrals_per_dependent',
    'has_referrals',
    'has_dependents',
    'has_referrals_and_dependents'
]
    for df in [X_train, X_val]:
        # Basic interactions
        df['referrals_x_dependents'] = (
            df['Number of Referrals'] * df['Number of Dependents']
        )
    
        df['referrals_plus_dependents'] = (
            df['Number of Referrals'] + df['Number of Dependents']
        )
    
        df['referrals_minus_dependents'] = (
            df['Number of Referrals'] - df['Number of Dependents']
        )
    
        df['referrals_per_dependent'] = (
            df['Number of Referrals'] / (df['Number of Dependents'] + 1)
        )
    
        # is Referrals interactions
        df['has_referrals'] = (
            df['Number of Referrals'] > 0
        ).astype(int)
    
        df['has_dependents'] = (
            df['Number of Dependents'] > 0
        ).astype(int)
    
        df['has_referrals_and_dependents'] = (
            (df['Number of Referrals'] > 0) &
            (df['Number of Dependents'] > 0)
        ).astype(int)
        
    FE += new_features
    
    for df in [X_train, X_val]:
    
        # Revenue / tenure interactions
        df['Revenue_per_Month'] = (
            df['Total Revenue'] / (df['Tenure in Months'] + 1)
        )
    
        df['Charges_per_Month'] = (
            df['Total Charges'] / (df['Tenure in Months'] + 1)
        )
    
        df['Refund_per_Month'] = (
            df['Total Refunds'] / (df['Tenure in Months'] + 1)
        )
    
        # Actual vs expected spending
        df['Expected_Charges'] = (
            df['Monthly Charge'] * df['Tenure in Months']
        )
    
        df['Charge_Gap'] = (
            df['Total Charges'] - df['Expected_Charges']
        )
    
        df['Charge_Ratio'] = (
            df['Total Charges'] / (df['Expected_Charges'] + 1)
        )
    
        # Refund-related interactions
        df['Refund_Rate'] = (
            df['Total Refunds'] / (df['Total Charges'] + 1)
        )
    
        df['Net_Revenue_Ratio'] = (
            df['Total Revenue'] / (df['Total Charges'] + 1)
        )
    
        df['Refunds_x_Tenure'] = (
            df['Total Refunds'] * np.log1p(df['Tenure in Months'])
        )
    
        # Long-distance behavior
        df['LD_Ratio'] = (
            df['Total Long Distance Charges'] / (df['Total Charges'] + 1)
        )
    
        df['AvgLD_x_Tenure'] = (
            df['Avg Monthly Long Distance Charges']
            * df['Tenure in Months']
        )
    
        df['LD_Gap'] = (
            df['Total Long Distance Charges']
            - df['Avg Monthly Long Distance Charges']
            * df['Tenure in Months']
        )
    
        # Data usage interactions
        df['GB_per_Dollar'] = (
            df['Avg Monthly GB Download']
            / (df['Monthly Charge'] + 1)
        )
    
        df['GB_x_MonthlyCharge'] = (
            df['Avg Monthly GB Download']
            * df['Monthly Charge']
        )
    
        # Extra-data spending
        df['ExtraData_Rate'] = (
            df['Total Extra Data Charges']
            / (df['Total Revenue'] + 1)
        )
    
        df['ExtraData_per_Month'] = (
            df['Total Extra Data Charges']
            / (df['Tenure in Months'] + 1)
        )
    
        # Revenue composition
        df['LD_Revenue_Share'] = (
            df['Total Long Distance Charges']
            / (df['Total Revenue'] + 1)
        )
    
        df['ExtraData_Revenue_Share'] = (
            df['Total Extra Data Charges']
            / (df['Total Revenue'] + 1)
        )
    
        df['Refund_Revenue_Share'] = (
            df['Total Refunds']
            / (df['Total Revenue'] + 1)
        )
    
        # Polynomial interactions
        df['Tenure_x_MonthlyCharge'] = (
            df['Tenure in Months']
            * df['Monthly Charge']
        )
    
        df['Tenure_x_GB'] = (
            df['Tenure in Months']
            * df['Avg Monthly GB Download']
        )
    
        df['MonthlyCharge_x_GB'] = (
            df['Monthly Charge']
            * df['Avg Monthly GB Download']
        )
    
        df['Refunds_x_MonthlyCharge'] = (
            df['Total Refunds']
            * df['Monthly Charge']
        )

    # Numerical features
    for col in num_cols:
        FE.append(col)

    # Ordinal Encoding
    for col in cat_cols:
        oe = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )

        X_train[f"oe_{col}"] = oe.fit_transform(X_train[[col]])
        X_val[f"oe_{col}"] = oe.transform(X_val[[col]])
        FE.append(f"oe_{col}")

    # Target Encoding
    te_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for col in cat_cols:
        te = TargetEncoder(cv=te_cv)

        X_train[f"te_{col}"] = te.fit_transform(
            X_train[[col]], y_train
        )
        X_val[f"te_{col}"] = te.transform(
            X_val[[col]]
        )
        FE.append(f"te_{col}")
######
    # Categorical interaction features + Target Encoding
    
    # Pairwise interactions
    interaction_pairs = [
    
        # Contract-focused
        ('Contract', 'Payment Method'),
        ('Contract', 'Internet_Service_detail'),
        ('Contract', 'City'),
        ('Contract', 'Dependents'),
        ('Contract', 'Referred a Friend'),
        ('Contract', 'Offer'),
        ('Contract', 'Streaming TV'),
        ('Contract', 'Streaming Movies'),
        ('Contract', 'Streaming Music'),
    
        # City-focused
        ('City', 'Internet_Service_detail'),
        ('City', 'Offer'),
        ('City', 'Payment Method'),
    
        # Referral / family
        ('Referred a Friend', 'Dependents'),
        ('Referred a Friend', 'Offer'),
    
        # Internet-focused
        ('Internet_Service_detail', 'Offer'),
        ('Internet_Service_detail', 'Payment Method'),
    
        # Streaming-focused
        ('Streaming TV', 'Streaming Movies'),
        ('Streaming TV', 'Streaming Music'),
        ('Streaming Movies', 'Streaming Music'),
    
        ('Internet_Service_detail', 'Streaming TV'),
        ('Internet_Service_detail', 'Streaming Movies'),
        ('Internet_Service_detail', 'Streaming Music'),
    
        # Geography
        ('City', 'Dependents'),
        ('City', 'Referred a Friend'),
    ]
    
    interaction_cols = []
    
    for c1, c2 in interaction_pairs:
    
        feat = f'{c1}_{c2}_FE'
    
        X_train[feat] = (
            X_train[c1].astype(str)
            + '__'
            + X_train[c2].astype(str)
        )
    
        X_val[feat] = (
            X_val[c1].astype(str)
            + '__'
            + X_val[c2].astype(str)
        )
    
        interaction_cols.append(feat)
    
    # --------------------------
    # Streaming bundle
    # --------------------------
    stream_cols = [
        'Streaming TV',
        'Streaming Movies',
        'Streaming Music'
    ]
    
    feat = 'Streaming_Bundle_FE'
    
    X_train[feat] = (
        X_train[stream_cols]
        .astype(str)
        .agg('__'.join, axis=1)
    )
    
    X_val[feat] = (
        X_val[stream_cols]
        .astype(str)
        .agg('__'.join, axis=1)
    )
    
    interaction_cols.append(feat)
    
    # --------------------------
    # Streaming count
    # --------------------------
    stream_map = {
        'Yes': 1,
        'No': 0
    }
    
    feat = 'Streaming_Count_FE'
    
    X_train[feat] = (
        X_train[stream_cols]
        .apply(lambda s: s.map(stream_map))
        .fillna(0)
        .sum(axis=1)
        .astype(int)
        .astype(str)
    )
    
    X_val[feat] = (
        X_val[stream_cols]
        .apply(lambda s: s.map(stream_map))
        .fillna(0)
        .sum(axis=1)
        .astype(int)
        .astype(str)
)
    
    interaction_cols.append(feat)
    
    # --------------------------
    # Contract x Streaming Count
    # --------------------------
    feat = 'Contract_StreamCount_FE'
    
    X_train[feat] = (
        X_train['Contract'].astype(str)
        + '__'
        + X_train['Streaming_Count_FE']
    )
    
    X_val[feat] = (
        X_val['Contract'].astype(str)
        + '__'
        + X_val['Streaming_Count_FE']
    )
    
    interaction_cols.append(feat)
    
    # --------------------------
    # City x Streaming Bundle
    # --------------------------
    feat = 'City_StreamingBundle_FE'
    
    X_train[feat] = (
        X_train['City'].astype(str)
        + '__'
        + X_train['Streaming_Bundle_FE']
    )
    
    X_val[feat] = (
        X_val['City'].astype(str)
        + '__'
        + X_val['Streaming_Bundle_FE']
    )
    
    interaction_cols.append(feat)
    
    # --------------------------
    # Triple interactions
    # --------------------------
    triple_features = {
        'Contract_Internet_Offer_FE':
            ['Contract', 'Internet_Service_detail', 'Offer'],
    
        'City_Contract_Internet_FE':
            ['City', 'Contract', 'Internet_Service_detail'],
    
        'Referral_Dependent_Contract_FE':
            ['Referred a Friend', 'Dependents', 'Contract'],
    
        'Contract_Payment_Internet_FE':
            ['Contract', 'Payment Method', 'Internet_Service_detail'],
    
        'City_Offer_Internet_FE':
            ['City', 'Offer', 'Internet_Service_detail'],
    }
    
    for feat, cols in triple_features.items():
    
        X_train[feat] = (
            X_train[cols]
            .apply(lambda x: '__'.join([str(val) for val in x]), axis=1)
        )
    
        X_val[feat] = (
            X_val[cols]
            .apply(lambda x: '__'.join([str(val) for val in x]), axis=1)
        )
    
        interaction_cols.append(feat)
    
    # ==========================================================
    # Target Encoding for engineered features
    # ==========================================================
    
    for col in interaction_cols:
    
        te = TargetEncoder(cv=5)
    
        X_train[f"te_{col}"] = te.fit_transform(
            X_train[[col]],
            y_train
        )
    
        X_val[f"te_{col}"] = te.transform(
            X_val[[col]]
        )
    
        FE.append(f"te_{col}")
    
    return X_train, X_val, FE