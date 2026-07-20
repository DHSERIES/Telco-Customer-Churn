RAW_FEATURES = ['Customer ID', 'Gender', 'Age', 'Under 30', 'Senior Citizen', 'Married', 'Dependents', 
                'Number of Dependents', 'Country', 'State', 'City', 'Zip Code', 'Latitude', 'Longitude',
                'Population', 'Quarter', 'Referred a Friend', 'Number of Referrals', 'Tenure in Months',
                'Offer', 'Phone Service', 'Avg Monthly Long Distance Charges', 'Multiple Lines', 
                'Internet Service', 'Internet Type', 'Avg Monthly GB Download', 'Online Security', 
                'Online Backup', 'Device Protection Plan', 'Premium Tech Support', 'Streaming TV',
                'Streaming Movies', 'Streaming Music', 'Unlimited Data', 'Contract', 'Paperless Billing',
                'Payment Method', 'Monthly Charge', 'Total Charges', 'Total Refunds',
                'Total Extra Data Charges', 'Total Long Distance Charges', 'Total Revenue', 
                'Satisfaction Score', 'Customer Status', 'Churn Score', 'CLTV', 
                'Churn Category', 'Churn Reason']
TARGET = "Churn Label"
CLEANED_NUM_COLS = ['Age', 'Number of Dependents', 'Latitude', 'Longitude', 'Population', 
            'Number of Referrals', 'Tenure in Months', 'Avg Monthly Long Distance Charges', 
            'Avg Monthly GB Download', 'Monthly Charge', 'Total Charges', 'Total Refunds', 
            'Total Extra Data Charges', 'Total Long Distance Charges', 'Total Revenue']

CLEANED_CAT_COLS = ['Gender', 'Senior Citizen', 'Married', 'Dependents', 'City', 
            'Referred a Friend', 'Offer', 'Phone Service', 'Multiple Lines','Internet_Service_detail',
            'Online Security', 'Online Backup', 'Device Protection Plan', 
            'Premium Tech Support', 'Streaming TV', 'Streaming Movies', 'Streaming Music', 
            'Unlimited Data', 'Contract', 'Paperless Billing', 'Payment Method','Zip Code']
# -----------------------------
# FEATURE ENGINEERING COLUMNS
# -----------------------------
FEATURE_ENGINEERING_REQUIRED_COLS = [
    "Latitude",
    "Longitude",
    "Population",
    "Number of Referrals",
    "Number of Dependents",
    "Total Revenue",
    "Tenure in Months",
    "Total Charges",
    "Total Refunds",
    "Monthly Charge",
    "Total Long Distance Charges",
    "Avg Monthly Long Distance Charges",
    "Avg Monthly GB Download",
    "Total Extra Data Charges",
    "Contract",
    "Payment Method",
    "Internet_Service_detail",
    "City",
    "Dependents",
    "Referred a Friend",
    "Offer",
    "Streaming TV",
    "Streaming Movies",
    "Streaming Music",
]


# =========================================================
# Numerical engineered columns
# =========================================================

ENGINEERED_NUM_COLS = [
    # Geographic
    "Latitude_2f",
    "Longitude_2f",
    "lat_x_lon",
    "lat_plus_lon",
    "lat_minus_lon",
    "dist_center",

    # Population
    "log_population",
    "pop_x_lat",
    "pop_x_lon",
    "logpop_x_lat",
    "logpop_x_lon",
    "pop_x_lat2f",
    "pop_x_lon2f",
    "pop_per_dist",

    # Referrals and dependents
    "referrals_x_dependents",
    "referrals_plus_dependents",
    "referrals_minus_dependents",
    "referrals_per_dependent",
    "has_referrals",
    "has_dependents",
    "has_referrals_and_dependents",

    # Financial and usage
    "Revenue_per_Month",
    "Charges_per_Month",
    "Refund_per_Month",
    "Expected_Charges",
    "Charge_Gap",
    "Charge_Ratio",
    "Refund_Rate",
    "Net_Revenue_Ratio",
    "Refunds_x_Tenure",
    "LD_Ratio",
    "AvgLD_x_Tenure",
    "LD_Gap",
    "GB_per_Dollar",
    "GB_x_MonthlyCharge",
    "ExtraData_Rate",
    "ExtraData_per_Month",
    "LD_Revenue_Share",
    "ExtraData_Revenue_Share",
    "Refund_Revenue_Share",
    "Tenure_x_MonthlyCharge",
    "Tenure_x_GB",
    "MonthlyCharge_x_GB",
    "Refunds_x_MonthlyCharge",
]


# =========================================================
# Basic categorical engineered columns
# =========================================================

BASIC_ENGINEERED_CAT_COLS = [
    "lat_lon_grid",
    "pop_geo",
]


# =========================================================
# Streaming columns
# =========================================================

STREAMING_COLS = [
    "Streaming TV",
    "Streaming Movies",
    "Streaming Music",
]

STREAMING_MAP = {
    "Yes": 1,
    "No": 0,
}


# =========================================================
# Pairwise categorical interactions
# =========================================================

PAIRWISE_INTERACTIONS = [
    # Contract-focused
    ("Contract", "Payment Method"),
    ("Contract", "Internet_Service_detail"),
    ("Contract", "City"),
    ("Contract", "Dependents"),
    ("Contract", "Referred a Friend"),
    ("Contract", "Offer"),
    ("Contract", "Streaming TV"),
    ("Contract", "Streaming Movies"),
    ("Contract", "Streaming Music"),

    # City-focused
    ("City", "Internet_Service_detail"),
    ("City", "Offer"),
    ("City", "Payment Method"),

    # Referral and family
    ("Referred a Friend", "Dependents"),
    ("Referred a Friend", "Offer"),

    # Internet-focused
    ("Internet_Service_detail", "Offer"),
    ("Internet_Service_detail", "Payment Method"),

    # Streaming-focused
    ("Streaming TV", "Streaming Movies"),
    ("Streaming TV", "Streaming Music"),
    ("Streaming Movies", "Streaming Music"),
    ("Internet_Service_detail", "Streaming TV"),
    ("Internet_Service_detail", "Streaming Movies"),
    ("Internet_Service_detail", "Streaming Music"),

    # Geography
    ("City", "Dependents"),
    ("City", "Referred a Friend"),
]

PAIRWISE_INTERACTION_COLS = [
    f"{col_1}_{col_2}_FE"
    for col_1, col_2 in PAIRWISE_INTERACTIONS
]

# Triple categorical interactions
# =========================================================

TRIPLE_INTERACTIONS = {
    "Contract_Internet_Offer_FE": [
        "Contract",
        "Internet_Service_detail",
        "Offer",
    ],
    "City_Contract_Internet_FE": [
        "City",
        "Contract",
        "Internet_Service_detail",
    ],
    "Referral_Dependent_Contract_FE": [
        "Referred a Friend",
        "Dependents",
        "Contract",
    ],
    "Contract_Payment_Internet_FE": [
        "Contract",
        "Payment Method",
        "Internet_Service_detail",
    ],
    "City_Offer_Internet_FE": [
        "City",
        "Offer",
        "Internet_Service_detail",
    ],
}

TRIPLE_INTERACTION_COLS = list(TRIPLE_INTERACTIONS.keys())

# Special categorical interactions
# =========================================================

SPECIAL_INTERACTION_COLS = [
    "Streaming_Bundle_FE",
    "Streaming_Count_FE",
    "Contract_StreamCount_FE",
    "City_StreamingBundle_FE",
]

# All categorical engineered columns
# =========================================================

INTERACTION_CAT_COLS = (
    PAIRWISE_INTERACTION_COLS
    + SPECIAL_INTERACTION_COLS
    + TRIPLE_INTERACTION_COLS
)

ENGINEERED_CAT_COLS = (
    BASIC_ENGINEERED_CAT_COLS
    + INTERACTION_CAT_COLS
)

# -----------------------------
# LATEST FEATURE CHOSEN
# -----------------------------

SELECTED_FEATURES = ['referrals_plus_dependents', 'Tenure in Months', 'Number of Referrals', 'Age', 
                  'Monthly Charge', 'te_Referral_Dependent_Contract_FE', 'te_Contract_StreamCount_FE',
                  'Total Long Distance Charges', 'te_Streaming_Bundle_FE', 
                  'te_Contract_Internet_Offer_FE', 'oe_pop_geo', 'Total Revenue', 
                  'te_Contract_Dependents_FE', 'Avg Monthly Long Distance Charges',
                  'te_City_Offer_FE', 'oe_City', 'te_City_Payment Method_FE', 'lat_minus_lon',
                  'te_Zip Code', 'dist_center', 'te_Internet_Service_detail_Offer_FE', 'logpop_x_lat',
                  'oe_Zip Code', 'te_Referred a Friend', 'te_Contract_Payment_Internet_FE', 
                  'Latitude_2f', 'te_Internet_Service_detail_Payment Method_FE', 'te_Streaming TV', 
                  'Total Charges', 'te_City_Dependents_FE', 'te_Streaming Movies_Streaming Music_FE',
                  'Latitude', 'referrals_per_dependent', 'te_Internet_Service_detail_Streaming TV_FE',
                  'te_Online Security', 'lat_x_lon', 'referrals_minus_dependents', 'log_population', 
                  'te_Married', 'te_Paperless Billing', 'lat_plus_lon', 'oe_lat_lon_grid', 
                  'te_Senior Citizen', 'te_City_StreamingBundle_FE', 
                  'te_Streaming TV_Streaming Movies_FE', 'oe_Premium Tech Support', 
                  'te_Device Protection Plan', 'te_Referred a Friend_Dependents_FE', 'te_City', 
                  'te_City_Referred a Friend_FE', 'Population', 'oe_Paperless Billing', 
                  'te_Contract_Payment Method_FE', 'te_Contract_Streaming Music_FE', 
                  'Avg Monthly GB Download', 'te_Gender', 'te_Premium Tech Support', 
                  'te_City_Internet_Service_detail_FE', 'oe_Streaming TV', 'oe_Online Security',
                  'te_Contract_Streaming TV_FE', 'te_Contract_Referred a Friend_FE', 'referrals_x_dependents', 
                  'pop_x_lat', 'te_Contract_City_FE', 'te_Internet_Service_detail', 'te_Payment Method', 'te_Offer', 
                  'te_Phone Service', 'oe_Married', 'te_City_Offer_Internet_FE', 'te_Contract_Offer_FE', 'pop_per_dist',
                  'pop_x_lat2f', 'te_Dependents', 'te_Streaming TV_Streaming Music_FE', 'logpop_x_lon', 'has_dependents',
                  'Longitude', 'te_Online Backup', 'Longitude_2f', 'te_Contract_Streaming Movies_FE', 'te_Contract',
                  'oe_Senior Citizen', 'oe_Payment Method', 'te_City_Contract_Internet_FE', 'oe_Referred a Friend', 
                  'te_Internet_Service_detail_Streaming Music_FE', 'te_Multiple Lines', 'has_referrals', 
                  'te_Internet_Service_detail_Streaming Movies_FE', 'oe_Device Protection Plan', 
                  'Total Extra Data Charges', 'te_Streaming_Count_FE', 'te_Referred a Friend_Offer_FE', 
                  'te_Contract_Internet_Service_detail_FE', 'oe_Multiple Lines', 'oe_Online Backup',
                  'oe_Streaming Movies', 'te_Streaming Music', 'te_Streaming Movies', 'te_Unlimited Data']
