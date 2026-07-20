import math

import pandas as pd
import streamlit as st

from inference import predict_new_data


def normalize_prediction_payload(customer: pd.DataFrame) -> pd.DataFrame:
    normalized = customer.copy()

    value_map = {
        "Internet_Service_detail": {
            "No internet service": "No_Internet_Service",
            "No": "No_Internet_Service",
        },
        "Multiple Lines": {"No phone service": "No"},
        "Online Security": {"No internet service": "No"},
        "Online Backup": {"No internet service": "No"},
        "Device Protection Plan": {"No internet service": "No"},
        "Premium Tech Support": {"No internet service": "No"},
        "Streaming TV": {"No internet service": "No"},
        "Streaming Movies": {"No internet service": "No"},
        "Streaming Music": {"No internet service": "No"},
        "Unlimited Data": {"No internet service": "No"},
    }

    for column, mapping in value_map.items():
        if column in normalized.columns:
            normalized[column] = normalized[column].replace(mapping)

    return normalized


st.set_page_config(
    page_title="Telco Customer Churn",
    page_icon="📊",
    layout="wide",
)

YES_NO = ["No", "Yes"]

EXAMPLES = {
    "Low risk": {
        "Gender": "Female",
        "Senior Citizen": "No",
        "Married": "Yes",
        "Dependents": "Yes",
        "City": "Los Angeles",
        "Referred a Friend": "Yes",
        "Offer": "Offer A",
        "Phone Service": "Yes",
        "Multiple Lines": "Yes",
        "Internet_Service_detail": "Fiber Optic",
        "Online Security": "Yes",
        "Online Backup": "Yes",
        "Device Protection Plan": "Yes",
        "Premium Tech Support": "Yes",
        "Streaming TV": "Yes",
        "Streaming Movies": "Yes",
        "Streaming Music": "Yes",
        "Unlimited Data": "Yes",
        "Contract": "Two Year",
        "Paperless Billing": "No",
        "Payment Method": "Credit Card",
        "Zip Code": "90001",
        "Age": 45,
        "Number of Dependents": 2,
        "Latitude": 33.9731,
        "Longitude": -118.2479,
        "Population": 57110,
        "Number of Referrals": 4,
        "Tenure in Months": 60,
        "Avg Monthly Long Distance Charges": 18.0,
        "Avg Monthly GB Download": 22.0,
        "Monthly Charge": 75.0,
        "Total Charges": 4500.0,
        "Total Refunds": 0.0,
        "Total Extra Data Charges": 0.0,
        "Total Long Distance Charges": 1080.0,
        "Total Revenue": 5580.0,
    },
    "Medium risk": {
        "Gender": "Male",
        "Senior Citizen": "No",
        "Married": "Yes",
        "Dependents": "No",
        "City": "San Diego",
        "Referred a Friend": "No",
        "Offer": "None",
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet_Service_detail": "Cable",
        "Online Security": "No",
        "Online Backup": "Yes",
        "Device Protection Plan": "No",
        "Premium Tech Support": "No",
        "Streaming TV": "Yes",
        "Streaming Movies": "No",
        "Streaming Music": "Yes",
        "Unlimited Data": "Yes",
        "Contract": "One Year",
        "Paperless Billing": "Yes",
        "Payment Method": "Bank Withdrawal",
        "Zip Code": "92101",
        "Age": 37,
        "Number of Dependents": 0,
        "Latitude": 32.7157,
        "Longitude": -117.1611,
        "Population": 37226,
        "Number of Referrals": 0,
        "Tenure in Months": 20,
        "Avg Monthly Long Distance Charges": 24.0,
        "Avg Monthly GB Download": 30.0,
        "Monthly Charge": 88.0,
        "Total Charges": 1760.0,
        "Total Refunds": 0.0,
        "Total Extra Data Charges": 20.0,
        "Total Long Distance Charges": 480.0,
        "Total Revenue": 2260.0,
    },
    "High risk": {
        "Gender": "Female",
        "Senior Citizen": "Yes",
        "Married": "No",
        "Dependents": "No",
        "City": "Los Angeles",
        "Referred a Friend": "No",
        "Offer": "None",
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet_Service_detail": "Fiber Optic",
        "Online Security": "No",
        "Online Backup": "No",
        "Device Protection Plan": "No",
        "Premium Tech Support": "No",
        "Streaming TV": "Yes",
        "Streaming Movies": "Yes",
        "Streaming Music": "No",
        "Unlimited Data": "Yes",
        "Contract": "Month-to-Month",
        "Paperless Billing": "Yes",
        "Payment Method": "Mailed Check",
        "Zip Code": "90001",
        "Age": 68,
        "Number of Dependents": 0,
        "Latitude": 33.9731,
        "Longitude": -118.2479,
        "Population": 57110,
        "Number of Referrals": 0,
        "Tenure in Months": 3,
        "Avg Monthly Long Distance Charges": 32.0,
        "Avg Monthly GB Download": 40.0,
        "Monthly Charge": 105.0,
        "Total Charges": 315.0,
        "Total Refunds": 0.0,
        "Total Extra Data Charges": 30.0,
        "Total Long Distance Charges": 96.0,
        "Total Revenue": 441.0,
    },
}


def apply_example(name: str) -> None:
    for field, value in EXAMPLES[name].items():
        st.session_state[f"field_{field}"] = value


def categorical_input(label: str, options: list[str]) -> str:
    key = f"field_{label}"
    current = st.session_state.get(key, options[0])
    if current not in options:
        options = [current, *options]
    return st.selectbox(label, options, index=options.index(current), key=key)


def number_input(
    label: str,
    default: float,
    minimum: float = 0.0,
    step: float = 1.0,
) -> float:
    key = f"field_{label}"
    return st.number_input(
        label,
        min_value=minimum,
        value=float(st.session_state.get(key, default)),
        step=step,
        key=key,
    )


def overview_page() -> None:
    st.title("Telco Customer Churn Prediction")
    st.write(
        "Predict customer churn risk, estimate near-term revenue exposure, "
        "and support targeted retention decisions."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Deployment model", "LightGBM")
    c2.metric("Test ROC-AUC", "0.926")
    c3.metric("Dataset", "7,000+")
    c4.metric("Input features", "37")

    st.subheader("Business problem")
    st.write(
        "Churn reduces recurring telecom revenue. This project returns churn "
        "probability and translates it into a revenue-at-risk estimate and "
        "retention priority."
    )

    st.subheader("Workflow")
    st.code(
        "Customer data → validation → feature engineering → encoding → "
        "LightGBM → churn probability → business action",
        language=None,
    )

    st.subheader("Key churn patterns")
    i1, i2, i3 = st.columns(3)
    i1.info("Month-to-month contracts show the highest churn risk.")
    i2.info("Short-tenure customers with high monthly charges are higher risk.")
    i3.info("Security, backup, protection, and support services improve retention.")

    st.subheader("Model comparison")
    results = pd.DataFrame(
        {
            "Model": [
                "LightGBM",
                "CatBoost",
                "XGBoost",
                "Logistic Regression",
                "Linear SVM",
                "Random Forest",
            ],
            "Test ROC-AUC": [
                0.925924,
                0.923392,
                0.922713,
                0.918177,
                0.916125,
                0.913288,
            ],
        }
    )
    st.dataframe(results, hide_index=True, use_container_width=True)

    st.subheader("Business scoring")
    f1 = st.container()
    f1.code(
        "Revenue at Risk = Churn Probability × Monthly Charge × 3",
        language=None,
    )

def prediction_page() -> None:
    st.title("Single Customer Prediction")
    st.write("Enter customer information or load an editable example profile.")

    def numeric_input(
        label: str,
        default: float,
        minimum: float = 0.0,
        step: float = 1.0,
        format_string: str | None = None,
    ) -> float:
        """
        Render a Streamlit number input with consistent float types.

        Streamlit requires value, min_value, max_value, and step to use
        the same numeric type.
        """
        key = f"field_{label}"
        stored_value = st.session_state.get(key, default)

        kwargs = {
            "label": label,
            "min_value": float(minimum),
            "value": float(stored_value),
            "step": float(step),
            "key": key,
        }

        if format_string is not None:
            kwargs["format"] = format_string

        return float(st.number_input(**kwargs))

    example_col, button_col = st.columns([3, 1])

    selected_example = example_col.selectbox(
        "Example profile",
        list(EXAMPLES.keys()),
    )

    if button_col.button(
        "Load example",
        use_container_width=True,
    ):
        apply_example(selected_example)
        st.rerun()

    with st.form("customer_prediction_form"):
        st.subheader("Customer and account")

        col1, col2, col3 = st.columns(3)

        with col1:
            gender = categorical_input(
                "Gender",
                ["Female", "Male"],
            )

            age = numeric_input(
                "Age",
                default=40.0,
                minimum=18.0,
                step=1.0,
                format_string="%.0f",
            )

            senior = categorical_input(
                "Senior Citizen",
                YES_NO,
            )

            married = categorical_input(
                "Married",
                YES_NO,
            )

            dependents = categorical_input(
                "Dependents",
                YES_NO,
            )

            number_dependents = numeric_input(
                "Number of Dependents",
                default=0.0,
                minimum=0.0,
                step=1.0,
                format_string="%.0f",
            )

        with col2:
            city = st.text_input(
                "City",
                value=str(
                    st.session_state.get(
                        "field_City",
                        "Los Angeles",
                    )
                ),
                key="field_City",
            )

            zip_code = st.text_input(
                "Zip Code",
                value=str(
                    st.session_state.get(
                        "field_Zip Code",
                        "90001",
                    )
                ),
                key="field_Zip Code",
            )

            latitude = numeric_input(
                "Latitude",
                default=33.9731,
                minimum=-90.0,
                step=0.0001,
                format_string="%.4f",
            )

            longitude = numeric_input(
                "Longitude",
                default=-118.2479,
                minimum=-180.0,
                step=0.0001,
                format_string="%.4f",
            )

            population = numeric_input(
                "Population",
                default=57110.0,
                minimum=0.0,
                step=100.0,
                format_string="%.0f",
            )

            tenure = numeric_input(
                "Tenure in Months",
                default=12.0,
                minimum=0.0,
                step=1.0,
                format_string="%.0f",
            )

        with col3:
            referred = categorical_input(
                "Referred a Friend",
                YES_NO,
            )

            referrals = numeric_input(
                "Number of Referrals",
                default=0.0,
                minimum=0.0,
                step=1.0,
                format_string="%.0f",
            )

            offer = categorical_input(
                "Offer",
                [
                    "None",
                    "Offer A",
                    "Offer B",
                    "Offer C",
                    "Offer D",
                    "Offer E",
                ],
            )

            contract = categorical_input(
                "Contract",
                [
                    "Month-to-Month",
                    "One Year",
                    "Two Year",
                ],
            )

            paperless = categorical_input(
                "Paperless Billing",
                YES_NO,
            )

            payment = categorical_input(
                "Payment Method",
                [
                    "Bank Withdrawal",
                    "Credit Card",
                    "Mailed Check",
                ],
            )

        st.subheader("Services")

        service_col1, service_col2, service_col3 = st.columns(3)

        internet_options = [
            "No",
            "Yes",
            "No internet service",
        ]

        with service_col1:
            phone = categorical_input(
                "Phone Service",
                YES_NO,
            )

            multiple_lines = categorical_input(
                "Multiple Lines",
                [
                    "No",
                    "Yes",
                    "No phone service",
                ],
            )

            internet = categorical_input(
                "Internet_Service_detail",
                [
                    "Fiber Optic",
                    "Cable",
                    "DSL",
                    "No",
                ],
            )

            online_security = categorical_input(
                "Online Security",
                internet_options,
            )

        with service_col2:
            online_backup = categorical_input(
                "Online Backup",
                internet_options,
            )

            device_protection = categorical_input(
                "Device Protection Plan",
                internet_options,
            )

            tech_support = categorical_input(
                "Premium Tech Support",
                internet_options,
            )

            unlimited = categorical_input(
                "Unlimited Data",
                internet_options,
            )

        with service_col3:
            streaming_tv = categorical_input(
                "Streaming TV",
                internet_options,
            )

            streaming_movies = categorical_input(
                "Streaming Movies",
                internet_options,
            )

            streaming_music = categorical_input(
                "Streaming Music",
                internet_options,
            )

            avg_gb = numeric_input(
                "Avg Monthly GB Download",
                default=20.0,
                minimum=0.0,
                step=1.0,
            )

        st.subheader("Charges and revenue")

        charge_col1, charge_col2, charge_col3 = st.columns(3)

        with charge_col1:
            monthly_charge = numeric_input(
                "Monthly Charge",
                default=75.0,
                minimum=0.0,
                step=1.0,
            )

            total_charges = numeric_input(
                "Total Charges",
                default=900.0,
                minimum=0.0,
                step=10.0,
            )

            total_revenue = numeric_input(
                "Total Revenue",
                default=1100.0,
                minimum=0.0,
                step=10.0,
            )

        with charge_col2:
            avg_long_distance = numeric_input(
                "Avg Monthly Long Distance Charges",
                default=20.0,
                minimum=0.0,
                step=1.0,
            )

            total_long_distance = numeric_input(
                "Total Long Distance Charges",
                default=240.0,
                minimum=0.0,
                step=10.0,
            )

        with charge_col3:
            total_refunds = numeric_input(
                "Total Refunds",
                default=0.0,
                minimum=0.0,
                step=1.0,
            )

            extra_data = numeric_input(
                "Total Extra Data Charges",
                default=0.0,
                minimum=0.0,
                step=1.0,
            )

        submitted = st.form_submit_button(
            "Predict churn",
            use_container_width=True,
        )

    if not submitted:
        return

    customer = pd.DataFrame(
        [
            {
                "Gender": gender,
                "Senior Citizen": senior,
                "Married": married,
                "Dependents": dependents,
                "City": city.strip(),
                "Referred a Friend": referred,
                "Offer": offer,
                "Phone Service": phone,
                "Multiple Lines": multiple_lines,
                "Internet_Service_detail": internet,
                "Online Security": online_security,
                "Online Backup": online_backup,
                "Device Protection Plan": device_protection,
                "Premium Tech Support": tech_support,
                "Streaming TV": streaming_tv,
                "Streaming Movies": streaming_movies,
                "Streaming Music": streaming_music,
                "Unlimited Data": unlimited,
                "Contract": contract,
                "Paperless Billing": paperless,
                "Payment Method": payment,
                "Zip Code": zip_code.strip(),
                "Age": int(age),
                "Number of Dependents": int(number_dependents),
                "Latitude": latitude,
                "Longitude": longitude,
                "Population": int(population),
                "Number of Referrals": int(referrals),
                "Tenure in Months": int(tenure),
                "Avg Monthly Long Distance Charges": avg_long_distance,
                "Avg Monthly GB Download": avg_gb,
                "Monthly Charge": monthly_charge,
                "Total Charges": total_charges,
                "Total Refunds": total_refunds,
                "Total Extra Data Charges": extra_data,
                "Total Long Distance Charges": total_long_distance,
                "Total Revenue": total_revenue,
            }
        ]
    )

    try:
        result = predict_new_data(customer)

        churn_probability = float(
            result.iloc[0]["Yes_probability"]
        )
        no_churn_probability = float(
            result.iloc[0]["No_probability"]
        )

    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        st.error(f"Prediction failed: {error}")
        return

    except Exception as error:
        st.error(
            "An unexpected prediction error occurred: "
            f"{error}"
        )
        return

    revenue_at_risk = (
        churn_probability
        * monthly_charge
        * 3.0
    )

    retention_priority = (
        revenue_at_risk
        * math.log1p(tenure)
    )

    if churn_probability >= 0.60:
        risk_level = "High"
        recommendation = (
            "Prioritize direct retention outreach and provide "
            "a personalized offer."
        )
        st.error(
            f"High churn risk: {churn_probability:.1%}"
        )

    elif churn_probability >= 0.30:
        risk_level = "Medium"
        recommendation = (
            "Review the customer's service needs and include "
            "them in an engagement campaign."
        )
        st.warning(
            f"Medium churn risk: {churn_probability:.1%}"
        )

    else:
        risk_level = "Low"
        recommendation = (
            "Maintain standard service and monitor the customer "
            "for future risk changes."
        )
        st.success(
            f"Low churn risk: {churn_probability:.1%}"
        )

    result_col1, result_col2, result_col3, result_col4 = st.columns(4)

    result_col1.metric(
        "Churn probability",
        f"{churn_probability:.1%}",
    )

    result_col2.metric(
        "No-churn probability",
        f"{no_churn_probability:.1%}",
    )

    result_col3.metric(
        "Quarter revenue at risk",
        f"${revenue_at_risk:,.2f}",
    )

    result_col4.metric(
        "Retention priority",
        f"{retention_priority:,.2f}",
    )

    st.subheader("Recommended action")
    st.write(
        f"**{risk_level} risk:** {recommendation}"
    )

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Single Customer Prediction"],
)

if page == "Overview":
    overview_page()
else:
    prediction_page()
