from fastapi import FastAPI
from pydantic import BaseModel
from inference import predict

app = FastAPI(title="Customer Churn API")

class Customer(BaseModel):

    Age: float
    Number_of_Dependents: float

    Zip_Code: float
    Latitude: float
    Longitude: float
    Population: float

    Number_of_Referrals: float
    Tenure_in_Months: float

    Avg_Monthly_Long_Distance_Charges: float
    Avg_Monthly_GB_Download: float

    Monthly_Charge: float
    Total_Charges: float
    Total_Refunds: float
    Total_Extra_Data_Charges: float
    Total_Long_Distance_Charges: float
    Total_Revenue: float

    Gender: str
    Under_30: str
    Senior_Citizen: str
    Married: str
    Dependents: str
    City: str
    Referred_a_Friend: str
    Offer: str
    Phone_Service: str
    Multiple_Lines: str
    Internet_Service: str
    Internet_Type: str
    Online_Security: str
    Online_Backup: str
    Device_Protection_Plan: str
    Premium_Tech_Support: str
    Streaming_TV: str
    Streaming_Movies: str
    Streaming_Music: str
    Unlimited_Data: str
    Contract: str
    Paperless_Billing: str
    Payment_Method: str


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API"}


@app.post("/predict")
def predict_customer(customer: Customer):

    data = customer.model_dump()  
    # convert names back to original dataset names

    mapping = {
        "Number_of_Dependents":"Number of Dependents",
        "Zip_Code":"Zip Code",
        "Number_of_Referrals":"Number of Referrals",
        "Tenure_in_Months":"Tenure in Months",
        "Avg_Monthly_Long_Distance_Charges":"Avg Monthly Long Distance Charges",
        "Avg_Monthly_GB_Download":"Avg Monthly GB Download",
        "Monthly_Charge":"Monthly Charge",
        "Total_Charges":"Total Charges",
        "Total_Refunds":"Total Refunds",
        "Total_Extra_Data_Charges":"Total Extra Data Charges",
        "Total_Long_Distance_Charges":"Total Long Distance Charges",
        "Total_Revenue":"Total Revenue",
        "Under_30":"Under 30",
        "Senior_Citizen":"Senior Citizen",
        "Referred_a_Friend":"Referred a Friend",
        "Phone_Service":"Phone Service",
        "Multiple_Lines":"Multiple Lines",
        "Internet_Service":"Internet Service",
        "Internet_Type":"Internet Type",
        "Online_Security":"Online Security",
        "Online_Backup":"Online Backup",
        "Device_Protection_Plan":"Device Protection Plan",
        "Premium_Tech_Support":"Premium Tech Support",
        "Streaming_TV":"Streaming TV",
        "Streaming_Movies":"Streaming Movies",
        "Streaming_Music":"Streaming Music",
        "Unlimited_Data":"Unlimited Data",
        "Paperless_Billing":"Paperless Billing",
        "Payment_Method":"Payment Method"
    }

    formatted = {}

    for k, v in data.items():
        formatted[mapping.get(k, k)] = v

    return predict(formatted)
