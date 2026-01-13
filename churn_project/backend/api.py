from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import os


app = FastAPI(title="Customer Churn API")

# Load trained pipeline
model = joblib.load("churn_model.joblib")

# =========================
# Schemas
# =========================
class CustomerInput(BaseModel):
    Gender: str
    Senior_Citizen: str
    Partner: str
    Dependents: str
    Tenure_Months: int
    Phone_Service: str
    Multiple_Lines: str
    Internet_Service: str
    Online_Security: str
    Online_Backup: str
    Device_Protection: str
    Tech_Support: str
    Streaming_TV: str
    Streaming_Movies: str
    Contract: str
    Paperless_Billing: str
    Payment_Method: str
    Monthly_Charges: float
    Total_Charges: float
    CLTV: float
    Latitude: float = 0.0
    Longitude: float = 0.0


# =========================
# Utility
# =========================
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={
        "Senior_Citizen": "Senior Citizen",
        "Tenure_Months": "Tenure Months",
        "Phone_Service": "Phone Service",
        "Multiple_Lines": "Multiple Lines",
        "Internet_Service": "Internet Service",
        "Online_Security": "Online Security",
        "Online_Backup": "Online Backup",
        "Device_Protection": "Device Protection",
        "Tech_Support": "Tech Support",
        "Streaming_TV": "Streaming TV",
        "Streaming_Movies": "Streaming Movies",
        "Paperless_Billing": "Paperless Billing",
        "Payment_Method": "Payment Method",
        "Monthly_Charges": "Monthly Charges",
        "Total_Charges": "Total Charges"
    })


# =========================
# Health Check
# =========================
@app.get("/")
def health():
    return {"status": "API running"}


# =========================
# Single Prediction
# =========================
@app.post("/predict")
def predict(customer: CustomerInput):
    df = pd.DataFrame([customer.model_dump()])
    df = preprocess(df)

    prob = float(model.predict_proba(df)[0][1])

    return {
        "churn_probability": round(prob, 3),
        "prediction": int(prob > 0.5)
    }


# =========================
# Overall Analytics
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(
    BASE_DIR,
    "model",
    "data",
    "cleaned_telco.xlsx"
)

@app.get("/analytics")
def analytics():
    # read Excel instead of CSV
    df = pd.read_excel(DATA_PATH)  
    df = preprocess(df)

    probs = model.predict_proba(df)[:, 1]

    high = (probs >= 0.7).sum()
    medium = ((probs >= 0.4) & (probs < 0.7)).sum()
    low = (probs < 0.4).sum()

    return {
        "total_customers": int(len(df)),
        "high_risk": int(high),
        "medium_risk": int(medium),
        "low_risk": int(low),
        "average_churn_probability": round(float(probs.mean()), 3),
        "critical_percentage": round(float(high / len(df) * 100), 2)
    }