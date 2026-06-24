# 📊 Telco Customer Churn Prediction

### End-to-End Machine Learning Project for Predicting Customer Churn in the Telecommunications Industry


**Using data-driven insights to identify customers at risk of churn and support business retention strategies.**

---

# 📌 Project Overview

Customer churn is one of the most significant challenges for subscription-based businesses. Losing existing customers directly impacts revenue, while acquiring new customers often costs substantially more.

This project develops a complete Machine Learning pipeline that:

* Cleans and preprocesses customer data
* Explores behavioral patterns through visual analytics
* Trains predictive models
* Evaluates model performance
* Predicts whether a customer is likely to churn
* Provides actionable business insights

The goal is not only to build a predictive model but also to demonstrate an end-to-end Data Science workflow from raw data to business decision-making.

---

# 🎯 Business Problem

**Question**

> Which customers are most likely to leave the telecom service?

**Business Objective**

* Reduce customer attrition
* Improve retention campaigns
* Optimize marketing resources
* Increase customer lifetime value

---

# 🏗 Project Architecture

```text
                        IBM Telco Dataset
                                │
                                ▼
                     Data Cleaning & Validation
                                │
                                ▼
                  Feature Engineering & Encoding
                                │
                                ▼
                  Exploratory Data Analysis (EDA)
                                │
                                ▼
                     Machine Learning Models
                                │
                                ▼
                      Model Evaluation Metrics
                                │
                                ▼
                     Churn Probability Prediction
                                │
                                ▼
                   Business Insights & Dashboard
```

---

# 📂 Dataset
The project uses the **IBM Telco Customer Churn Dataset**, containing over **7,000 customer records**.

# data source:
IBM Telco Customer Churn (11.1.3+) sample dataset, distributed via Kaggle mirror by Al Fath Terry (or Jack Chang).

## Features
### 👤 Demographic Information
Customer demographic characteristics and household composition.

Gender,Age,Senior Citizen,Married,Dependents,Number of Dependents
### 📍 Geographic Information
Customer location and regional attributes.

City,Zip Code,Latitude,Longitude,Population
### 🤝 Customer Acquisition & Referrals
Information related to customer acquisition channels and referral programs.

Referred a Friend,Number of Referrals,Offer
### 📄 Account Information
Core account and subscription details.

Tenure in Months,Contract,Payment Method,Paperless Billing
### ☎️ Phone Services
Customer phone service subscriptions and associated charges.

Phone Service,Multiple Lines,Avg Monthly Long Distance Charges,Total Long Distance Charges
### 🌐 Internet Services
Internet connectivity and value-added service features.

Internet Service,Internet Type,Online Security,Online Backup,Device Protection Plan,Premium Tech Support,Unlimited Data
### 🎬 Entertainment Services
Streaming and media-related subscriptions.

Streaming TV,Streaming Movies,Streaming Music
### 📊 Usage Metrics
Customer service consumption and usage patterns.

Avg Monthly GB Download
### 💰 Billing & Revenue
Customer billing history, charges, refunds, and revenue contribution.

Monthly Charge,Total Charges,Total Refunds,Total Extra Data Charges,Total Revenue

### 🎯 Target Variable
Churn Label

# ⚙ Tech Stack

| Category         | Technologies          
| ---------------- | --------------------- 
| Language         | Python                
| Data Analysis    | Pandas, NumPy         
| Visualization    | Matplotlib, Plotly    
| Machine Learning | Scikit-learn, XGBoost,LightBGM, CatBoost
| Development      | Jupyter Notebook      

---

# 🔬 Data Science Workflow

```text
            Raw Data
                │
                ▼
           Data Cleaning
                │
                ▼
       Missing Value Handling
                │
                ▼
          Feature Encoding
                │
                ▼
        Exploratory Analysis
                │
                ▼
           Model Training
                │
                ▼
       Performance Evaluation
                │
                ▼
            Prediction
```

---

# 📊 Exploratory Data Analysis

The analysis investigates several business questions:

* Which customers churn the most?
* Does contract type influence retention?
* Does tenure reduce churn probability?
* How do monthly charges affect churn?
* Which services increase customer loyalty?

Example visualizations:

```
assets/
│
├── churn_distribution.png
├── tenure_analysis.png
├── contract_type.png
├── monthly_charges.png
└── correlation_heatmap.png
```

> Replace these placeholders with screenshots from your project for a stronger portfolio presentation.

---

# 🤖 Machine Learning

## Data Preprocessing

* Missing value handling
* Categorical encoding
* Feature engineering
* Train/Test split

---

## Models

* Support Vector Machine (SVM)
* XGBoost
* Additional baseline models (if applicable)

---

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

---

# 📈 Results

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | XX.XX% |
| Precision | XX.XX% |
| Recall    | XX.XX% |
| F1 Score  | XX.XX  |
| ROC-AUC   | XX.XX  |

> Replace the placeholder values above with your actual evaluation metrics.

---

# 🎯 Example Prediction

## Customer Profile

| Feature          | Value          |
| ---------------- | -------------- |
| Contract         | Month-to-month |
| Tenure           | 4 Months       |
| Internet Service | Fiber Optic    |
| Monthly Charges  | $85            |
| Tech Support     | No             |

↓

```text
Prediction Result

Risk Score

████████████░░ 82%

Classification

⚠ Likely to Churn
```

---

# 💼 Business Impact

The model can support decision-making by helping businesses:

* Identify high-risk customers
* Launch targeted retention campaigns
* Reduce customer acquisition costs
* Increase long-term profitability
* Improve customer satisfaction

---

# 📁 Repository Structure

```text
Telco-Customer-Churn/

├── data/
├── notebooks/
├── src/
├── assets/
│
├── app.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/DHSERIES/Telco-Customer-Churn.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

---

# 📷 Screenshots

## Dashboard

```text
assets/dashboard.png
```

## Model Performance

```text
assets/model_results.png
```

## Exploratory Analysis

```text
assets/eda_overview.png
```

> Add actual screenshots to the `assets/` folder and update the README to display them using Markdown images.

---

# 🔮 Future Improvements

* Hyperparameter optimization
* SHAP/LIME explainability
* REST API deployment
* Docker containerization
* Cloud deployment
* CI/CD pipeline
* Automated model retraining

---

# 🧠 Skills Demonstrated

* Data Cleaning
* Exploratory Data Analysis
* Feature Engineering
* Statistical Analysis
* Machine Learning
* Model Evaluation
* Data Visualization
* Dashboard Development
* Business Analytics

---

# ⭐ Project Highlights

* End-to-end Data Science workflow
* Business-oriented problem solving
* Interactive analytics and visualization
* Predictive Machine Learning model
* Portfolio-ready implementation

---

## License

This project is released under the MIT License and is intended for educational and portfolio purposes.


