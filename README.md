# 🏦 Bank Fraud Investigation Dashboard (Streamlit)

## 📌 Overview
This project is an interactive **Bank Fraud Investigation Dashboard** built using **Streamlit**.  
Users upload transaction data in CSV format, and the system automatically detects suspicious accounts using **Isolation Forest anomaly detection** and feature-based fraud scoring.

The dashboard provides:
- Fraud probability per account  
- Risk level classification (LOW / MEDIUM / HIGH)  
- Feature contribution analysis (explainable AI)  
- Fraud reason explanations like a real bank investigation tool  

---

## 🚀 How the System Works

1. User uploads a transaction CSV file  
2. System automatically detects the Account ID column  
3. Feature engineering is performed at account level  
4. Isolation Forest detects anomalous accounts  
5. Fraud probability is computed using transaction behavior  
6. Contribution score explains why an account is suspicious  
7. Final investigation report is generated and downloadable  

---

## 📂 Required CSV Columns

Your CSV file **must contain these columns**:

| Column Name | Description |
|-------------|-------------|
| `AccountID` | Unique bank account identifier |
| `Amount` | Transaction amount |
| `TransactionID` | Unique transaction ID |
| *(Optional)* Email, PhoneNumber, IPAddress, DeviceID | Customer metadata |

⚠️ If `AccountID` is missing, the system will stop execution.

---

## 🧠 Feature Engineering

The system aggregates transaction data per account:

- `total_amount` = Total transaction volume  
- `avg_amount` = Average transaction size  
- `unique_targets` = Number of unique transactions  

These features represent real-world fraud behavior patterns.

---

## 🤖 Fraud Detection Model

### 🔹 Isolation Forest
Isolation Forest is used for **unsupervised anomaly detection** to identify suspicious accounts without labeled fraud data.

---

### 🔹 Fraud Probability Scoring
Fraud probability is computed using:
- Anomaly flag  
- Transaction volume normalization  
- Average transaction amount behavior  

Output range: **0 to 1**

---

## 🔍 Explainable AI (Contribution Analysis)

The system calculates **feature contribution percentages** for each account:

- Total transaction volume contribution  
- Average transaction size contribution  
- Unique target behavior contribution  

This helps explain **why an account was flagged**, similar to bank investigation tools.

---

## 🚨 Risk Classification

| Fraud Probability | Risk Label |
|------------------|------------|
| > 0.75 | HIGH |
| 0.40 – 0.75 | MEDIUM |
| < 0.40 | LOW |

---

## 🖥️ Streamlit Dashboard Features

- CSV upload interface  
- Data preview  
- Fraud investigation table  
- Account search functionality  
- Downloadable fraud report  

---

## 🛠️ Tech Stack

- Python  
- Streamlit  
- Pandas  
- NumPy  
- Scikit-learn (Isolation Forest)  

---

## ▶️ Run the App Locally

```bash
pip install -r requirements.txt
streamlit run app.py


👤 Author
Vishv Pandya
