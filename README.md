# 🛡️ Real-Time Credit Card Fraud Detection Engine

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)](https://streamlit.io)
[![MLflow](https://img.shields.io/badge/MLflow-Tracked-orange?logo=mlflow)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Business Context

Credit card fraud costs the global financial industry over **$32 billion USD annually**. This deployment-grade machine learning system classifies credit card transactions as **fraudulent or legitimate in real-time**, built specifically to address the extreme class imbalance found in real-world fraud datasets (only **0.172%** of transactions are fraudulent).

Designed with the Australian Banking and FinTech sector in mind, this project demonstrates a complete MLOps lifecycle — from raw data exploration through to containerised microservice deployment.

---

## Live Demo

| Service | URL |
|---------|-----|
| 🎨 **Streamlit Frontend** | [pranjay19-real-time-fraud-detection-frontendapp-qc5hbp.streamlit.app](https://pranjay19-real-time-fraud-detection-frontendapp-qc5hbp.streamlit.app) |
| ⚡ **FastAPI Backend (Swagger UI)** | [real-time-fraud-detection-9vl8.onrender.com/docs](https://real-time-fraud-detection-9vl8.onrender.com/docs) |

> **Note:** The backend runs on Render's free tier and may take ~30 seconds to wake up after inactivity. Simply retry the request after the cold start.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    User's Browser                        │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP
                        ▼
┌─────────────────────────────────────────────────────────┐
│         Streamlit Frontend  (Port 8501)                  │
│         frontend/app.py                                  │
│         • 30-feature input form                          │
│         • Fraud / Legitimate alert cards                 │
└───────────────────────┬─────────────────────────────────┘
                        │ POST /predict (JSON)
                        ▼
┌─────────────────────────────────────────────────────────┐
│         FastAPI Backend  (Port 8000)                     │
│         api/main.py                                      │
│         • Pydantic validation                            │
│         • RobustScaler preprocessing                     │
│         • Random Forest inference                        │
└─────────────────────────────────────────────────────────┘
```

### Tech Stack

| Layer | Technology |
|-------|-----------|
| **Data Engineering** | Python, Pandas, NumPy, Scikit-Learn |
| **Class Balancing** | Imbalanced-Learn (SMOTE) |
| **Experiment Tracking** | MLflow |
| **Backend API** | FastAPI + Uvicorn |
| **Frontend Dashboard** | Streamlit |
| **Containerisation** | Docker + Docker Compose |
| **Deployment** | Render (API) + Streamlit Cloud (UI) |

---

## Project Structure

```
real-time-fraud-detection/
├── api/
│   ├── main.py              # FastAPI application (predict endpoint)
│   ├── requirements.txt     # API dependencies
│   └── Dockerfile           # Backend container definition
├── frontend/
│   ├── app.py               # Streamlit dashboard
│   ├── requirements.txt     # Frontend dependencies
│   └── Dockerfile           # Frontend container definition
├── models/
│   ├── model.pkl            # Trained Random Forest classifier
│   └── scaler.pkl           # Fitted RobustScalers (Amount + Time)
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb   # EDA, scaling, train/test split
│   └── 02_model_training.ipynb              # SMOTE, RF training, MLflow logging
├── data/raw/                # Raw creditcard.csv (gitignored — download separately)
├── docker-compose.yml       # Orchestrates both services on a shared network
└── requirements.txt         # Full project dependencies
```

---

## 🚀 5-Minute Setup Guide (Docker)

You do **not** need to install Python or any dependencies natively. Docker handles everything.

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/pranjay19/real-time-fraud-detection.git
cd real-time-fraud-detection
```

**2. Launch both services with a single command**
```bash
docker-compose up --build
```

**3. Access the applications**

| App | URL |
|-----|-----|
| 🎨 Streamlit Dashboard | http://localhost:8501 |
| ⚡ FastAPI Swagger UI | http://localhost:8000/docs |

**4. Stop the services**
```bash
docker-compose down
```

---

## Running Locally (Without Docker)

**Prerequisites:** Python 3.10+, `pip`

```bash
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
source venv/bin/activate       # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset
# Place creditcard.csv from Kaggle into data/raw/

# 4. Run the notebooks (in order) to generate model.pkl and scaler.pkl
jupyter notebook

# 5. Start the FastAPI backend
cd api && python main.py

# 6. Start the Streamlit frontend (new terminal)
cd frontend && streamlit run app.py
```

---

## ML Pipeline Summary

| Phase | Day | Deliverable |
|-------|-----|-------------|
| **Phase 1: Data Science** | 1–2 | EDA, class imbalance visualisation, RobustScaling |
| | 3 | Stratified 80/20 train/test split with leakage prevention |
| | 4 | Baseline Logistic Regression — AUPRC performance floor |
| **Phase 2: MLOps** | 5 | MLflow experiment tracking initialisation |
| | 6 | SMOTE applied strictly to training set |
| | 7 | Random Forest trained on balanced data, logged to MLflow |
| | 8 | Best model exported as `model.pkl` via joblib |
| **Phase 3: Backend** | 9–11 | FastAPI scaffold → Pydantic schema → `/predict` endpoint |
| **Phase 4: Frontend** | 12–14 | Streamlit dashboard → sidebar UI → live API integration |
| **Phase 5: DevOps** | 15–17 | Dockerfiles → Docker Compose orchestration |
| | 18 | README, v1.0 release tag |

---

## Dataset

This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle (ULB Machine Learning Group).

- **284,807** transactions
- **492** fraudulent (0.172%)
- 28 PCA-anonymised features (V1–V28) + Time + Amount

> The raw CSV is **not included** in this repository (143 MB). Download it from Kaggle and place it at `data/raw/creditcard.csv`.

---

## License

This project is licensed under the MIT License.