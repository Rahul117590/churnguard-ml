# ChurnGuard ML 🎯

An end-to-end machine learning pipeline that predicts telecom customer churn using **XGBoost**, with a modular architecture, experiment tracking via **DVC**, a **Flask** web application for real-time predictions, and **Docker** for containerized deployment.

---

## 📌 Overview

Customer churn — when a customer stops using a company's service — is one of the most critical problems telecom companies face. This project builds a complete, production-style ML pipeline that:

- Ingests raw customer data
- Cleans and preprocesses it
- Engineers features (encoding + scaling)
- Trains an XGBoost classifier
- Evaluates the model with multiple metrics
- Serves predictions through a Flask web interface
- Runs anywhere via Docker

---

## 📊 Dataset

**Telco Customer Churn** — [Kaggle Link](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

- ~7,043 rows, 21 columns
- Target column: `Churn` (Yes/No)
- Features include customer demographics, account information, subscribed services, and billing details

---

## 🏗️ Project Architecture

The pipeline follows a modular, five-stage structure (plus deployment):

```
Data Ingestion → Data Preprocessing → Feature Engineering → Model Training → Model Evaluation → Flask Deployment
```

### Folder Structure

```
churnguard-ml/
│
├── data/
│   ├── raw/            # Raw ingested data (train/test split)
│   ├── processed/       # Cleaned data (duplicates removed, missing values handled, target encoded)
│   └── final/            # Feature-engineered data (encoded + scaled)
│
├── models/
│   ├── encoder.pkl        # Saved OneHotEncoder
│   ├── scaler.pkl         # Saved StandardScaler
│   └── xgboost_model.pkl  # Final trained XGBoost model
│
├── src/
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── model_evaluation.py
│
├── reports/
│   ├── metrics.json
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
│
├── templates/
│   ├── index.html       # Input form
│   └── result.html      # Prediction result page
│
├── static/
│   └── style.css
│
├── logs/                 # Logs for each pipeline stage
├── app.py                # Flask application
├── dvc.yaml               # DVC pipeline definition
├── params.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| ML Model | XGBoost |
| Data Handling | Pandas, NumPy |
| Preprocessing | Scikit-learn (OneHotEncoder, StandardScaler) |
| Pipeline Tracking | DVC |
| Web Framework | Flask |
| Frontend | HTML, CSS |
| Deployment | Docker |
| Experiment Comparison | Logistic Regression, Decision Tree, Random Forest, AdaBoost, XGBoost |

---

## 🔄 Pipeline Stages

1. **Data Ingestion** — Loads raw data, drops unwanted columns, splits into train/test sets
2. **Data Preprocessing** — Removes duplicates, handles missing values (`TotalCharges`), encodes the target column
3. **Feature Engineering** — One-Hot Encodes categorical features, scales numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`)
4. **Model Training** — Trains an XGBoost classifier on the final processed data
5. **Model Evaluation** — Evaluates using Accuracy, Precision, Recall, F1-score, and ROC-AUC; generates confusion matrix, ROC curve, and feature importance plots

---

## 🧪 Model Experimentation

Alongside the XGBoost-only production pipeline, model comparisons were run separately (`experiment.py`) across:

- Logistic Regression
- Decision Tree
- Random Forest
- AdaBoost
- XGBoost

XGBoost consistently performed best on ROC-AUC and F1-score for the minority (churned) class, owing to its built-in regularization and `scale_pos_weight` handling of class imbalance.

---

## 🚀 Running the Project

### 1. Clone the repository
```bash
git clone https://github.com/Rahul117590/churnguard-ml.git
cd churnguard-ml
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the DVC pipeline
```bash
dvc repro
```

### 4. Run the Flask app
```bash
python app.py
```
Visit `http://127.0.0.1:5000` in your browser.

### 5. Run with Docker
```bash
docker build -t churn-app .
docker run -p 5000:5000 churn-app
```

---

## 📈 Results

| Metric | Score |
|---|---|
| Accuracy | *TBD* |
| Precision | *TBD* |
| Recall | *TBD* |
| F1-Score | *TBD* |
| ROC-AUC | *TBD* |

*(Update this table once model_evaluation.py generates final metrics.)*

---

## 👤 Author

**Rahul**
GitHub: [Rahul117590](https://github.com/Rahul117590)

---

## 📄 License

This project is open-source and available under the MIT License.