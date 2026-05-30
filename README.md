## End to end machine learning project
# Customer Churn Prediction System

An End-to-End Machine Learning project that predicts whether a telecom customer is likely to churn.

## Features

- Data Ingestion Pipeline
- Data Transformation Pipeline
- Hyperparameter Tuning
- Multiple Model Comparison
- Decision Tree
- Random Forest
- XGBoost
- Model Persistence
- Prediction Pipeline
- Flask Web Application
- Customer Churn Probability Score

---

## Project Structure

```text
MLPROJECT
│
├── artifacts
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── train.csv
│   ├── test.csv
│
├── notebook
│
├── src
│   ├── components
│   │   ├── data_ingestion.py
│   │   ├── data_transformer.py
│   │   ├── model_trainer.py
│   │
│   ├── pipeline
│   │   └── predict_pipeline.py
│   │
│   ├── logger.py
│   ├── exception.py
│   └── utils.py
│
├── templates
│   └── home.html
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Dataset

Telecom Customer Churn Dataset from Kaggle.

Target Variable:

```text
Churn
```

- 0 = Customer Stays
- 1 = Customer Leaves

---

## Machine Learning Workflow

1. Data Ingestion
2. Data Cleaning
3. Feature Engineering
4. Train-Test Split
5. Data Transformation
6. Hyperparameter Tuning
7. Model Training
8. Model Evaluation
9. Model Selection
10. Prediction Pipeline
11. Flask Deployment

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Flask
- HTML/CSS

---

## Run Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Train Model:

```bash
python -m src.components.data_ingestion
```

Run Flask App:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## Future Improvements

- Streamlit Dashboard
- Docker Deployment
- Cloud Deployment
- Customer Segmentation
- SHAP Explainability
- Real-Time Prediction API

