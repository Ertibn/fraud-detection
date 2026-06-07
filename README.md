# Fraud Detection for E-commerce and Bank Transactions

**Organization**: Adey Innovations Inc.  
**Project**: Unified Fraud Detection System  
**Challenge**: Week 5&6 - Artificial Intelligence Mastery

---

## Overview

This project develops machine learning models to detect fraudulent transactions across two distinct data streams:

1. **E-commerce Transactions** - Rich contextual features (user, device, behavior, geolocation)
2. **Bank Credit Card Transactions** - Anonymized PCA-transformed features (privacy-preserving)

Both datasets are highly imbalanced (fraud << legitimate), requiring specialized resampling and evaluation metrics.

---

## Project Structure

```
fraud-detection/
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml
├── data/
│   ├── raw/                    # Original datasets (add to .gitignore)
│   └── processed/              # Cleaned and feature-engineered data
├── notebooks/
│   ├── __init__.py
│   ├── eda-fraud-data.ipynb
│   ├── eda-creditcard.ipynb
│   ├── feature-engineering.ipynb
│   ├── modeling.ipynb
│   ├── shap-explainability.ipynb
│   └── README.md
├── src/
│   └── __init__.py
├── tests/
│   └── __init__.py
├── models/                     # Saved model artifacts
├── scripts/
│   ├── __init__.py
│   └── README.md
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Task Overview

### Task 1: Data Analysis and Preprocessing (Due: 07 Jun 2026)

**Objectives:**
- Clean and preprocess both datasets
- Perform exploratory data analysis (EDA)
- Integrate geolocation data (IP-to-Country mapping)
- Engineer behavioral and temporal features
- Handle severe class imbalance

**Deliverables:**
- Cleaned datasets
- EDA report with visualizations
- Feature engineering documentation
- Resampling strategy justification

### Task 2: Model Building and Training (Due: 14 Jun 2026)

**Objectives:**
- Build baseline model (Logistic Regression)
- Train ensemble models (Random Forest, XGBoost, or LightGBM)
- Evaluate using appropriate metrics (AUC-PR, F1-Score)
- Select best model with justification
- Apply stratified cross-validation

**Deliverables:**
- Trained models
- Model comparison metrics table
- Model selection writeup

### Task 3: Model Explainability (Due: 16 Jun 2026)

**Objectives:**
- Extract feature importance from ensemble model
- Generate SHAP plots (summary, force plots)
- Interpret findings
- Translate to actionable business recommendations

**Deliverables:**
- Feature importance visualizations
- SHAP analysis with interpretation
- Business recommendations with justification

---

## Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/[username]/fraud-detection.git
cd fraud-detection
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Data

Place the three datasets in `data/raw/`:
- `Fraud_Data.csv` - E-commerce transactions
- `IpAddress_to_Country.csv` - IP-to-Country mapping
- `creditcard.csv` - Bank credit card transactions

### 5. Run Notebooks

```bash
jupyter notebook
```

---

## Key Concepts

### Class Imbalance Handling

Both datasets have severe class imbalance:
- E-commerce: ~3.3% fraud rate
- Bank credit cards: ~0.17% fraud rate

**Solutions Applied:**
- SMOTE (Synthetic Minority Over-sampling)
- Undersampling of majority class
- Class-weighted loss functions
- Stratified train-test split

### Evaluation Metrics

For imbalanced data, **accuracy is misleading**. Primary metrics:

- **AUC-PR (Area Under Precision-Recall Curve)**: Best for imbalanced data
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: True/false positives and negatives
- **ROC-AUC**: Traditional metric, still useful

### Feature Engineering

Key features engineered from raw data:

- **Transaction Velocity**: # transactions in time windows (1h, 24h, 7d)
- **Time Features**: hour_of_day, day_of_week, is_weekend
- **Time-Since-Signup**: Duration from signup to purchase (captures new account risk)
- **Geolocation**: Country from IP address (detects cross-border fraud)

### Model Explainability (SHAP)

SHAP (SHapley Additive exPlanations) provides:
- **Global Importance**: Which features matter most across all predictions
- **Force Plots**: Why a specific prediction was made for an individual transaction
- **Waterfall Plots**: Cumulative contribution of each feature

---

## Timeline

| Date | Milestone | Status |
|------|-----------|--------|
| 04 Jun | Challenge Discussion | - |
| 05 Jun | Tutorials (Geolocation, Imbalanced Data) | - |
| 07 Jun | **Interim-1 Submission** (Task 1) | Pending |
| 08 Jun | Tutorials (Modeling, SHAP) | - |
| 14 Jun | **Interim-2 Submission** (Tasks 1-2) | Pending |
| 16 Jun | **Final Submission** (All Tasks) | Pending |

---

## Technologies

- **Data Processing**: pandas, numpy
- **Feature Engineering**: scikit-learn, imbalanced-learn
- **Modeling**: scikit-learn, XGBoost, LightGBM
- **Evaluation**: sklearn metrics, confusion_matrix
- **Explainability**: SHAP
- **Visualization**: matplotlib, seaborn, plotly
- **Testing**: pytest, pytest-cov
- **CI/CD**: GitHub Actions
- **Code Quality**: black, flake8

---

## References

### Fraud Detection
- [Kaggle: Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Kaggle: IEEE Fraud Detection Competition](https://www.kaggle.com/competitions/ieee-fraud-detection)
- [ComplyAdvantage: What is Fraud Detection?](https://complyadvantage.com/knowledgebase/fraud-detection/)

### Handling Imbalanced Data
- [imbalanced-learn Documentation](https://imbalanced-learn.org/)
- [Analytics Vidhya: 10 Techniques to Handle Class Imbalance](https://www.analyticsvidhya.com/)
- [SMOTE Paper](https://arxiv.org/abs/1609.02287)

### Evaluation Metrics
- [scikit-learn: Precision-Recall Curves](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)
- [Machine Learning Mastery: ROC vs Precision-Recall Curves](https://machinelearningmastery.com/)

### Model Explainability
- [SHAP Documentation](https://shap.readthedocs.io/)
- [DataCamp: Explainable AI](https://www.datacamp.com/)

---

## Team

**Tutors**: Kerod, Mahbubah, Feven  
**Organization**: 10 Academy  
**Date**: 4 Jun – 16 Jun 2026

---

## License

This project is part of the 10 Academy AI Mastery Program.

