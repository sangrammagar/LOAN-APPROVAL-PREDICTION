# 💳 Loan Approval Prediction | LoanLens

A portfolio-ready Data Science application that explores loan applications and predicts approval probability using a Random Forest machine-learning model. The app includes a polished Streamlit dashboard, interactive analytics, and model evaluation.

> **Important:** 
- Educational Purpose only
-The included data is synthetic and created for demonstration and learning. 
-This app is not a real bank underwriting system and must not be used to make financial or lending decisions.

## STREAMLIT LINK :
                   https://loan-approval-prediction-ftag7xvmpnakntsmze83ap.streamlit.app/

## Features
- Executive dashboard with KPI cards
- Loan approval prediction from applicant inputs
- Interactive Plotly analytics
- Model metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- Confusion matrix and feature importance
- Scikit-learn preprocessing pipeline
- Reproducible synthetic dataset generator

## Tech Stack
Python • Pandas • NumPy • Scikit-learn • Plotly • Streamlit

## Run locally

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/loan-approval-prediction.git
cd loan-approval-prediction

# 2. Create virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train model (optional; app auto-trains if model file is missing)
python train.py

# 5. Start Streamlit
streamlit run app.py
```

## Project structure

```text
loan-approval-prediction/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── loan_applications_demo.csv  # created by train.py
├── models/
│   └── loan_model.pkl             # created by train.py
└── src/
    ├── data_generator.py
    └── model_utils.py
```

## Model workflow
1. Generate demo data.
2. Split into train/test sets with stratification.
3. Impute missing values.
4. Encode categorical variables with OneHotEncoder.
5. Scale numerical fields.
6. Train Random Forest classifier.
7. Evaluate on a held-out test set.
8. Use the fitted pipeline for probability predictions.

## Disclaimer
The synthetic labels reflect a simplified, randomly generated relationship between applicant attributes and approval outcomes. Real-world lending requires validated data, regulatory compliance, explainability, fairness testing, privacy protections, and human review.
