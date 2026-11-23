import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)

CAT_COLS = ["gender","married","dependents","education","self_employed","property_area"]
NUM_COLS = ["applicant_income","coapplicant_income","loan_amount","loan_term","credit_history"]
FEATURES = CAT_COLS + NUM_COLS
TARGET = "loan_status"

def train_model(df):
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=42, stratify=y
    )
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    preprocessor = ColumnTransformer([("categorical",categorical,CAT_COLS),
                                     ("numeric",numeric,NUM_COLS)])
    pipeline = Pipeline([
        ("preprocessor",preprocessor),
        ("classifier",RandomForestClassifier(n_estimators=350,max_depth=8,
          min_samples_leaf=3,class_weight="balanced",random_state=42,n_jobs=-1))
    ])
    pipeline.fit(X_train,y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:,1]
    metrics = {
        "accuracy":float(accuracy_score(y_test,y_pred)),
        "precision":float(precision_score(y_test,y_pred,zero_division=0)),
        "recall":float(recall_score(y_test,y_pred,zero_division=0)),
        "f1":float(f1_score(y_test,y_pred,zero_division=0)),
        "roc_auc":float(roc_auc_score(y_test,y_prob)),
        "confusion_matrix":confusion_matrix(y_test,y_pred).tolist(),
        "classification_report":classification_report(y_test,y_pred,output_dict=True,zero_division=0)
    }
    return {"model":pipeline,"metrics":metrics,"feature_names":FEATURES}

def evaluate_model(bundle, df):
    return bundle["metrics"]

def get_model_bundle():
    return train_model
