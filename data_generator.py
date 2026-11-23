import numpy as np
import pandas as pd

def generate_dataset(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    gender = rng.choice(["Male", "Female", "Other"], n, p=[.48,.49,.03])
    married = rng.choice(["Yes","No"], n, p=[.62,.38])
    dependents = rng.choice(["0","1","2","3+"], n, p=[.55,.18,.17,.10])
    education = rng.choice(["Graduate","Not Graduate"], n, p=[.78,.22])
    self_employed = rng.choice(["No","Yes"], n, p=[.86,.14])
    applicant_income = np.clip(rng.lognormal(mean=10.55, sigma=.55, size=n), 8000, 250000).round()
    coapplicant_income = np.where(married=="Yes", rng.lognormal(9.6,.65,n), 0).round()
    loan_amount = np.clip(rng.normal(160,70,n), 20, 500).round()
    loan_term = rng.choice([120,180,240,300,360], n, p=[.08,.08,.16,.18,.50])
    credit_history = rng.choice([0,1], n, p=[.18,.82])
    property_area = rng.choice(["Urban","Semiurban","Rural"], n, p=[.34,.42,.24])

    # Synthetic target: illustrative relationships only, not a real bank policy.
    income_score = np.log1p(applicant_income + coapplicant_income)
    logit = (-1.0 + .85*credit_history + .35*(education=="Graduate")
             + .25*(married=="Yes") + .18*(property_area=="Semiurban")
             + .10*(property_area=="Urban") + .24*(income_score-10.5)
             - .008*(loan_amount-150) - .0005*(loan_amount/(income_score+1))
             + .08*(self_employed=="No") + rng.normal(0,.65,n))
    probability = 1/(1+np.exp(-logit))
    loan_status = rng.binomial(1, probability)
    return pd.DataFrame({
        "gender":gender, "married":married, "dependents":dependents,
        "education":education, "self_employed":self_employed,
        "applicant_income":applicant_income, "coapplicant_income":coapplicant_income,
        "loan_amount":loan_amount, "loan_term":loan_term,
        "credit_history":credit_history, "property_area":property_area,
        "loan_status":loan_status
    })
