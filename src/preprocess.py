import pandas as pd
from pathlib import Path

def clean(path):
    raw_data_path = Path(path)
    df = pd.read_csv(raw_data_path)

    # print(df.info())    # nulls contained in person_emp_length, loan_int_rate
    # print(df.describe())

    # mark which entries are missing years of employment (could be self-employed, contractor, etc.)
    df['missing_employment'] = df['person_emp_length'].isnull().astype(int)
    # fill null employment with 0, 'missing_employment' column differentiates true 0 from filled 0.
    df['person_emp_length'] = df['person_emp_length'].fillna(value=0)
    # remove invalid ages (oldest living person was younger than 123)
    df = df[df['person_age'] <= 122]
    # remove cases where applicants claim to have been working since before they were 13 years old
    df = df[df['person_emp_length'] <= (df['person_age'] - 13)] 
    # Drop rows with missing loan_int_rate
    # too context-dependent to impute without introducing noise
    df.dropna(subset=["loan_int_rate"], inplace=True)
    # one-hot encode nominal variables
    df = pd.get_dummies(df, columns=['person_home_ownership', 'loan_intent', 'cb_person_default_on_file'],
                            drop_first=True, dtype=int)
    # map ordinal values (loan grade)
    grade_key = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7}
    df['loan_grade'] = df['loan_grade'].map(grade_key)
    print(df.describe())

    target = Path("data/processed/credit_risk_dataset_cleaned.csv")
    if target.exists():
        print("Overwriting existing dataset...")
    else:
        print("Saving new dataset...")

    # Write data into target location
    df.to_csv(target, index=False)

    # Show data locaation
    print("Path to cleaned dataset:", target)
    
    return target
