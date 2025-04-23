import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

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
    # print(df.describe())

    target = Path("data/processed/credit_risk_dataset_cleaned.csv")
    if target.exists():
        print("Overwriting existing dataset...")
    else:
        print("Saving new dataset...")

    # Write data into target location
    df.to_csv(target, index=False)

    # Show data location
    print("Path to cleaned dataset:", target)
    
    return df

def split_data(df):
    # split the data into stratified train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=2025, stratify=df["loan_status"])
    
    # save paths for train and test sets
    target_train = Path("data/processed/train_data.csv")
    target_test = Path("data/processed/test_data.csv")

    if target_train.exists():
        print("Overwriting training dataset...")
    else:
        print("Saving new training dataset...")

    if target_test.exists():
        print("Overwriting test dataset...")
    else:
        print("Saving new test dataset...")

    # Write data into target locations
    train_df.to_csv(target_train, index=False)
    test_df.to_csv(target_test, index=False)

    # Show data locaations
    print("Path to training dataset:", target_train)
    print("Path to test dataset:", target_test)

    # split into x and y for each subset
    x_train = train_df.drop(columns='loan_status')
    y_train = train_df['loan_status']
    x_test = test_df.drop(columns='loan_status')
    y_test = test_df['loan_status']

    # verify splits
    print(y_train.value_counts(normalize=True))
    print(y_test.value_counts(normalize=True))

    # test identical number of features and entries
    assert x_train.shape[1] == x_test.shape[1], "Mismatch in feature columns between train and test!"
    assert len(y_train) == len(x_train), "Mismatch between features and labels in training set!"
    assert len(y_test) == len(x_test), "Mismatch between features and labels in test set!"

    return x_train, y_train, x_test, y_test
