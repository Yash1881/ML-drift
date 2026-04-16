import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import joblib
import os

def main():
    print("Loading data...")
    df = pd.read_csv('data/telco_churn.csv')
    
    # Preprocess
    # TotalCharges is object in the csv, handle empty strings
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    
    # Features mentioned specifically
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
    
    # We'll use all other columns as categorical (except customerID and target)
    target = 'Churn'
    categorical_features = [col for col in df.columns if col not in numeric_features + [target, 'customerID']]

    X = df.drop(columns=[target, 'customerID'])
    y = df[target].apply(lambda x: 1 if x == 'Yes' else 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Building model pipeline...")
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    print("Training model...")
    model.fit(X_train, y_train)
    
    # Ensure baseline directory exists
    os.makedirs('baseline', exist_ok=True)
    
    print("Saving model...")
    joblib.dump(model, 'baseline/model.pkl')
    
    print("Extracting baseline statistics...")
    baseline_stats = {}
    
    for col in numeric_features:
        col_data = X_train[col].values
        baseline_stats[col] = {
            'mean': float(np.mean(col_data)),
            'std': float(np.std(col_data)),
            'min': float(np.min(col_data)),
            'max': float(np.max(col_data)),
            'values': col_data  # saving full array as requested
        }
        
    joblib.dump(baseline_stats, 'baseline/baseline_stats.pkl')
    
    print("Extracting baseline prediction probability distribution...")
    # use prediction probabilities for the positive class (i.e. Churn=1)
    baseline_preds = model.predict_proba(X_train)[:, 1]
    joblib.dump(baseline_preds, 'baseline/baseline_predictions.pkl')
    
    print("Baseline generation complete. Objects saved to baseline/ directory.")

if __name__ == "__main__":
    main()
