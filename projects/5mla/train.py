#!/usr/bin/env python3
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--model_param1', type=int, default=100)
    args = parser.parse_args()

    numeric_features = ["if"+str(i) for i in range(1,14)]
    categorical_features = ["cf"+str(i) for i in range(1,27)]
    fields = ["id", "label"] + numeric_features + categorical_features

    df = pd.read_table(args.train_path, sep="\t", names=fields, index_col=False)
    
    X = df.drop(['id', 'label'], axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('logregression', LogisticRegression(solver='liblinear', max_iter=args.model_param1))
    ])

    with mlflow.start_run():
        mlflow.log_param("max_iter", args.model_param1)
        mlflow.log_param("train_path", args.train_path)

        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_test)
        loss = log_loss(y_test, y_pred_proba)
        mlflow.log_metric("log_loss", loss)

        mlflow.sklearn.log_model(model, artifact_path="model")
        
        print(f"Model trained with log_loss: {loss}")

if __name__ == "__main__":
    main()
