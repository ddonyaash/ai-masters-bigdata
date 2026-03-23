#!/opt/conda/envs/dsenv/bin/python
import sys
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression

path_in, path_out = sys.argv[1], sys.argv[2]

df = pd.read_csv(path_in)
X = df.drop('label', axis=1)
y = df['label']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

dump(model, path_out)
