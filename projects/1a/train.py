#!/opt/conda/envs/dsenv/bin/python

import os, sys
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from joblib import dump
import numpy as np

from model import model, fields

logging.basicConfig(level=logging.DEBUG)

try:
  proj_id = sys.argv[1] 
  train_path = sys.argv[2]
except:
  logging.critical("Need to pass both project_id and train dataset path")
  sys.exit(1)

read_table_opts = dict(sep="\t", names=fields, index_col=False)
df = pd.read_table(train_path, **read_table_opts)

X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:100000, 2:], df.iloc[:100000, 1], test_size=0.33, random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)
model_score = log_loss(y_test, y_pred[:, 1])
logging.info(f"model score: {model_score:.3f}")

dump(model, "{}.joblib".format(proj_id))
