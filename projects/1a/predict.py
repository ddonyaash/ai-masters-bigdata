#!/opt/conda/envs/dsenv/bin/python

import sys, os
import logging
from joblib import load
import pandas as pd
import numpy as np

sys.path.append('.')
from model import fields_val

logging.basicConfig(level=logging.DEBUG)

# Загружаем 1a.joblib
model = load("1a.joblib")

read_opts = dict(
        sep='\t', names=fields_val, index_col=False, header=None,
        iterator=True, chunksize=100
)

for df in pd.read_csv(sys.stdin, **read_opts):
    # Предсказание на всех колонках кроме id
    y_pred = model.predict_proba(df.iloc[:, 1:])
    out = zip(df.id, y_pred[:, 1])
    print("\n".join(["{0}\t{1}".format(*i) for i in out]))
