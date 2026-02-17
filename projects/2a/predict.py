#!/opt/conda/envs/dsenv/bin/python

import sys
import os
import logging
from joblib import load
import pandas as pd

# Добавляем путь, чтобы скрипт видел model.py в Hive
sys.path.append('.')
from model import fields_val

# Загружаем модель (имя файла должно быть 2a.joblib по условию)
model = load("2a.joblib")

# Читаем данные из stdin, которые передает Hive
read_opts = dict(
    sep='\t', 
    names=fields_val, 
    index_col=False, 
    header=None,
    na_values='\\N' # Hive передает NULL как \N
)

try:
    for df in pd.read_csv(sys.stdin, **read_opts, chunksize=1000):
        # Предсказание вероятности (второй столбец — вероятность клика)
        y_pred = model.predict_proba(df.iloc[:, 1:])[:, 1]
        
        # Вывод в формате: id <tab> prediction
        for i, p in zip(df.id, y_pred):
            print(f"{i}\t{p}")
except EOFError:
    pass
