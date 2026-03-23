#!/opt/conda/envs/dsenv/bin/python
import sys
import pandas as pd
from joblib import load
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import *

test_in, pred_out, model_path = sys.argv[1], sys.argv[2], sys.argv[3]
spark = SparkSession.builder.getOrCreate()

model = load(model_path)
sc_model = spark.sparkContext.broadcast(model)

@pandas_udf(DoubleType())
def predict_udf(*cols):
    X = pd.concat(cols, axis=1)
    return pd.Series(sc_model.value.predict(X))

df = spark.read.parquet(test_in)
feature_cols = [col("features")[i].alias(f"f_{i}") for i in range(100)]
df_test = df.select(['id'] + feature_cols)

input_cols = [f"f_{i}" for i in range(100)]
df_test = df_test.withColumn("prediction", predict_udf(*input_cols))

df_test.select("id", "prediction").write.mode("overwrite").csv(pred_out, header=False)
