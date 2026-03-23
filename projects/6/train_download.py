#!/opt/conda/envs/dsenv/bin/python
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

path_in, path_out = sys.argv[1], sys.argv[2]
spark = SparkSession.builder.getOrCreate()

df = spark.read.parquet(path_in)
num_features = 100
df_flat = df.select(['label'] + [col("features")[i].alias(f"f_{i}") for i in range(num_features)])

df_flat.toPandas().to_csv(path_out, index=False)
