#!/opt/conda/envs/dsenv/bin/python
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array

path_in, path_out = sys.argv[1], sys.argv[2]

spark = SparkSession.builder.getOrCreate()

schema = StructType([
    StructField("id", LongType()),
    StructField("label", DoubleType()),
    StructField("reviewText", StringType()),
])

data = spark.read.json(path_in, schema=schema).fillna({'reviewText': ''})

tokenizer = RegexTokenizer(inputCol="reviewText", outputCol="words", pattern="\\W")
stop_remover = StopWordsRemover(inputCol="words", outputCol="filtered")
hashing_tf = HashingTF(numFeatures=100, inputCol="filtered", outputCol="vector")

pipeline = Pipeline(stages=[tokenizer, stop_remover, hashing_tf])
model = pipeline.fit(data)
df = model.transform(data)

df = df.withColumn("features", vector_to_array("vector"))

if "test" in path_in:
    df.select("id", "features").write.mode("overwrite").parquet(path_out)
else:
    df.select("label", "features").write.mode("overwrite").parquet(path_out)
