#!/opt/conda/envs/dsenv/bin/python
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import *

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

from model import pipeline

path_to_dataset = sys.argv[1]
path_to_save_model = sys.argv[2]

schema = StructType([
    StructField("overall", DoubleType()),
    StructField("reviewText", StringType())
])

dataset = spark.read.json(path_to_dataset, schema=schema)
dataset = dataset.fillna({'reviewText': ''})

pipeline_model = pipeline.fit(dataset)

pipeline_model.write().overwrite().save(path_to_save_model)

spark.stop()
