#!/opt/conda/envs/dsenv/bin/python
import os, sys
import logging
from pyspark.sql import SparkSession
from pyspark.sql.types import *

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

from model import pipeline

logging.basicConfig(level=logging.INFO)

try:
    path_to_dataset = sys.argv[1]
    path_to_save_model = sys.argv[2]
except IndexError:
    logging.critical("Usage: train.py <dataset_path> <model_save_path>")
    sys.exit(1)

schema = StructType([
    StructField("overall", DoubleType()),
    StructField("reviewText", StringType()),
    StructField("summary", StringType())
])

dataset = spark.read.json(path_to_dataset, schema=schema)

dataset = dataset.fillna({'reviewText': '', 'summary': ''})

dataset.cache()

logging.info("Starting model training...")
pipeline_model = pipeline.fit(dataset)

pipeline_model.write().overwrite().save(path_to_save_model)
logging.info(f"Model saved to {path_to_save_model}")

spark.stop()
