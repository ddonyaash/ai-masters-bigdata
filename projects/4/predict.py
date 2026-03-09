#!/opt/conda/envs/dsenv/bin/python
import os, sys
import logging
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml import PipelineModel

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

logging.basicConfig(level=logging.INFO)

try:
    path_to_saved_model = sys.argv[1]
    path_to_test_dataset = sys.argv[2]
    path_to_save_inference = sys.argv[3]
except IndexError:
    logging.critical("Usage: predict.py <model_path> <test_data_path> <output_path>")
    sys.exit(1)

# Загружаем модель
model = PipelineModel.load(path_to_saved_model)

# Читаем тестовые данные (обязательно указываем ID и Текст)
test = spark.read.json(path_to_test_dataset)
test = test.fillna({'reviewText': '', 'summary': ''})

predictions = model.transform(test)

predictions.select("id", "prediction").write.mode("overwrite").save(path_to_save_inference)
logging.info(f"Predictions saved to {path_to_save_inference}")

spark.stop()
