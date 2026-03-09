#!/opt/conda/envs/dsenv/bin/python
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml import PipelineModel

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

path_to_saved_model = sys.argv[1]
path_to_test_dataset = sys.argv[2]
path_to_save_inference = sys.argv[3]

model = PipelineModel.load(path_to_saved_model)

test_schema = StructType([
    StructField("id", StringType()),
    StructField("reviewText", StringType())
])

test = spark.read.json(path_to_test_dataset, schema=test_schema)
test = test.fillna({'reviewText': ''})

predictions = model.transform(test)

predictions.select("id", "prediction").write.mode("overwrite").save(path_to_save_inference)

spark.stop()
