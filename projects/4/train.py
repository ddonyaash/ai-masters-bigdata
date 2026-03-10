#!/opt/conda/envs/dsenv/bin/python
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

from model import pipeline

def main():
    train_path = sys.argv[1]
    model_path = sys.argv[2]

    schema = StructType([
        StructField("id", LongType(), True),
        StructField("overall", DoubleType(), True),
        StructField("reviewText", StringType(), True)
    ])

    train_df = spark.read.json(train_path, schema=schema) \
        .fillna({"reviewText": "missingreview"})

    pipeline_model = pipeline.fit(train_df)

    pipeline_model.write().overwrite().save(model_path)
    spark.stop()

if __name__ == "__main__":
    main()
