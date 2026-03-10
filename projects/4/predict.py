#!/opt/conda/envs/dsenv/bin/python
import sys
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

def main():
    model_path = sys.argv[1]
    test_path = sys.argv[2]
    prediction_path = sys.argv[3]

    schema = StructType([
        StructField("id", LongType(), True),
        StructField("reviewText", StringType(), True)
    ])

    model = PipelineModel.load(model_path)

    test_df = spark.read.json(test_path, schema=schema) \
        .fillna({"reviewText": "missingreview"})

    predictions = model.transform(test_df)

    predictions.select("id", "prediction") \
        .coalesce(1) \
        .write.mode("overwrite") \
        .csv(prediction_path)

    spark.stop()

if __name__ == "__main__":
    main()
