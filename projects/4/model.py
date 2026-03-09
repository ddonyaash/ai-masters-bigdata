#!/opt/conda/envs/dsenv/bin/python
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline

tokenizer = RegexTokenizer(inputCol="reviewText", outputCol="words", pattern="\\W")

swr = StopWordsRemover(inputCol="words", outputCol="filtered_words")

hasher = HashingTF(numFeatures=1000, inputCol="filtered_words", outputCol="features")

lr = LinearRegression(featuresCol="features", labelCol="overall", regParam=0.1, elasticNetParam=0.5)

pipeline = Pipeline(stages=[tokenizer, swr, hasher, lr])
