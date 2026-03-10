#!/opt/conda/envs/dsenv/bin/python
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline

tokenizer = Tokenizer(inputCol="reviewText", outputCol="words")
swr = StopWordsRemover(inputCol="words", outputCol="filtered_words")
hasher = HashingTF(numFeatures=1000, inputCol="filtered_words", outputCol="features")
lr = LinearRegression(featuresCol="features", labelCol="overall", regParam=0.1)

pipeline = Pipeline(stages=[tokenizer, swr, hasher, lr])
