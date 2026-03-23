#!/opt/conda/envs/dsenv/bin/python
import sys
import pandas as pd
from joblib import load
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import DoubleType

if len(sys.argv) < 4:
    sys.exit(1)

test_in, pred_out, model_path = sys.argv[1], sys.argv[2], sys.argv[3]
spark = SparkSession.builder.getOrCreate()

# Загружаем модель
model = load(model_path)
sc_model = spark.sparkContext.broadcast(model)

@pandas_udf(DoubleType())
def predict_udf(*cols):
    # Создаем DataFrame из входящих колонок
    X = pd.concat(cols, axis=1)
    
    # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: 
    # Назначаем имена колонок f_0, f_1 ... f_99, чтобы sklearn не ругался
    X.columns = [f"f_{i}" for i in range(len(cols))]
    
    # Также можно использовать X.columns = sc_model.value.feature_names_in_
    # если вы уверены, что модель сохранила их.
    
    return pd.Series(sc_model.value.predict(X))

# Читаем данные
df = spark.read.parquet(test_in)

# Раскрываем массив признаков в отдельные колонки
# Убедитесь, что размерность (range) совпадает с тем, что было при обучении
feature_cols = [col("features")[i].alias(f"f_{i}") for i in range(100)]
df_test = df.select(['id'] + feature_cols)

# Список имен колонок для передачи в UDF
input_cols = [f"f_{i}" for i in range(100)]
df_test = df_test.withColumn("prediction", predict_udf(*input_cols))

# Сохраняем результат
df_test.select("id", "prediction").write.mode("overwrite").csv(pred_out, header=False)
