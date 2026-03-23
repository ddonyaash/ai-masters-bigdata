from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

pyspark_python = "/opt/conda/envs/dsenv/bin/python"

# Создаем одну базовую переменную-шаблон. 
# Airflow сам подставит значение base_dir в момент исполнения каждой задачи.
BASE = "{{ dag_run.conf.get('base_dir', '/home/users/ddonyaash/ai-masters-bigdata/projects/6') }}"

with DAG(
    dag_id="ddonyaash_dag",
    start_date=datetime(2026, 3, 1),
    schedule=None,
    catchup=False,
) as dag:

    feature_eng_train_task = SparkSubmitOperator(
        task_id="feature_eng_train_task",
        application=f"{BASE}/feature_eng.py",
        spark_binary="/usr/bin/spark3-submit",
        application_args=["/datasets/amazon/amazon_extrasmall_train.json", "ddonyaash_train_out"],
        env_vars={"PYSPARK_PYTHON": pyspark_python},
        conf={"spark.driver.python": pyspark_python}
    )

    download_train_task = SparkSubmitOperator(
        task_id="download_train_task",
        application=f"{BASE}/train_download.py",
        spark_binary="/usr/bin/spark3-submit",
        application_args=["ddonyaash_train_out", f"{BASE}/ddonyaash_train_out_local"],
        env_vars={"PYSPARK_PYTHON": pyspark_python}
    )

    train_task = BashOperator(
        task_id="train_task",
        # Для BashOperator шаблоны работают идеально
        bash_command=f"{pyspark_python} {BASE}/train.py {BASE}/ddonyaash_train_out_local {BASE}/6.joblib"
    )

    model_sensor = FileSensor(
        task_id="model_sensor",
        filepath=f"{BASE}/6.joblib",
        poke_interval=10,
        timeout=120
    )

    feature_eng_test_task = SparkSubmitOperator(
        task_id="feature_eng_test_task",
        application=f"{BASE}/feature_eng.py",
        spark_binary="/usr/bin/spark3-submit",
        application_args=["/datasets/amazon/amazon_extrasmall_test.json", "ddonyaash_test_out"],
        env_vars={"PYSPARK_PYTHON": pyspark_python}
    )

    predict_task = SparkSubmitOperator(
        task_id="predict_task",
        application=f"{BASE}/predict.py",
        spark_binary="/usr/bin/spark3-submit",
        application_args=["ddonyaash_test_out", "ddonyaash_hw6_prediction", f"{BASE}/6.joblib"],
        env_vars={"PYSPARK_PYTHON": pyspark_python}
    )

    feature_eng_train_task >> download_train_task >> train_task >> model_sensor >> feature_eng_test_task >> predict_task
