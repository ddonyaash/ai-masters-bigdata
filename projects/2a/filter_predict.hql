-- Добавляем файлы проекта в кэш, чтобы узлы кластера их видели
ADD FILE projects/2a/predict.py;
ADD FILE projects/2a/model.py;
ADD FILE 2a.joblib;

INSERT INTO TABLE hw2_pred
select transform(*) USING '/opt/conda/envs/dsenv/bin/python predict.py'  AS (id INT, prediction DOUBLE)
FROM hw2_test
where if1 is not null and if1 > 20 and if1 < 40;
