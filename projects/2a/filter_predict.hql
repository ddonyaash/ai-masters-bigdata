-- Добавляем файлы проекта в кэш, чтобы узлы кластера их видели
ADD FILE projects/2a/predict.py;
ADD FILE projects/2a/model.py;
ADD FILE 2a.joblib;

INSERT INTO TABLE hw2_pred
SELECT TRANSFORM (id, if1, if2, if3, if4, if5, if6, if7, if8, if9, if10, if11, if12, if13, cf1, cf2, cf3, cf4, cf5, cf6, cf7, cf8, cf9, cf10, cf11, cf12, cf13, cf14, cf15, cf16, cf17, cf18, cf19, cf20, cf21, cf22, cf23, cf24, cf25, cf26)
USING 'python3 predict.py' AS (id INT, prediction DOUBLE)
FROM hw2_test
WHERE if1 > 20 AND if1 < 40;
