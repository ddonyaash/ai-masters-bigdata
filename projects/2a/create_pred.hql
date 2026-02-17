CREATE TABLE hw2_pred (
    id INT,
    prediction DOUBLE
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
LOCATION '${env:USER}_hw2_pred';
