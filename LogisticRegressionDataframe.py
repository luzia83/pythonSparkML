# File name: LogisticRegressionDataframe.py
# Author: Antonio J. Nebro
# Date created: 11/03/2020
# Python Version: 3.8.3
# Spark version: 3.0.0
# Brief description: Example of using decision trees with PySpark

from pyspark.ml.classification import DecisionTreeClassifier, MultilayerPerceptronClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession

if __name__ == "__main__":

    spark_session = SparkSession\
        .builder\
        .appName("Spark ML logistic regression example")\
        .master("local[4]")\
        .getOrCreate()

    data_frame = spark_session\
        .read\
        .format("libsvm")\
        .load("data/iris.scale.txt")

    data_frame.show()

    (training_data, test_data) = data_frame.randomSplit([0.8, 0.2])

    print("training data: " + str(training_data.count()))
    print("test data: " + str(test_data.count()))

    logistic_regression = LogisticRegression(maxIter=100, regParam=0.3, elasticNetParam=0.8)

    model = logistic_regression.fit(training_data)

    prediction = model.transform(test_data)
    prediction.printSchema()
    prediction.show()

    prediction\
        .select("prediction", "label", "features")\
        .show(5)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(prediction)
    print("Test Error = %g " % (1.0 - accuracy))

    spark_session.stop()











