from pyspark.ml.classification import MultilayerPerceptronClassifier, NaiveBayes
from pyspark.sql import SparkSession
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

if __name__ == "__main__":
    spark_session = SparkSession \
        .builder \
        .appName("Spark ML KMEANS with dataframes") \
        .master("local[4]") \
        .getOrCreate()

    data_frame = spark_session \
        .read \
        .format("libsvm") \
        .load("data/wine.scale.txt")

    data_frame.printSchema()
    data_frame.show()

    (training_data, test_data) = data_frame.randomSplit([0.8, 0.2])

    naiveBayes = NaiveBayes(modelType="gaussian")
    perceptron = MultilayerPerceptronClassifier(seed=123)

    paramGrid_old = ParamGridBuilder() \
        .addGrid(NaiveBayes.smoothing, [0.05, 0.0, 0.1, 0.2, 0.5]) \
        .build()

    paramGrid = ParamGridBuilder() \
        .addGrid(perceptron.maxIter, [10, 30, 100, 500])\
        .addGrid(perceptron.layers, [[13, 7, 5, 3], [13, 8, 4, 5, 3]])\
        .build()

    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    crossval_old = CrossValidator(estimator=naiveBayes,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=5)

    crossval = CrossValidator(estimator=perceptron, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

    wine_svm_model = crossval.fit(training_data)

    #wine_svm_model.save("SVMMODEL_WINE")
    print(wine_svm_model)

    accuracy = evaluator.evaluate(wine_svm_model.transform(test_data))
    print("Test Error = %g " % (1.0 - accuracy))

    spark_session.stop()

