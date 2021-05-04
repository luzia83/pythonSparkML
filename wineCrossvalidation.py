from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import SparkSession
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

if __name__ == "__main__":
    spark_session = SparkSession \
        .builder \
        .appName("Spark Crossvalidation with Random Forest dataframes") \
        .master("local[4]") \
        .getOrCreate()

    data_frame = spark_session \
        .read \
        .format("libsvm") \
        .load("data/wine.scale.txt")

    data_frame.printSchema()
    data_frame.show()

    randomForest = RandomForestClassifier()
    accuracies = []

    for contador in range(0, 10):
        (training_data, test_data) = data_frame.randomSplit([0.8, 0.2])

        paramGrid = ParamGridBuilder() \
            .addGrid(randomForest.numTrees, [10, 50, 100])\
            .addGrid(randomForest.maxDepth, [3, 5])\
            .build()

        evaluator = MulticlassClassificationEvaluator(metricName="accuracy")

        crossval = CrossValidator(estimator=randomForest,
                                  estimatorParamMaps=paramGrid,
                                  evaluator=evaluator,
                                  numFolds=5)

        wine_svm_model = crossval.fit(training_data)

        best_model = wine_svm_model.bestModel
        param_num_trees = best_model.getParam("numTrees")
        param_max_depth = best_model.getParam("maxDepth")
        num_trees = best_model.extractParamMap().get(best_model.getParam("numTrees"))
        max_depth = best_model.extractParamMap().get(best_model.getParam("maxDepth"))
        print("Best model " + str(contador))
        print("Number of trees: " + str(num_trees))
        print("Maximum depth " + str(max_depth))

        accuracy = evaluator.evaluate(wine_svm_model.transform(test_data))
        print("Accuracy: " + str(accuracy) + "\n")
        accuracies.append(accuracy)

    mean_accuracy = sum(accuracies)/len(accuracies)
    print("Mean accuracy: " + str(mean_accuracy))

    spark_session.stop()

