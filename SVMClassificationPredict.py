from pyspark.ml.classification import LinearSVCModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession

if __name__ == "__main__":

    spark_session = SparkSession \
        .builder \
        .appName("Spark ML SVM") \
        .master("local[4]") \
        .getOrCreate()

    model = LinearSVCModel.load("SVMModel")
    print("Model loaded")

    # For prediction, labels are index from rows at dataframe
    prediction_dataframe = spark_session.createDataFrame([
        (0, Vectors.dense([1.0, 1.2])),
        (1, Vectors.dense([5.3, 2.4])),
        (2, Vectors.dense([1.2, 1.3])),
        (3, Vectors.dense([5.1, 2.3]))],
        ["label", "features"]) \
        .cache()

    prediction_dataframe.printSchema()
    prediction_dataframe.show()

    prediction = model.transform(prediction_dataframe)

    prediction.printSchema()
    prediction.show()

    selected = prediction.select("label", "prediction")
    selected.printSchema()
    print(selected)

    for row in selected.collect():
        print(str(row))
        print("(%d) -->  prediction=%f" % (row[0], row[1]))

    spark_session.stop()
