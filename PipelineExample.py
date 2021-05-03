from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Tokenizer, HashingTF
from pyspark.sql import SparkSession

if __name__ == "__main__":
    sparkSession = SparkSession\
        .builder\
        .getOrCreate()

    # Prepare training documents from a list of (id, text, label) tuples.
    training = sparkSession.createDataFrame([
        (0, "a b c d e spark", 1.0),
        (1, "b d", 0.0),
        (2, "spark f g h", 1.0),
        (3, "hadoop mapreduce", 0.0)
    ], ["id", "text", "label"])

    # Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and logistic regression.
    tokenizer = Tokenizer(inputCol="text", outputCol="words")

    tk = tokenizer.transform(training)
    tk.printSchema()
    tk.show()


    hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
    hs = hashingTF.transform(tk)
    hs.printSchema()
    hs.show()

    logistic_regression = LogisticRegression(maxIter=10, regParam=0.001)

    pipeline = Pipeline(stages=[tokenizer, hashingTF, logistic_regression])

    # Fit the pipeline to training documents.
    model = pipeline.fit(training)

    # Prepare test documents, which are unlabeled (id, text) tuples.
    test = sparkSession.createDataFrame([
        (4, "spark i j k"),
        (5, "l m n"),
        (6, "spark hadoop spark"),
        (7, "apache hadoop")
    ], ["id", "text"])

    # Make predictions on test documents and print columns of interest.
    prediction = model.transform(test)
    prediction.printSchema()
    prediction.show()

    selected = prediction.select("id", "text", "probability", "prediction")
    for row in selected.collect():
        rid, text, prob, prediction = row
        print("(%d, %s) --> prob=%s, prediction=%f" % (rid, text, str(prob), prediction))

    sparkSession.stop()