from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils

"""
0 1:1   2:3
0 1:1.1 2:2.5
...
"""

if __name__ == "__main__":
    sparkConf = SparkConf()
    sparkContext = SparkContext(conf=sparkConf)

    sparkContext.setLogLevel("OFF")
    # Load and parse the data
    parsed_data = MLUtils \
        .loadLibSVMFile(sparkContext, "data/classificationDataLibsvm.txt") \
        .cache()

    for point in parsed_data.collect():
        print(point)

    # Build the model
    model = SVMWithSGD.train(parsed_data, iterations=100)

    model.save(sparkContext, "SVModel124")