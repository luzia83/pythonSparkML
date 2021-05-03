import sys

from pyspark import SparkConf, SparkContext
from pyspark.mllib.clustering import KMeans, KMeansModel
import numpy as np

if __name__ == "__main__":
    sparkConf = SparkConf()
    sparkContext = SparkContext(conf=sparkConf)

    model = KMeansModel.load(sparkContext, path=sys.argv[1])

    print("Model loaded")

    point = [1.1, 3.2]
    print("Point 1: " + str(point))
    print("Predict: " + str(model.predict(point)))

    point = [5.1, 1.4]
    print("Point 2: " + str(point))
    print("Predict: " + str(model.predict(point)))

    point = np.array([5.2, 2.0])
    print("Point 3: " + str(point))
    print("Predict: " + str(model.predict(point)))

    point = np.array([1.0, 4.0])
    print("Point 4: " + str(point))
    print("Predict: " + str(model.predict(point)))

    point = [3.4, 2.0]
    print("Point 2: " + str(point))
    print("Predict: " + str(model.predict(point)))

    sparkContext.stop()