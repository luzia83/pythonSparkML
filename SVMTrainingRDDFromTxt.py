from pyspark import SparkContext, SparkConf
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.regression import LabeledPoint


def parse_point(line):
    values = [float(x) for x in line.split(' ')]
    return LabeledPoint(values[0], values[1:])

"""
1 1 3 -> [1.0, 1.0, 3.0"]
0 1.1 2.5
..
"""

if __name__ == "__main__":
    sparkConf = SparkConf()
    sparkContext = SparkContext(conf=sparkConf)

    sparkContext.setLogLevel("OFF")

    data = sparkContext \
        .textFile("data/classificationdata.txt")

    parsed_data = data \
        .map(parse_point) \
        .cache()

    for point in parsed_data.collect():
        print(point)

    model = SVMWithSGD.train(parsed_data, iterations=100)

    model.save(sparkContext, "SVModel123")