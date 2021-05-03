from math import sqrt

from pyspark import SparkConf, SparkContext
from pyspark.mllib.clustering import KMeans

if __name__ == "__main__":
    sparkConf = SparkConf()
    sparkContext = SparkContext(conf=sparkConf)

    data = sparkContext\
        .textFile("data/clusteringData.txt")

    parsed_data = data\
        .map(lambda line: [float(x) for x in line.split(' ')])\
        .cache()

    number_of_clusters = 4
    number_of_iterations = 20

    clusters = KMeans.train(parsed_data, number_of_clusters, number_of_iterations, initializationMode="random")

    def error(point):
        center = clusters.centers[clusters.predict(point)]
        return sqrt(sum([x ** 2 for x in (point - center)]))

    WSSSE = parsed_data.map(lambda point: error(point)).reduce(lambda x, y: x + y)
    print("Within Set Sum of Squared Error = " + str(WSSSE))

    centers = clusters.clusterCenters
    print("Cluster Centers: ")
    for center in centers:
        print(center)

    # Save and load model
    clusters.save(sparkContext, "KMEANSMODEL4")

    sparkContext.stop()