from pyspark.ml.clustering import KMeans
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark_session = SparkSession \
        .builder \
        .appName("Spark ML KMEANS with dataframes") \
        .master("local[4]") \
        .getOrCreate()

    data_frame = spark_session \
        .read \
        .format("libsvm") \
        .load("data/classificationDataLibsvm.txt")

    data_frame.printSchema()
    data_frame.show()

    kmeans = KMeans().setK(2).setMaxIter(100)
    kmeans_model = kmeans.fit(data_frame)

    centers = kmeans_model.clusterCenters()
    print("Cluster centers: ")

    for center in centers:
        print(center)

    kmeans_model.save("KMEANSMODELDF")

    print(str(kmeans_model.hasSummary))

    spark_session.stop()

