from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
import PIL.Image as Image
import numpy as np
import io

def split_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_satellite_image(spark, path):
    img_df = spark.read.format("binaryFile").load(path)
    img_array = Image.open(io.BytesIO(img_df.collect()[0][3]))
    return np.array(img_array).tolist()

def histogram_ndvi(satellite_image):
    image_name, red, nir = satellite_image[0], satellite_image[1], satellite_image[2]
    histogram_limits = []
    min_ndvi = -1
    max_ndvi = 1
    interval = 0.1
    current_min = min_ndvi
    current_max = min_ndvi + interval

    while current_max <= max_ndvi:
        histogram_limits.append((current_min, current_max))
        current_min = current_max
        current_max = current_max + interval

    histogram = [0] * len(histogram_limits)
    for row_index in range(len(red)):
        for column_index in range(len(red[row_index])):
            if red[row_index][column_index] + nir[row_index][column_index] != 0:
                ndvi_value = (nir[row_index][column_index] - red[row_index][column_index]) / (nir[row_index][column_index] + red[row_index][column_index])
                limits_index = 0
                for limits in histogram_limits:
                    if ndvi_value >= limits[0] and ndvi_value < limits[1]:
                        histogram[limits_index] += 1
                        break
                    limits_index += 1
    return image_name, histogram

def write_csv(histograms):
    with open("results.csv", "a") as csv_file:
        for histogram in histograms:
            csv_file.write(f"{histogram[0]};")
            for histogram_interval in histogram[1]:
                csv_file.write(f"{histogram_interval};")
            csv_file.write("\n")

conf = SparkConf().setAppName('satellite-images').setMaster('spark://10.223.77.3:7077')
spark = SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()

with open("images.txt", "r") as images_file:
    images = images_file.readlines()
    images = [image[:-1] for image in images]

batch_images_list = list(split_list(images, 10))

for batch in batch_images_list:
    rdds, histograms = [], []

    for image in batch:
        image_b4 = get_satellite_image(spark, f"hdfs://10.223.77.3:9000/satellite_images/{image}4.TIF")
        image_b5 = get_satellite_image(spark, f"hdfs://10.223.77.3:9000/satellite_images/{image}5.TIF")
        image_rdd = spark.sparkContext.parallelize([[image, image_b4, image_b5]])
        rdds.append(image_rdd)

    for rdd in rdds:
        histogram_rdd = rdd.map(lambda rdd: histogram_ndvi(rdd))
        histogram_data = histogram_rdd.collect()
        histograms.append(histogram_data[0])

    print(histograms)

    write_csv(histograms)
