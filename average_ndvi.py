from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
import PIL.Image as Image
import numpy as np
import io

conf = SparkConf().setAppName('satellite-images').setMaster('spark://10.223.77.3:7077')
spark = SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()

def get_satellite_image(spark, path):
    img_df = spark.read.format("binaryFile").load(path)
    img_array = Image.open(io.BytesIO(img_df.collect()[0][3]))
    return np.array(img_array).tolist()

img_path_b4 = "hdfs://10.223.77.3:9000/LT05_L2SP_006036_20070625_20200830_02_T2_SR_B4.TIF"
img_path_b5 = "hdfs://10.223.77.3:9000/LT05_L2SP_006036_20070625_20200830_02_T2_SR_B5.TIF"

img_list_b4 = get_satellite_image(spark, img_path_b4)
img_list_b5 = get_satellite_image(spark, img_path_b4)

satellite_images_rdd = spark.sparkContext.parallelize([[img_list_b4, img_list_b5]])

def average_ndvi(satellite_image):
    red, nir = satellite_image[0], satellite_image[1]
    ndvi_sum = 0
    content_pxls = 0
    for row_index in range(len(red)):
        for column_index in range(len(red[row_index])):
            if red[row_index][column_index] + nir[row_index][column_index] != 0:
                ndvi_value = (nir[row_index][column_index] - red[row_index][column_index]) / (nir[row_index][column_index] + red[row_index][column_index])
                ndvi_sum += ndvi_value
                content_pxls += 1

    ndvi_avg = ndvi_sum / content_pxls
    return ndvi_avg

avg_ndvi_rdd = satellite_images_rdd.map(lambda satellite_image: average_ndvi(satellite_image))

avg_ndvi_value = avg_ndvi_rdd.first()

print(f"The average NDVI index: {avg_ndvi_value}")