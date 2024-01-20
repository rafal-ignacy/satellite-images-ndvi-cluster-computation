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

def threshold_ndvi(satellite_image):
    image_name, red, nir = satellite_image[0], satellite_image[1], satellite_image[2]
    ndvi_result = []
    threshold = 0.12
    for row_index in range(len(red)):
        ndvi_result_column = []
        for column_index in range(len(red[row_index])):
            if red[row_index][column_index] + nir[row_index][column_index] != 0:
                ndvi_value = (nir[row_index][column_index] - red[row_index][column_index]) / (nir[row_index][column_index] + red[row_index][column_index])
                ndvi_result_column.append(ndvi_value)
            else:
                ndvi_result_column.append(0)
        ndvi_result.append(ndvi_result_column)
    return image_name, ndvi_result

conf = SparkConf().setAppName('satellite-images').setMaster('spark://10.223.77.3:7077')
spark = SparkContext(conf=conf)

spark = SparkSession.builder.config("spark.driver.memory", "15g").config("spark.executor.memory", "14g").config('spark.rpc.message.maxSize','1024').config("spark.memory.offHeap.enabled","true").getOrCreate()

with open("images.txt", "r") as images_file:
    images = images_file.readlines()
    images = [image[:-1] for image in images]

batch_images_list = list(split_list(images, 10))

for batch in batch_images_list:
    rdds = []
    for image in batch:
        image_b4 = get_satellite_image(spark, f"hdfs://10.223.77.3:9000/satellite_images/{image}4.TIF")
        image_b5 = get_satellite_image(spark, f"hdfs://10.223.77.3:9000/satellite_images/{image}5.TIF")
        image_rdd = spark.sparkContext.parallelize([[image, image_b4, image_b5]])
        rdds.append(image_rdd)

    for rdd in rdds:
        threshold_rdd = rdd.map(lambda rdd: threshold_ndvi(rdd))
        threshold_img_data = threshold_rdd.collect()[0]
        threshold_img_array = np.array(threshold_img_data[1], dtype=np.float64)
        img_file = Image.fromarray(threshold_img_array)
        img_file.save("results/threshold/" + threshold_img_data[0] + "-threshold-ndvi.tif")