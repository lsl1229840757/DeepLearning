'''
Author: lsl
Date: 2020-09-01 15:48:58
LastEditTime: 2020-09-01 16:24:05
Description: 测试Spark的SparkFiles
'''
from pyspark import SparkFiles
import os
import numpy as np
from pyspark import SparkContext
from pyspark import SparkConf

# 连接spark
conf = SparkConf()
conf.set("master", "spark://hadoop-maste:7077")
context = SparkContext(conf=conf)

# 创建一个num_data文件并写入数据
file_name = "num_data.txt"

# tmpdir = "/root/workspace/sparkcontext"  # 这是本地路径
# path = os.path.join(tmpdir, file_name)
# with open(path, "w") as f:
#     f.write("100")

tmpdir = "hdfs://hadoop-maste:9000/datas/"  # 这是hdfs路径, 不过这需要自己先上传到hdfs中
path = tmpdir + file_name
# 将文件上传到spark集群中
context.addFile(path)
# 创建一个rdd
rdd = context.parallelize(np.arange(10))


# 定义mapPartions的操作函数
def func(iterator):
    # 读取文件
    with open(SparkFiles.get(file_name)) as f:
        file_val = int(f.readline())
        return [x * file_val for x in iterator]


print(rdd.mapPartitions(func).collect())
context.stop()
