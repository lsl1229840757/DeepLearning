'''
Author: lsl
Date: 2020-09-01 16:28:28
LastEditTime: 2020-09-01 16:32:13
Description: 获取注册到集群的应用id
'''
from pyspark import SparkConf, SparkContext
import numpy as np

conf = SparkConf()
conf.set("master", "spark://hadoop-maste:7077")
sc = SparkContext(conf=conf)

rdd = sc.parallelize(np.arange(10))
print("applicationId:", sc.applicationId)
print(rdd.collect())
sc.stop()
