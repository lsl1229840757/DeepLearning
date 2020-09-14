'''
                       _oo0oo_
                      o8888888o
                      88" . "88
                      (| -_- |)
                      0\  =  /0
                    ___/`---'\___
                  .' \\|     |// '.
                 / \\|||  :  |||// \
                / _||||| -:- |||||- \
               |   | \\\  - /// |   |
               | \_|  ''\---/''  |_/ |
               \  .-\__  '-'  ___/-. /
             ___'. .'  /--.--\  `. .'___
          ."" '<  `.___\_<|>_/___.' >' "".
         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
         \  \ `_.   \_ __\ /__ _/   .-` /  /
     =====`-.____`.___ \_____/___.-`___.-'=====
                       `=---='


     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

           佛祖保佑       永不宕机     永无BUG

Author: lsl
Date: 2020-09-09 21:16:32
LastEditTime: 2020-09-09 21:36:26
Description: rdd的aggregate函数
'''
from pyspark import SparkConf, SparkContext
import numpy as np

conf = SparkConf()
conf.set("matster", "spark://hadoop-maste:7077")
sc = SparkContext(conf=conf)
# 创建一个三分区的rdd
rdd = sc.parallelize(np.arange(11), 3)
# 查看rdd如何分区
res1 = sc.runJob(rdd, lambda iterator: iterator, partitions=[0])
res2 = sc.runJob(rdd, lambda iterator: iterator, partitions=[1])
res3 = sc.runJob(rdd, lambda iterator: iterator, partitions=[2])
print("res1:", res1)
print("res2:", res2)
print("res3:", res3)
# 累加得到这个rdd的sum
print(rdd.aggregate(0, lambda x, y: x+y, lambda x, y: x+y))
# 分区累乘法
print(rdd.aggregate(0, lambda x, y: x+y, lambda x, y: x*y))

sc.stop()
