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
Date: 2020-09-09 21:41:33
LastEditTime: 2020-09-09 22:01:42
Description: rdd的transformation函数aggregateByKey
'''
from pyspark import SparkConf, SparkContext
import operator

conf = SparkConf()
conf.set("matster", "spark://hadoop-maste:7077")
sc = SparkContext(conf=conf)
# 创建rdd
datas = [("a", 22), ("b", 33), ("c", 44), ("b", 55), ("a", 66)]
rdd = sc.parallelize(datas, 3)
# 查看rdd如何分区
res1 = sc.runJob(rdd, lambda iterator: iterator, partitions=[0])
res2 = sc.runJob(rdd, lambda iterator: iterator, partitions=[1])
res3 = sc.runJob(rdd, lambda iterator: iterator, partitions=[2])
print("res1:", res1)
print("res2:", res2)
print("res3:", res3)
res = rdd.aggregateByKey(1, operator.add, operator.add).collect()
print(res)
print("==============================================")
# 创建rdd
datas1 = [("a", 22), ("a", 66), ("b", 33), ("b", 55), ("c", 44)]
rdd1 = sc.parallelize(datas1, 2)
# 查看rdd如何分区
res11 = sc.runJob(rdd1, lambda iterator: iterator, partitions=[0])
res12 = sc.runJob(rdd1, lambda iterator: iterator, partitions=[1])
print("res11:", res11)
print("res22:", res12)
re = rdd1.aggregateByKey(1, operator.add, operator.add)
print(re.collect())
sc.stop()
