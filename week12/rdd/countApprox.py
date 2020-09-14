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
Date: 2020-09-10 21:03:02
LastEditTime: 2020-09-10 21:08:20
Description: countApprox在timeout的时间之内返回一个估计的个数
'''
from pyspark import SparkConf, SparkContext

conf = SparkConf()
conf.set("master", "spark://hadoop-maste:7077")
sc = SparkContext(conf=conf)
rdd = sc.parallelize(range(1000000), 100)
print("1s, confidence 1: ", rdd.countApprox(1000, 1.0))
print("1s, confidence 0,95: ", rdd.countApprox(1000))
sc.stop()
