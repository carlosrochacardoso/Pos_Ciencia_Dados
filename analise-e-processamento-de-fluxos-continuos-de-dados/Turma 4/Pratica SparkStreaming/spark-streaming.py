from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils

def get_cpf(pessoa):
    if pessoa == "Pedro":
        return "123.098.456-87"
    else:
        return "854.098.456-87"

spark = SparkSession \
    .builder \
    .appName("teste") \
    .config("spark.jars.packages", "org.apache.spark:spark-streaming-kafka-0-8_2.11:2.2.0" ) \
    .config("spark.streaming.kafka.maxRatePerPartition", 1000) \
    .config("spark.default.parallelism", "2") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()


sc = spark.sparkContext
ssc = StreamingContext(sc, 5)

kvs = KafkaUtils.createStream(ssc, '127.0.0.1:2181', "spark-streaming-consumer", {"testing": 1})
lines = kvs.map(lambda x: x[1] + ' ' + get_cpf(x[1]))
lines.pprint()

#remova a funcao get_cpf do map acima, comente o lines.pprint
#depois remova os comentarios e veja a contagem de palavras

#counts = lines.flatMap(lambda line: line.split(" ")) \
#              .map(lambda word: (word, 1)) \
#              .reduceByKey(lambda a, b: a+b)
#counts.pprint()

ssc.start()
ssc.awaitTermination()