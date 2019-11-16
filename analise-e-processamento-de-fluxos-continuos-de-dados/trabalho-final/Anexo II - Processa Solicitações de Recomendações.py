from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from pymongo import MongoClient
from kafka import KafkaProducer
import json

def processa_solicitacao(usuarios):

    #Conexão com o banco de recomendações
    client = MongoClient('localhost', 27017)
    db = client.movies.knn_recomendacoes

    #Inicaliza producer para o tópico kafka de saida das recomendações
    producer = KafkaProducer(bootstrap_servers='localhost:9092', \
                             value_serializer=lambda v: json.dumps(v).encode('utf-8'))

    for id in usuarios:
        doc = db.find_one({'userId': int(id)})
        del doc['_id']
        producer.send('saida-recomendacao',doc)

    client.close()

spark = SparkSession \
    .builder \
    .appName("GetRecomendacaoService") \
    .config("spark.jars.packages", "org.apache.spark:spark-streaming-kafka-0-8_2.11:2.2.0") \
    .config("spark.streaming.kafka.maxRatePerPartition", 1000) \
    .config("spark.default.parallelism", "2") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

sc = spark.sparkContext
ssc = StreamingContext(sc, 1)

kvs = KafkaUtils.createStream(ssc, '127.0.0.1:2181', "spark-streaming-consumer", {"entrada-recomendacao": 1})
lines = kvs.map(lambda x: x[1])
lines.foreachRDD(lambda rdd: rdd.foreach(processa_solicitacao))
lines.pprint()

ssc.start()
ssc.awaitTermination()

