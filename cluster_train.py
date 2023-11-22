from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql.session import SparkSession

from pyspark.ml.feature import HashingTF, IDF, Tokenizer,StringIndexer
from pyspark.ml import Pipeline

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import confusion_matrix,precision_score,f1_score,recall_score
from pyspark.ml.feature import StopWordsRemover

import numpy as np
import json
import pickle
count = 0
kmeans = MiniBatchKMeans(n_clusters=2,random_state=0,batch_size=15200)
kmeans_filename = "kmeans_model.pkl"

def preprocess(batch):
	for i in list(batch.collect()):
		if i == []:
			continue
		else:
			global count
			count+=1
			df = json.loads(i)
			df = spark.createDataFrame(df.values())
			
			(train_set, val_set) = df.randomSplit([0.80, 0.20], seed = 2000)
			tokenizer = Tokenizer(inputCol="feature1", outputCol="words")
			removersp = StopWordsRemover(inputCol="words", outputCol="words_filtered")
			hashtf = HashingTF(numFeatures=2**7, inputCol="words_filtered", outputCol='tf')
			idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5)
			label_stringIdx = StringIndexer(inputCol = "feature0", outputCol = "label")

			pipeline = Pipeline(stages=[tokenizer,removersp, hashtf, idf, label_stringIdx])
			pipelineFit = pipeline.fit(train_set)
			train_df = pipelineFit.transform(train_set)
			val_df = pipelineFit.transform(val_set)

			a = np.array(train_df.select("features").collect()).reshape(-1, 128)
			b = np.array(val_df.select("features").collect()).reshape(-1, 128)
			y = np.array(train_df.select("feature0").collect()).reshape(-1)
			p =np.array(val_df.select("feature0").collect()).reshape(-1)

			kmeans.partial_fit(a,y)

			kmeans_pred = kmeans.predict(b)
			print(kmeans_pred)
			kmeans_prednp = np.array(kmeans_pred)
			kmeans_accuracy = sum(kmeans_prednp==p)/len(p)

			print(count,kmeans_accuracy)

			fopen = open(kmeans_filename,"wb")
			pickle.dump(kmeans,fopen)
			fopen.close()


sc = SparkContext()
spark=SparkSession(sc)
ssc = StreamingContext(sc, 1)
sc.setLogLevel("OFF")

lines = ssc.socketTextStream("localhost", 6100)

words = lines.flatMap(lambda line: line.split('\n'))
words.foreachRDD(preprocess)

ssc.start()             # Start the computation
ssc.awaitTermination()