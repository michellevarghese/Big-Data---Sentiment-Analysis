from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql.session import SparkSession

from pyspark.ml.feature import HashingTF, IDF, Tokenizer,StringIndexer,StopWordsRemover
from pyspark.ml import Pipeline

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import confusion_matrix,precision_score,f1_score,recall_score

import numpy as np
import json
import pickle
count = 0
kmeans = MiniBatchKMeans(n_clusters=2,random_state=0,batch_size=900)
kmeans_filename = "kmeans_model.pkl"

mod_pred = np.array([0])
mod_pred2 = np.array([0])

def preprocess(batch):
	for i in list(batch.collect()):
		if i == []:
			continue
		else:
			global count
			count+=1
			fopen = open(kmeans_filename,"rb")
			kmeans = pickle.load(fopen)

			test_set = json.loads(i)
			test_set = spark.createDataFrame(test_set.values())
			
			tokenizer = Tokenizer(inputCol="feature1", outputCol="words")
			removersp = StopWordsRemover(inputCol="words", outputCol="words_filtered")
			hashtf = HashingTF(numFeatures=2**7, inputCol="words", outputCol='tf')
			idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5)
			label_stringIdx = StringIndexer(inputCol = "feature0", outputCol = "label")

			pipeline = Pipeline(stages=[tokenizer, removersp, hashtf, idf, label_stringIdx])
			pipelineFit = pipeline.fit(test_set)
			test_df = pipelineFit.transform(test_set)

			a = np.array(test_df.select("features").collect()).reshape(-1, 128)
			p =np.array(test_df.select("feature0").collect()).reshape(-1)

			global mod_pred
			global mod_pred2

			kmeans_pred = kmeans.predict(a)
			kmeans_prednp = np.array(kmeans_pred)
			mod_pred = np.hstack((mod_pred,kmeans_prednp))
			mod_pred2 = np.hstack((mod_pred2,p))
			print(mod_pred.shape)

			# kmeans_accuracy = sum(kmeans_prednp==p)/len(p)

			# print(count,kmeans_accuracy)

			np.save("Kmeans_tested.npy",mod_pred)
			np.save("Kmeans_tested_truevals.npy",mod_pred2)

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
