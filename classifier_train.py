from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql.session import SparkSession

from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from pyspark.sql.functions import col
from pyspark.sql.types import StringType,BooleanType,DateType

import numpy as np
import pandas as pd
import json
import pickle

sgd = SGDClassifier()
perceptron = Perceptron()
mlb = MultinomialNB()
mlp = MLPClassifier() 

count=0
sgd_filename = "sgd_model.pkl"
perceptron_filename = "perceptron_model.pkl"
mlb_filename = "mlb_model.pkl"
mlp_filename = "mlp_model.pkl"


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
			#removersp = StopWordsRemover(inputCol="words", outputCol="words_filtered") ---uncomment to remove stopwords
			#hashtf = HashingTF(numFeatures=2**7, inputCol="words_filtered", outputCol='tf') ---use this to apply stop words removal
            hashtf = HashingTF(numFeatures=2**7, inputCol="words", outputCol='tf')
			idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5)
			label_stringIdx = StringIndexer(inputCol = "feature0", outputCol = "label")
			
			pipeline = Pipeline(stages=[tokenizer, hashtf, idf, label_stringIdx])
            #pipeline = Pipeline(stages=[tokenizer, removersp, hashtf, idf, label_stringIdx]) ---use this to apply stop words removal
			pipelineFit = pipeline.fit(train_set)
			train_df = pipelineFit.transform(train_set)
			val_df = pipelineFit.transform(val_set)
			
			a = np.array(train_df.select("features").collect()).reshape(-1, 128)
			b = np.array(val_df.select("features").collect()).reshape(-1, 128)
			y = np.array(train_df.select("feature0").collect()).reshape(-1)
			p =np.array(val_df.select("feature0").collect()).reshape(-1)

			sgd.partial_fit(a, y, classes=np.array([0,4]))
			perceptron.partial_fit(a,y,classes = np.array([0,4]))
			mlb.partial_fit(a,y,classes = np.array([0,4]))
			mlp.partial_fit(a,y,classes = np.array([0,4]))

			mlb_pred = mlb.predict(b)
			mlb_prednp = np.array(mlb_pred)
			mlb_accuracy = sum(mlb_prednp==p)/len(p)			

			fopen = open(mlb_filename,"wb")
			pickle.dump(mlb,fopen)
			fopen.close()

			mlp_pred = mlp.predict(b)
			mlp_prednp = np.array(mlp_pred)
			mlp_accuracy = sum(mlp_prednp==p)/len(p)

			fopen = open(mlp_filename,"wb")
			pickle.dump(mlp,fopen)
			fopen.close()

			perceptron_predictions = perceptron.predict(b)
			perceptron_prednp = np.array(perceptron_predictions)
			perceptron_accuracy=sum(perceptron_prednp==p)/len(p)

			fopen = open(perceptron_filename,"wb")
			pickle.dump(perceptron,fopen)
			fopen.close()

			sgd_predictions = sgd.predict(b)
			sgd_prednp=np.array(sgd_predictions)
			sgd_accuracy=sum(sgd_prednp==p)/len(p)
			
			fopen = open(sgd_filename,"wb")
			pickle.dump(sgd,fopen)
			fopen.close()
			print(count, mlb_accuracy, sgd_accuracy, perceptron_accuracy, mlp_accuracy)			

# Create a local StreamingContext with two working thread and batch interval of 1 second
sc = SparkContext()
spark=SparkSession(sc)
ssc = StreamingContext(sc, 1)
sc.setLogLevel("OFF")

lines = ssc.socketTextStream("localhost", 6100)

words = lines.flatMap(lambda line: line.split('\n'))
words.foreachRDD(preprocess)

ssc.start()             # Start the computation
ssc.awaitTermination()
