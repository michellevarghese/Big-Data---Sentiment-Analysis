from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql.session import SparkSession

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from pyspark.sql.functions import col
from pyspark.sql.types import StringType,BooleanType,DateType

import numpy as np
import pandas as pd
import json
import pickle

count=0
filename1 = "mlb_model.pkl"
filename2 = "sgd_model.pkl"
filename3 = "perceptron_model.pkl"
filename4 = "mlp_model.pkl"
cm_file = "confusion_matrix.txt"
scores_mlb = "scores_mlb.txt"
scores_sgd = "scores_sgd.txt"
scores_perceptron = "scores_perceptron.txt"
scores_mlp = "scores_mlp.txt"



def preprocess(batch):
	for i in list(batch.collect()):
		if i == []:
			continue
		else:
			global count
			count+=1
			
			fopen1 = open(filename1,"rb")
			mlb = pickle.load(fopen1)

			fopen2 = open(filename2,"rb")
			sgd = pickle.load(fopen2)

			fopen3 = open(filename3,"rb")
			perceptron = pickle.load(fopen3)

			fopen4 = open(filename4,"rb")
			mlp = pickle.load(fopen4)

			fopen5 = open(cm_file,"a+")
			fopen6 = open(scores_mlb,"a+")
			fopen7 = open(scores_sgd,"a+")
			fopen8 = open(scores_perceptron,"a+")
			fopen9 = open(scores_mlp,"a+")


			test_set = json.loads(i)
			test_set = spark.createDataFrame(test_set.values())

			tokenizer = Tokenizer(inputCol="feature1", outputCol="words")
			hashtf = HashingTF(numFeatures=2**7, inputCol="words", outputCol='tf')
			idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5)
			label_stringIdx = StringIndexer(inputCol = "feature0", outputCol = "label")
			
			pipeline = Pipeline(stages=[tokenizer, hashtf, idf, label_stringIdx])
			pipelineFit = pipeline.fit(test_set)
			test_df = pipelineFit.transform(test_set)
			
			a = np.array(test_df.select("features").collect()).reshape(-1, 128)
			y = np.array(test_df.select("feature0").collect()).reshape(-1)


			print("Batch : ",count)

			predictions3 = mlb.predict(a)
			prednp3=np.array(predictions3)
			np.savetxt(cm_file,confusion_matrix(y,prednp3))
			fopen6.write(str(precision_score(y, prednp3,average='weighted'))+ " ")
			fopen6.write(str(recall_score(y, prednp3,average='weighted')) + " ")
			fopen6.write(str(f1_score(y, prednp3,average='weighted')) + "\n")


			predictions1 = sgd.predict(a)
			prednp1=np.array(predictions1)
			np.savetxt(cm_file,confusion_matrix(y,prednp1))
			fopen7.write(str(precision_score(y, prednp1,average='weighted'))+ " ")
			fopen7.write(str(recall_score(y, prednp1,average='weighted') )+" ")
			fopen7.write(str(f1_score(y, prednp1,average='weighted') )+ "\n")

			predictions2 = perceptron.predict(a)
			prednp2 = np.array(predictions2)
			np.savetxt(cm_file,confusion_matrix(y,prednp2))
			fopen8.write(str(precision_score(y, prednp2,average='weighted')) + " ")
			fopen8.write(str(recall_score(y, prednp2,average='weighted')) + " ")
			fopen8.write(str(f1_score(y, prednp2,average='weighted')) + "\n")

			
			predictions4 = mlp.predict(a)
			prednp4 = np.array(predictions4)
			np.savetxt(cm_file,confusion_matrix(y,prednp4))
			fopen9.write(str(precision_score(y, prednp4,average='weighted')) + " ")
			fopen9.write(str(recall_score(y, prednp4,average='weighted')) + " ")
			fopen9.write(str(f1_score(y, prednp4,average='weighted')) + "\n")

			fopen1.close()
			fopen2.close()
			fopen3.close()
			fopen4.close()
			fopen5.close()
			fopen6.close()
			fopen7.close()
			fopen8.close()
			fopen9.close()


			
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