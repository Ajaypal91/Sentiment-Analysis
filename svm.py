from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import Word2Vec
from pyspark.ml.feature import StringIndexer
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.linalg import Vectors
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

def getData(path="/FileStore/tables/wldrhwsy1480045356097/1.csv") :
  data = sqlContext.read.format("com.databricks.spark.csv").option("header",True).option("inferSchema", True).option("delimiter","\t").load(path) 
  return data.toDF("ID","lab","review")

def createTFIDFFeatures(inputData,numOfFeatures=300,inputColumn="review", outputColumn="result") :
  tokenizer = Tokenizer(inputCol=inputColumn, outputCol="words")
  remover = StopWordsRemover(inputCol="words", outputCol="filtered")
  hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=numOfFeatures)
  idf = IDF(inputCol="rawFeatures", outputCol=outputColumn)
  pipeline = Pipeline(stages=[tokenizer,remover, hashingTF,idf])
  model = pipeline.fit(inputData)
  return model.transform(inputData).drop("words").drop("rawFeatures")

def createWord2VecFeatures(inputData,numOfFeatures=150,inputColumn="review",outputColumn="result") :
  tokenizer = Tokenizer(inputCol=inputColumn, outputCol="words")
  remover = StopWordsRemover(inputCol="words", outputCol="filtered")
  word2Vec = Word2Vec(vectorSize=numOfFeatures, minCount=0, inputCol="filtered", outputCol="result")
  pipeline = Pipeline(stages=[tokenizer,remover,word2Vec])
  model = pipeline.fit(inputData)
  return model.transform(inputData).drop("words").drop("filtered")

def stringIndexer(inputDF,inputColumn="lab",outputColumn="label") :
  stringIndexer = StringIndexer(inputCol="lab", outputCol="label")
  si_model = stringIndexer.fit(inputDF)
  return si_model.transform(inputDF).drop("lab")

def myFunc(val) :
  return LabeledPoint(val[0],val[1:])

data = getData()
data = getData()
startFeatureCol="review"
startLabCol="lab"
indexedDF = stringIndexer(data,inputColumn=startLabCol)
td = createWord2VecFeatures(indexedDF,numOfFeatures=700,inputColumn=startFeatureCol)
fd = td.select("label","result").rdd.map(myFunc)
fd.cache()
(train, test) = fd.randomSplit([0.8,0.2])
model = SVMWithSGD.train(train, iterations=1000)
# model = LogisticRegressionWithLBFGS.train(train,iterations=1000) #accuracy of 85%
labelsAndPreds = test.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(test.count())
print("Testing Error = " + str(trainErr))
