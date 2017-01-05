from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import Word2Vec
from pyspark.ml.feature import StringIndexer
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

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
 
def getMultiClassEvaluator():
  evaluator = MulticlassClassificationEvaluator()
  return evaluator


def runNeuralNetworkTFIDF(startFeatureCol="review",startLabCol="lab",numbFeaturesForTFIDF=1000)  :
    data = getData()
    evaluator = getMultiClassEvaluator()
    indexedDF = stringIndexer(data,inputColumn=startLabCol)
    td = createTFIDFFeatures(indexedDF,numOfFeatures=numbFeaturesForTFIDF,inputColumn=startFeatureCol)
    (train, test) = td.randomSplit([0.8,0.2])
    # specify layers for the neural network:
    # input layer of size 4 (features), two intermediate of size 5 and 4
    # and output of size 2 (classes)
    layers = [numbFeaturesForTFIDF, 10, 10, 2]
    # create the trainer and set its parameters
    trainer = MultilayerPerceptronClassifier(maxIter=500, layers=layers)
    trainer.setFeaturesCol("result")
    trainer.setLabelCol("label")
    # train the model
    model = trainer.fit(train)
    # compute accuracy on the test set
    result = model.transform(test)
    predictionAndLabels = result.select("prediction", "label")
    print("f1 score: " + str(evaluator.evaluate(predictionAndLabels)))
    return predictionAndLabels

def runNeuralNetworkWord2Vec(startFeatureCol="review",startLabCol="lab",numFeaturesForWord2Vec=700)  :
    data = getData()
    evaluator = getMultiClassEvaluator()
    indexedDF = stringIndexer(data,inputColumn=startLabCol)
    td = createWord2VecFeatures(indexedDF,numOfFeatures=numFeaturesForWord2Vec,inputColumn=startFeatureCol)
    td.cache()
    (train, test) = td.randomSplit([0.8,0.2])
    # specify layers for the neural network:
    # input layer of size 4 (features), two intermediate of size 5 and 4
    # and output of size 2 (classes)
    layers = [numFeaturesForWord2Vec, 10, 2]
    # create the trainer and set its parameters
    trainer = MultilayerPerceptronClassifier(maxIter=700, layers=layers)
    trainer.setFeaturesCol("result")
    trainer.setLabelCol("label")
    # train the model
    model = trainer.fit(train)
    # compute accuracy on the test set
    result = model.transform(test)
    predictionAndLabels = result.select("prediction", "label")
    print("f1 score: " + str(evaluator.evaluate(predictionAndLabels))) 
    return predictionAndLabels

def myFunc(val):
  if val[0] == val[1]:
    return 0
  else :
    return 1
  
def getAcc(y):
  err = y.rdd.map(myFunc)
  print "Accuracy = %s" % str(1 - float(err.sum())/err.count())
    
# y = runNeuralNetworkTFIDF() #f1 score: 0.763550709879
# getAcc(y)
z = runNeuralNetworkWord2Vec() #f1 score: 0.859760386145
getAcc(z) #Accuracy = 0.859759700611

