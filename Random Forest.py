from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql.functions import col
from pyspark.ml.feature import Word2Vec
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover

def getData(path="/FileStore/tables/wldrhwsy1480045356097/1.csv") :
  data = sqlContext.read.format("com.databricks.spark.csv").option("header",True).option("inferSchema", True).option("delimiter","\t").load(path) 
  return data.toDF("ID","lab","review")

def createTFIDFFeatures(inputData,numOfFeatures=300,inputColumn="review", outputColumn="result") :
  tokenizer = Tokenizer(inputCol=inputColumn, outputCol="words")
  hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=numOfFeatures)
  idf = IDF(inputCol="rawFeatures", outputCol=outputColumn)
  pipeline = Pipeline(stages=[tokenizer,hashingTF,idf])
  model = pipeline.fit(inputData)
  return model.transform(inputData).drop("words").drop("rawFeatures")

def createWord2VecFeatures(inputData,numFeatures=150,inputColumn="review",outputColumn="result") :
  tokenizer = Tokenizer(inputCol=inputColumn, outputCol="words")
  remover = StopWordsRemover(inputCol="words", outputCol="filtered")
  word2Vec = Word2Vec(vectorSize=numFeatures, minCount=0, inputCol="filtered", outputCol="result")
  pipeline = Pipeline(stages=[tokenizer,remover,word2Vec])
  model = pipeline.fit(inputData)
  return model.transform(inputData).drop("words").drop("filtered")

def stringIndexer(inputDF,inputColumn="lab",outputColumn="label") :
  stringIndexer = StringIndexer(inputCol="lab", outputCol="label")
  si_model = stringIndexer.fit(inputDF)
  return si_model.transform(inputDF).drop("lab")

def getRandomForestTreeClassifier(data,startFeatureCol="review",startLabCol="lab",usingTFIDFNotWord2Vec=True,numberOfTrees=15,numbFeaturesForTFIDF=500,numFeaturesForWord2Vec=300) :
  indexedDF = stringIndexer(data,inputColumn=startLabCol)
  rf = RandomForestClassifier(labelCol="label", featuresCol="result", numTrees=numberOfTrees)
  print "Classifying Using Random Forest\n"
  if usingTFIDFNotWord2Vec :
    print "Performing TF IDF Processing"
    td = createTFIDFFeatures(indexedDF,numOfFeatures=numbFeaturesForTFIDF,inputColumn=startFeatureCol)
    (trainingData, testingData) = td.randomSplit([0.8,0.2])
    rfModel = rf.fit(trainingData)
    return rfModel.transform(testingData)
  else :
    print "Performing Word2Vec Processing"
    td = createWord2VecFeatures(indexedDF,numFeaturesForWord2Vec,inputColumn=startFeatureCol) 
    (trainingData, testingData) = td.randomSplit([0.8,0.2])    
    rfModel = rf.fit(trainingData)
    return rfModel.transform(testingData)
  
def getRandomForestTreeClassifierUsingCV(data,startFeatureCol="review",startLabCol="lab",usingTFIDFNotWord2Vec=True,numbFeaturesForTFIDF=500,numFeaturesForWord2Vec=300,evaluator=None) :                                      
  indexedDF = stringIndexer(data,inputColumn=startLabCol)
  rf = RandomForestClassifier(labelCol="label", featuresCol="result")
  paramGrid = (ParamGridBuilder()
             .addGrid(rf.maxDepth, [2, 4, 6])
             .addGrid(rf.maxBins, [20, 60])
             .addGrid(rf.numTrees, [5, 15, 20])
             .build())
  cv = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
  print "Classifying Using Random Forest and Cross Validator\n"
  if usingTFIDFNotWord2Vec :
    print "Performing TF IDF Processing"
    td = createTFIDFFeatures(indexedDF,numOfFeatures=numbFeaturesForTFIDF,inputColumn=startFeatureCol)
    (trainingData, testingData) = td.randomSplit([0.8,0.2])
    cvModel = cv.fit(trainingData)
    return cvModel.transform(testingData) 
  else :
    print "Performing Word2Vec Processing"
    td = createWord2VecFeatures(indexedDF,numFeaturesForWord2Vec,inputColumn=startFeatureCol) 
    (trainingData, testingData) = td.randomSplit([0.8,0.2])    
    cvModel = cv.fit(trainingData)
    return cvModel.transform(testingData)
  
def evaluationUsingBinaryClassificationEvaluator(labelCol="label",rawPredictionCol="rawPrediction") :
  evaluator = BinaryClassificationEvaluator()
  evaluator.setLabelCol(labelCol)
  evaluator.setRawPredictionCol(rawPredictionCol)
  return evaluator
  
def getScore(evaluator,predictions) :
  return evaluator.evaluate(predictions)

def getResultFromTFRandomForestClassifier(crossValidator=False) :
  data = getData()
  evaluator = evaluationUsingBinaryClassificationEvaluator()
  if not crossValidator :
    pred = getRandomForestTreeClassifier(data,numbFeaturesForTFIDF=1500)
    print "Area under ROC curve of Model = ", getScore(evaluator,pred)
  else :
    pred = getRandomForestTreeClassifierUsingCV(data,numbFeaturesForTFIDF=1000,evaluator=evaluator)
    print "Area under ROC curve of Model = ", getScore(evaluator,pred)
    
def getResultFromWord2VecRandomForestClassifier(crossValidator=False) :
  data = getData()
  evaluator = evaluationUsingBinaryClassificationEvaluator()
  if not crossValidator :
    pred = getRandomForestTreeClassifier(data,usingTFIDFNotWord2Vec=False,numFeaturesForWord2Vec=300)
    print "Area under ROC curve of Model = ", getScore(evaluator,pred) #you can change the evaluation metric by setting the metric in evaluator or to get the default metric use getMetricName() method 
  else :
    pred = getRandomForestTreeClassifierUsingCV(data,usingTFIDFNotWord2Vec=False,numFeaturesForWord2Vec=300,evaluator=evaluator)
    print "Area under ROC curve of Model = ", getScore(evaluator,pred)
    
  
getResultFromTFRandomForestClassifier() #get Score For Training random forest and data transformations with TF and IDF
# getResultFromTFRandomForestClassifier(crossValidator=True) #get Score For Training random forest and data transformations with TF and IDF and CrossValidation
# getResultFromWord2VecRandomForestClassifier() #get Score For Training random forest and data transformations with Word2Vec
# getResultFromWord2VecRandomForestClassifier(crossValidator=True) #get Score For Training random forest and data transformations with Word2Vec and CrossValidation
