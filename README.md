# Sentiment-Analysis

<h1>Introduction</h1>
This project is based on the famous bag of words kaggle problem, which analyses the sentiment of the IMDB movies review dataset.

The problem was solved using pyspark on databricks using different supervised learning algorithm. 

<h1>Work Pipeline</h1>

![alt text](/Snapshots/1.png)

<h1>Data Pre Processing</h1>
The review field in the file contains data like HTML tags, garbage values, and other symbols. This data affects the accuracy of the model, thus needs to be cleaned first. For cleaning the data, we have used ‘beautifulsoup4’, an open source Python data analysis library.
We had to decide whether to include the syllables like punctuation marks and smiles, which may hold meaning to the reviews. But for the simplicity, we discarded such entities.

Training data is a .csv file with three columns. The first column is the unique ID, then its corresponding sentiment value and the last is the review.
Sentiment 1 implies that the review is positive while 0 shows a negative review

Spark provides a handful of these algorithms such as TF- IDF (Term frequency and Inverse Document Frequency) , count vectorization, Word2Vec, N-grams and few more. 
For the project, we used two algorithms, TF-IDF and Word2Vec and after the features are created, we used each algorithm of ML to above two feature creation methods and compared the evaluation metric results for both of these methods.

<h1>Observations on running ML Models</h1>

In each of the following models 5 fold cross validation was used. AUC and F1 score was obtained from each model.

<h2>Random Forest</h2>
Area under curve obtained: 86%

<h2>Support Vector Machines</h2>
F1 Score: 80%

<h2>Neural Networks</h2>
F1 Score: 85%

<h2>Logistic Regression</h2>
F1 Score: 90%


<h1>Kibana Data Analysis</h1>
The data was processord using ELK and kibana was used to project the importance of positive and negative words in projecting the sentiment of the reviews. For example as shown below the presence of word 'best' can project the possibility of the review to be positive with the probability of around 65%. Similarly the usage of word 'waste' can surely means the review was bad with probability of aroung 95%.

![alt text](/Snapshots/2.png)

![alt text](/Snapshots/3.png)

If you have any questions, please feel free to reach out to me at erajaypal91@gmail.com
