import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

def removeTags(review) :
    text = BeautifulSoup(review)
    letters_only = re.sub("[^a-zA-Z]", " ", text.getText())
    return str(letters_only)
    
def removeStops(review) :
    words = review.lower().split()
    stops = set(stopwords.words("english"))
    meaningfulwords = [w for w in words if not w in stops]
    return " ".join(meaningfulwords)        
    
train2 = pd.read_csv("Data/TagsRemoved.csv", header=0,delimiter="\t", quoting=3)
#print removeTags(train['review'][0])
#train2['review'] = train2['review'].map(removeTags)
#train2.to_csv('Data/TagsRemoved.csv',sep='\t',index=False)
#print train2['review'][0]
size = len(train2)
clean_train = []
for i in range(0, size):
    clean_train.append( removeStops( train2["review"][i] ) )

vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 1000) 
train_data_features = vectorizer.fit_transform(clean_train) 
finaldata = pd.DataFrame(train_data_features.toarray())
finaldata['sentiment'] = train2['sentiment']
finaldata.to_csv("data.csv",index=False)
                            