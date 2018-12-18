# ML
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import bs4
from bs4 import BeautifulSoup
import re
import string
###TRAINING DATA----
path="D:/DS/aclImdb_v1.tar/aclImdb/train/pos"
data=[]
files=[path+'/'+ f for f in os.listdir(path) if os.path.isfile(path+'/'+ f)]
for f in files:
    with open (f,"r",encoding="utf8")as myfile:
        data.append(myfile.read())
df_trainpos=pd.DataFrame(data,columns=['review'])    
df_trainpos['label']="1"
path="D:/DS/aclImdb_v1.tar/aclImdb/train/neg"
data1=[]
files=[path+'/'+ f for f in os.listdir(path) if os.path.isfile(path+'/'+ f)]
for f in files:
    with open (f,"r",encoding="utf8")as myfile:
        data1.append(myfile.read())
df_trainneg=pd.DataFrame({'review':data1})
df_trainneg['label']="0"
df_trainneg.head()
train_df =pd.concat([df_trainneg,df_trainpos],axis=0)

###--------------------------------------------------
###TESTING DATA----
path="D:/DS/aclImdb_v1.tar/aclImdb/test/pos"
data2=[]
files=[path+'/'+ f for f in os.listdir(path) if os.path.isfile(path+'/'+ f)]
for f in files:
    with open (f,"r",encoding="utf8")as myfile:
        data2.append(myfile.read())
df_testpos=pd.DataFrame(data2,columns=['review'])    
df_testpos['label']="1"
path="D:/DS/aclImdb_v1.tar/aclImdb/test/neg"
data3=[]
files=[path+'/'+ f for f in os.listdir(path) if os.path.isfile(path+'/'+ f)]
for f in files:
    with open (f,"r",encoding="utf8")as myfile:
        data3.append(myfile.read())
df_testneg=pd.DataFrame({'review':data3})
df_testneg['label']="0"
df_testneg.head()
test_df =pd.concat([df_testneg,df_testpos],axis=0)
test_df.head()
###--------STOP-WORDS REMOVAL
stop_words = set(stopwords.words('english')) 
test_df['review'] = test_df['review'].str.split()
test_df['review']=test_df['review'].apply(lambda x: [word for word in x if word not in stop_words])

train_df['review'] = train_df['review'].str.split()
train_df['review'] =train_df['review'].apply(lambda x: [word for word in x if word not in stop_words])

###-----------
####----Removing punctuations, HTML tags (like br) etc.

def stripTags(x):
        # BeautifulSoup on content
        soup = BeautifulSoup(str(x), "html.parser")
        # Stripping all <code> tags with their content if any
        if soup.code:
            soup.code.decompose()
        # Get all the text out of the html
        text =  soup.get_text()
        return text
train_df['review'] = train_df['review'].map(stripTags)
test_df['review'] = test_df['review'].map(stripTags)
train_df['review']=train_df['review'].apply(lambda x:" ".join([BeautifulSoup(str(word)).getText() for word in x.split()]))
#test_df['review'] = [BeautifulSoup(str(X)).getText() for X in train_df['review']]
def remove_punctuation(x):
    # Removing non ASCII chars
    x = re.sub("[^\x00-\x7f]", " ",str(x))
    x = re.sub("<[^<]+?>", " ",str(x))
    
    # Removing (replacing with empty spaces actually) all the punctuations
    return re.sub("["+string.punctuation+"]", " ", x)
df_testneg['review'] = df_testneg['review'].apply(remove_punctuation)
train_df['review'] = train_df['review'].apply(remove_punctuation)
test_df['review'] = test_df['review'].apply(remove_punctuation)
#print(train_df.head(50))
#####-------------------------------------------
from nltk.stem import PorterStemmer
stemmer = PorterStemmer() 
#print(train_df.head())
train_df['review']=train_df['review'].apply(lambda x:" ".join([stemmer.stem(word) for word in x.split()]))
test_df['review']=test_df['review'].apply(lambda x:" ".join([stemmer.stem(word) for word in x.split()]))
#print(train_df.tail(10))
from nltk.stem.wordnet import WordNetLemmatizer 
lem = WordNetLemmatizer()
train_df['review']=train_df['review'].apply(lambda x:" ".join([lem.lemmatize(word) for word in x.split()]))
test_df['review']=test_df['review'].apply(lambda x:" ".join([lem.lemmatize(word) for word in x.split()]))
#print(train_df.tail(15))
###-------------
freq_trainneg = pd.Series(' '.join(df_trainneg['review']).split()).value_counts()
max=freq_trainneg.max()
freq_trainpos =  pd.Series(' '.join(df_trainpos['review']).split()).value_counts()
freq_testpos = pd.Series(' '.join(df_testpos['review']).split()).value_counts()
freq_testneg = pd.Series(' '.join(df_testneg['review']).split()).value_counts()

print(freq_trainneg)
print(max)
