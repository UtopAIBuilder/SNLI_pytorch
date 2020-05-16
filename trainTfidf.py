#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import torch


# In[4]:


TRAIN_PATH = '.data/snli/snli_1.0/snli_1.0_train.txt'
#DEV_PATH = '.data/snli/snli_1.0/snli_1.0_dev.txt'
#TEST_PATH = '.data/snli/snli_1.0/snli_1.0_test.txt'


# In[8]:


train_df = pd.read_csv(TRAIN_PATH, sep='\t', keep_default_na=False)
#dev_df = pd.read_csv(DEV_PATH, sep='\t', keep_default_na=False)
#test_df = pd.read_csv(TEST_PATH, sep='\t', keep_default_na=False)
print(f'Number of train examples:',len(train_df))


# In[9]:


#print(dev_df)


# In[10]:


def df_to_list(df):
    return list(zip(df['sentence1'], df['sentence2'], df['gold_label']))

train_data = df_to_list(train_df)
#dev_data = df_to_list(dev_df)
#test_data = df_to_list(test_df)
train_df,dev_df,test_df=0,0,0


# In[62]:


#print(dev_data)


# In[12]:


def filter_no_consensus(data):
    return [(sent1, sent2, label) for (sent1, sent2, label) in data if label != '-']

print(f'Examples before filtering:',len(train_data))
train_data = filter_no_consensus(train_data)
#dev_data = filter_no_consensus(dev_data)
#test_data = filter_no_consensus(test_data)
print(f'Examples after filtering:',len(train_data))


# In[ ]:





# In[14]:


import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import re
#nltk.download('stopwords')
nltk.download('wordnet')


# In[15]:


wordnet=WordNetLemmatizer()
#stopwordset=set(stopwords.words('english'))
def tokenize(string):
  review=re.sub('[^a-zA-Z]',' ',string)
  review=review.lower()
  review=review.split()
  review=[wordnet.lemmatize(word) for word in review]
  return ' '.join(review)
def tokenize_data(data):
    return [(tokenize(sent1), tokenize(sent2), label) for (sent1, sent2, label) in tqdm(data)]
train_data = tokenize_data(train_data)
#dev_data = tokenize_data(dev_data)
#test_data = tokenize_data(test_data)


# TF-IFD prepration for train data

# In[16]:


def lableMap(label):
  if(label=='entailment'):
    return 1
  if(label=='neutral'):
    return 3
  if(label=='contradiction'):
    return 2
  print('something is freaking wrong with labels')


# In[17]:


def printOutputInFile(labels):
    f=open("tfidf.txt","w+")
    for i in labels:
        if(i==1):
            f.write("entailment\r\n")
        elif(i==3):
            f.write("neutral\r\n")
        elif(i==2):
            f.write("contradiction\r\n")
        else:
            print("dude you got crap as a prediction label\n")


# In[18]:


def corpus(data):
  corpus1=[]
  corpus2=[]
  labels=[]
  for (sent1, sent2, label) in tqdm(data):
    corpus1.append(sent1)
    corpus2.append(sent2)
    labels.append(lableMap(label))
  return corpus1,corpus2,np.array(labels)


# In[19]:


corpus1,corpus2,labels=corpus(train_data)
cv1=TfidfVectorizer(ngram_range=(1, 2))
cv2=TfidfVectorizer(ngram_range=(1, 2))
corpus1=cv1.fit_transform(corpus1)
corpus2=cv2.fit_transform(corpus2)


# In[20]:


from scipy.sparse import hstack
corpus1=hstack([corpus1,corpus2])
corpus2=0


# In[21]:


torch.save(cv1,'cv1.sav')
torch.save(cv2,'cv2.sav')


# In[ ]:





# In[22]:


from sklearn.linear_model import LogisticRegression
import joblib
logmodel=LogisticRegression(solver='saga')
logmodel.fit(corpus1,labels)
joblib.dump(logmodel,'logmodel.sav')


# In[ ]:





# Validation on Dev Data

# In[23]:


'''corpus1,corpus2,labels=corpus(dev_data)
#print(type(corpus1),type(corpus2),len(corpus1),len(corpus2))
corpus1=cv1_l.transform(corpus1)
corpus2=cv2_l.transform(corpus2)
#print(type(corpus1),type(corpus2))
corpus1=hstack([corpus1,corpus2])
corpus2=0
pred_labels=logload.predict(corpus1)
acc=np.mean(labels==pred_labels)
print(acc)
#print(np.concatenate((labels.reshape(-1,1),pred_labels.reshape(-1,1)),axis=1)[:10])'''


# testData accuracy
# 

# In[24]:


'''corpus1,corpus2,labels=corpus(test_data)
#print(labels[:10])
#print(type(corpus1),type(corpus2),len(corpus1),len(corpus2))
corpus1=cv1.transform(corpus1)
corpus2=cv2.transform(corpus2)
#print(type(corpus1),type(corpus2))
corpus1=hstack([corpus1,corpus2])
corpus2=0
pred_labels=logmodel.predict(corpus1)
acc=np.mean(labels==pred_labels)
print(acc)
#print(np.concatenate((labels.reshape(-1,1),pred_labels.reshape(-1,1)),axis=1)[:10])'''


# In[20]:


#printOutputInFile(pred_labels)


# In[25]:


#type(pred_labels)


# In[ ]:


#a,b,l=corpus(test_data[:5])
#print(l.shape)


# In[ ]:




