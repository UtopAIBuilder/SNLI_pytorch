#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import torch
import torchtext
from torchtext.data import Field, BucketIterator
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import nli_model as nli


# In[3]:


from torchtext import data
from torchtext import datasets


# In[3]:



print('welcome to Testing...')
import nltk
from nltk.stem import WordNetLemmatizer
import re
nltk.download('wordnet')
wordnet=WordNetLemmatizer()
#stopwordset=set(stopwords.words('english'))
def tokenizer(string):
  review=re.sub('[^a-zA-Z]',' ',string)
  #review=review.lower()
  review=review.split()
  review=[wordnet.lemmatize(word) for word in review]
  return review

sentences = data.Field(lower=True, tokenize=tokenizer)
ans = data.Field(sequential=False)
print('downloading data...')
train, dev, test = datasets.SNLI.splits(sentences, ans)
print('Done!')

# In[28]:

print('building vocab..')
sentences.build_vocab(train, dev, test,min_freq=3)
ans.build_vocab(train, dev, test)
print('vocab Length:',len(sentences.vocab))
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
#print('device',device)    
Batch_Size=64
train_iter, dev_iter, test_iter = data.BucketIterator.splits(
            (train, dev, test), batch_size=Batch_Size, device=device)

print('Done!')


def printOutputInFile(labels,filename):
    f=open('outputDocs/'+filename,"w+")
    for i in labels:
        if(i==1):
            f.write("entailment\r\n")
        elif(i==3):
            f.write("neutral\r\n")
        elif(i==2):
            f.write("contradiction\r\n")
        else:
            print("dude you got crap as a prediction label\n")
    f.close()
    print('Output File Created '+filename)


# In[17]:


def accuracy(model,train_loader):
    model.eval()
    running_corrects=0.0
    running_loss=0.0
    total=0.0
    pred_final=[]
    with torch.no_grad():
        for inputs in train_loader:
            #inputs=inputs.to(device)
            #labels=labels.to(device)
            output=model(inputs)
            _,pred=torch.max(output, 1)
            pred_final.append(pred)
            running_corrects += torch.sum(pred == inputs.label)
            total+=inputs.batch_size
    print('Deep Model Accuracy: {:.6f}'.format((running_corrects/total)))
    return running_corrects/total,torch.cat(pred_final,0)


# In[9]:

print('Loading Deep Model')
model=nli.SnliClassifier(len(sentences.vocab))
model.to(device)
model.load_state_dict(torch.load("models/NLI.pt",map_location=device))
print('Done!')

# In[18]:


acc,pred_final=accuracy(model,test_iter)


# In[27]:


printOutputInFile(pred_final,"deep_model.txt")


# # Testing tfidf Now and writing the output file

# In[5]:


TEST_PATH = '.data/snli/snli_1.0/snli_1.0_test.txt'


# In[6]:


test_df = pd.read_csv(TEST_PATH, sep='\t', keep_default_na=False)
#print(f'test examples:',len(test_df))


# In[7]:


def df_to_list(df):
    return list(zip(df['sentence1'], df['sentence2'], df['gold_label']))

test_data = df_to_list(test_df)
test_df=0


# In[8]:


def filter_no_consensus(data):
    return [(sent1, sent2, label) for (sent1, sent2, label) in data if label != '-']

#print(f'Examples before filtering:',len(test_data))
test_data = filter_no_consensus(test_data)
#print(f'Examples after filtering:',len(test_data))


# In[9]:


from tqdm import tqdm
#nltk.download('stopwords')


# In[10]:


#wordnet=WordNetLemmatizer()
#stopwordset=set(stopwords.words('english'))
def tokenize(string):
  review=re.sub('[^a-zA-Z]',' ',string)
  review=review.lower()
  review=review.split()
  review=[wordnet.lemmatize(word) for word in review]
  return ' '.join(review)
def tokenize_data(data):
    return [(tokenize(sent1), tokenize(sent2), label) for (sent1, sent2, label) in data]
test_data = tokenize_data(test_data)


# In[11]:


def lableMap(label):
  if(label=='entailment'):
    return 1
  if(label=='neutral'):
    return 3
  if(label=='contradiction'):
    return 2
  print('something is freaking wrong with labels')


# In[12]:


def corpus(data):
  corpus1=[]
  corpus2=[]
  labels=[]
  for (sent1, sent2, label) in data:
    corpus1.append(sent1)
    corpus2.append(sent2)
    labels.append(lableMap(label))
  return corpus1,corpus2,np.array(labels)


# In[13]:


import joblib
cv1=torch.load('models/cv1.sav')
cv2=torch.load('models/cv2.sav')
logmodel=joblib.load('models/logmodel.sav')


# In[14]:


from scipy.sparse import hstack
corpus1,corpus2,labels=corpus(test_data)
#print(labels[:10])
#print(type(corpus1),type(corpus2),len(corpus1),len(corpus2))
corpus1=cv1.transform(corpus1)
corpus2=cv2.transform(corpus2)
#print(type(corpus1),type(corpus2))
corpus1=hstack([corpus1,corpus2])
corpus2=0
pred_labels=logmodel.predict(corpus1)
acc=np.mean(labels==pred_labels)
print('Logistic Model Accuracy:',acc)
#print(np.concatenate((labels.reshape(-1,1),pred_labels.reshape(-1,1)),axis=1)[:10])


# In[15]:


printOutputInFile(pred_labels,"tfidf.txt")


# In[ ]:




