
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import scipy as scp


# In[2]:

from nltk.tokenize import word_tokenize as wt
import string
code_word_list = ['#','+++', 'defined', 'if', 'elseif', 'elif', 'else', 'exit', '0', 'define','char','int', 'float', 'html', '<', '>', '<table>' , '()', '::', 'unsigned', 'signed', 'long', 'double', '.cpp', '.py','{' ,'}', 'for(', 'while(', "=", "==", "++","--","+"]
#code_file = open('code_file.txt', "w+")
#nl_file = open('nl_file.txt', "w+")
def is_line_code(line):
    temp = line.decode('utf-8').split("::")
    if(len(temp)>1):
        return True
    code_count = len(temp)
    line = wt(" ".join(temp))    
    if (len(line) <= 2):
        return False;
    for word in line:
        if word.lower() in code_word_list or len(word) > 9 or word[:2] == "0x":
            code_count+=1
    if (code_count+1)/float(len(line)+1) >= 0.5:        
        return True
    else:        
        return False
        
def seperate_code(doc):
    lines = doc.split('\n')
    nl, code = [], []
    
    for line in lines:
        if "Additional Details".lower() in line.lower() or "Updated by".lower() in line.lower() or "Created by".lower() in line.lower(): 
            continue
        if is_line_code(line):
            if line != "":
                code.append(line)
                #code_file.write(line + '\n')

        else:
            if line!="":
                nl.append(line)
                #nl_file.write(line + '\n')

    return (nl, code)
    


# In[28]:

from ast import literal_eval as le
import re
import string
from pandas import DataFrame
import pandas
import time
from nltk.tokenize import word_tokenize as wt
f = open('MozillaBugs.json')
cols = ('bug_id', 'title', 'description', 'code', 'dup_id')
df = DataFrame(columns=cols)
row_num = 0
c_lines = 0
n_lines = 0
t = time.time()
l = []
for line in f:
    line = line.strip()
    bug = le(line)
    dup_id = bug['dup_id']
    nl = None
    code = None
    title = None
    if dup_id != '[]' and dup_id != []:
        #print dup_id
        dup_id = int(dup_id)    
    else:
        dup_id = None    
    if 'description' in bug and bug['description']!=[]:
        nl, code = seperate_code(bug['description'])
        c_lines += len(code)
        n_lines += len(nl)
        
    if 'short_desc' in bug:
        title = " ".join(wt(bug['short_desc'].decode('utf-8')))
    row_num+=1
    if type(nl) == type("a"):
        nl = nl.split()
    if type(code) == type("a"):
        code = code.split()
    if nl != None:
        nl = " ".join(wt(" ".join(nl).decode('utf-8')))
    if code != None:
        code = " ".join(wt(" ".join(code).decode('utf-8')))
    temp = [bug['bug_id'], title, nl, code, dup_id]
    l.append(temp)
    if(len(temp) != 5):
        print t
        break
        
    if(row_num%5000 == 0):
        print row_num, 'done', time.time() - t
        t = time.time()
        #try:
        df2 = pandas.DataFrame(l, columns = ('bug_id', 'title', 'description' ,'code', 'dup_id'))
        #except AssertionError:
        #    for i in l:
        #        if len(i) != 5:
        #            print i
            
        df = df.append(df2)
        l = []
df2 = pandas.DataFrame(l, columns = cols)
df = df.append(df2)        
df.to_csv('SegmentedDataset.csv', index='bug_id')
#nl_file.close()
#code_file.close()


# In[32]:

len(df)


# In[33]:

df.head()


# In[34]:

dups = df[pd.notnull(df['dup_id'])]


# In[35]:

non_dups = df[pd.isnull(df['dup_id'])]


# In[36]:

print len(dups)
print len(non_dups)


# In[37]:

non_dups.head()


# In[43]:

dups[['bug_id']] = dups[['bug_id']].apply(pd.to_numeric)
df[['bug_id']] = df[['bug_id']].apply(pd.to_numeric)
dup_pairs = pandas.merge(dups, df,  how='inner', left_on=['dup_id'], right_on = ['bug_id'])
dup_pairs2 = pandas.merge(dups, dups,  how='inner', left_on=['dup_id'], right_on = ['bug_id'])


# In[48]:

print len(dup_pairs)
print len(dup_pairs2)


# In[ ]:




# In[47]:

from sklearn.model_selection import train_test_split
train, test = train_test_split(dup_pairs, test_size = 15000)
test, val = train_test_split(test, test_size = 5000)


# In[49]:

print len(train)
print len(val)
print len(test)


# In[56]:

train.to_csv('DupsTrain.csv', index='bug_id',encoding='utf-8',sep="\t")
val.to_csv('DupsVal.csv', index='bug_id',encoding='utf-8',sep="\t")
test.to_csv('DupsTest.csv', index='bug_id',encoding='utf-8',sep="\t")


# In[57]:

non_dups.to_csv('Non_Dups.csv', index = 'bug_id',encoding='utf-8',sep="\t")


# In[60]:

train.head()


# In[ ]:



