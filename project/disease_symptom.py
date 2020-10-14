import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import re
import nltk
#nltk.download('averaged_perceptron_tagger')
  
url="http://people.dbmi.columbia.edu/~friedma/Projects/DiseaseSymptomKB/index.html"
r = requests.get(url) 
#print(r.content) 

html=BeautifulSoup(r.content,"html5lib")
#print(html.prettify())

disease_directory=[]
table=html.find("table",attrs={'class':'MsoTableWeb3'})

for row in table.findAll('tr'):
    qoute={}
    qoute['disease']=row.text
    
#     if row.text.isdigit():
#         continue 
#     #elif row.text !='\xa0':
#     qoute["text"]=row.text
    disease_directory.append(qoute)

#print(disease_directory)
df=pd.DataFrame(disease_directory)
#print(df.head())

new_df=df.disease.apply(lambda x: pd.Series(str(x).split("\n  \n  \n  ")))
new_df.rename(columns={0: "Disease",1: "Weight",2:"Symptoms"},inplace=True)
# new_df['Disease'].replace(regex=True,inplace=True,to_replace=r'\n  \n  ',value=r'')
# new_df['Symptoms'].replace(regex=True,inplace=True,to_replace=r'\n  \n ',value=r'')

new_df['Disease']=[re.sub(r'(\s+|\n)', ' ', i ) for i in new_df['Disease']]
new_df['Symptoms']=[re.sub(r'(\s+|\n)', ' ', i ) for i in new_df['Symptoms']]
new_df['Weight']=[re.sub(r'(\s+|\n)', ' ', i ) for i in new_df['Weight']]

new_df['Disease']=new_df.Disease.str.strip()
new_df['Symptoms']=new_df.Symptoms.str.strip()
header=new_df.iloc[0]

new_df=new_df[1:]
new_df.columns=header
#print(new_df.head())



# new_df.to_csv("db.csv",index=False)

import csv
from collections import defaultdict

disease_list = []

def return_list(disease):
    disease_list = []
    match = disease.replace('^','_').split('_')
    ctr = 1
    for group in match:
        if ctr%2==0:
            disease_list.append(group)
        ctr = ctr + 1

    return disease_list

disease=""
weight = 0
disease_list = []
dict_wt = {}
dict_=defaultdict(list)
for row in new_df.index:

    if new_df['Disease'][row]!="\xc2\xa0" and new_df['Disease'][row]!="":
        disease = new_df['Disease'][row]
        disease_list = return_list(disease)
        weight = new_df['Count of Disease Occurrence'][row]

    if new_df['Symptom'][row]!="\xc2\xa0" and new_df['Symptom'][row]!="":
        symptom_list = return_list(new_df['Symptom'][row])

        for d in disease_list:
            for s in symptom_list:
                dict_[d].append(s)
            dict_wt[d] = weight

    #print (dict_)
database_list=[]
for key,values in dict_.items():
    for v in values:
        #key = str.encode(key)
        key = str.encode(key).decode('utf-8')
        #.strip()
        #v = v.encode('utf-8').strip()
        #v = str.encode(v)
        database_list.append([key,v,dict_wt[key]])
        
columns = ['Source','Target','Weight']
data = pd.DataFrame(database_list, columns=['Source','Target','Weight'])
#print(data.head())

import nltk
from nltk.stem import WordNetLemmatizer
lematizer=WordNetLemmatizer()
from fuzzywuzzy import process
df=data
df["Target"].fillna("NA", inplace = True)
df["Frequency"]=0
 
def Sentence_preprocessing(text):
    tokens=nltk.word_tokenize(text)
    postag=nltk.pos_tag(tokens)
    #print(postag)
    grammar = r""" symp: {<NN.?>*<NNP>?<JJ>?}
                """
    chunkParser = nltk.RegexpParser(grammar)
    tree = chunkParser.parse(postag)
    symp=[]
    for subtree in tree.subtrees(filter=lambda t: t.label() == 'symp'):
        symp.append(str(subtree))
    pattern = r'\w+:?(?=\/)'
     
    lis_of_symp=[re.findall(pattern,i) for i in symp]
    list_of_symp=[]
    for i in lis_of_symp:
        if len(i)>1:
            two_word_symp=''.join(i)
            list_of_symp.append(two_word_symp)
        else:
            for item in i:
                list_of_symp.append(item) 
             
    return list_of_symp
 
 
 
def fuzy_matching(text):
    Processed_text=Sentence_preprocessing(text)
    str2Match = Processed_text
    #print(str2Match)
    strOptions = df["Target"]
    high_match=[]
    for i in str2Match:
        highest = process.extractOne(i,strOptions)
        highest=list(highest)
        if highest[1]>=80:
            high_match.append(highest[0])
    return high_match
 
 
 
#
 
def disease_finder(text):
    symptom_list=fuzy_matching(text)
    df_dis_filtered=df[df.Target.str.contains('|'.join(symptom_list))]
    dis= df_dis_filtered.groupby("Source").count()
    dis=dis.sort_values(by=["Target"],ascending=False)
    dis=dis.head(5)
     
    return dis
 
def weight_gain(text):
    dis_list=list(disease_finder(text))
    #print(dis_list)
    for i in dis_list:
        df.loc[df["Source"]==i,"Frequency"] += 1   
     
    dis=df.sort_values(by=["Frequency"],ascending=False)
    dis=dis.groupby("Source").count()
    dis=dis.sort_values(by=["Target"],ascending=False)
    #dis=dis.head(5)
     
    return dis
 
def symptoms_recommendation(text):
    dis_list=list(disease_finder(text))
    symptom_list=fuzy_matching(text)
    symp_recom=df[df.Source.str.contains('|'.join(dis_list))]
    symp_recom=list(symp_recom["Target"])
    symp_recom=list(set(symp_recom))
    symp_recom=[i for i in symp_recom if i not in symptom_list]
    symp_recom=symp_recom
    return symp_recom