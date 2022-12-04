#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gensim


# In[2]:


import pandas as pd


# In[3]:


from nltk.corpus import wordnet


# In[4]:


import nltk


# In[5]:


from nltk.corpus import stopwords


# In[6]:


import numpy as np


# In[7]:


col_names = ["user", "date", "rating", "title", "content"]
df = pd.read_csv('C:\\Users\\Ilmor SM\\Documents\\thesis\\Review Per Hotel\\Bintang 5\lotte.csv', sep=';')


# In[8]:


df.head()


# In[9]:


df = df.rename(columns={'user ': 'user'})
df = df.rename(columns={'date ': 'date'})
df = df.rename(columns={'rating ': 'rating'})
df = df.rename(columns={'title ': 'title'})
df = df.rename(columns={'content ': 'content'})


# In[10]:


df['title']=df['title'].str.lower()


# In[11]:


df['content']=df['content'].str.lower()


# In[12]:


df.head()


# In[13]:


nltk.download('stopwords')


# In[14]:


" ".join(stopwords.words('english'))
stop_words = set(stopwords.words('english'))


# In[15]:


def remove_stop(x):
    return " ".join([word for word in str(x).split() if word not in stop_words])
df['title']=df['title'].apply(lambda x : remove_stop(x))


# In[16]:


def remove_stop(x):
    return " ".join([word for word in str(x).split() if word not in stop_words])
df['content']=df['content'].apply(lambda x : remove_stop(x))


# In[17]:


df.describe()


# In[18]:


df.head()


# In[19]:


df.content[5]


# In[20]:


review_text = df.content.apply(gensim.utils.simple_preprocess)
review_text


# In[21]:


model = gensim.models.Word2Vec()


# In[22]:


model.build_vocab(review_text,progress_per=1000)


# In[23]:


model.epochs


# In[24]:


model.train(review_text, total_examples=model.corpus_count, epochs=model.epochs)


# In[25]:


model.save("./Word2Vec-tripadvisor-reviews-short.model")


# # Dimensi Sincerity

# In[56]:


#ngetes
wordnet.synsets("continued", pos = wordnet.ADJ)


# In[71]:


trait1=model.wv.most_similar("honest", topn=35)
simtrait1=[]
for i in range(0,len(trait1)):
    if wordnet.synsets(trait1[i][0], pos = wordnet.ADJ) != []:
        simtrait1.append(trait1[i][0])
    else:
        continue
sim1=list(simtrait1)
print(sim1)        


# In[72]:


synonyms1 = []
for syn in wordnet.synsets("honest", pos = wordnet.ADJ):
    for l in syn.lemmas():
        synonyms1.append(l.name()) 
syn1 = list(synonyms1)        
print(syn1)        


# In[88]:


for a in range(0,len(sim1)):
    for b in range(0,len(syn1)):
        try:
            tes=model.wv.similarity(w1=sim1[a],w2=syn1[b])
        except KeyError: 
            pass
        if tes > 0.5:
            print(sim1[a], syn1[b], tes)
        else: 
            pass
        


# In[96]:


trait5=model.wv.most_similar("sincere", topn=51)
simtrait5=[]
for i in range(0,len(trait5)):   
    if wordnet.synsets(trait5[i][0], pos = wordnet.ADJ) != []:
        simtrait5.append(trait5[i][0])
    else:
        continue
sim5=list(simtrait5)
print(sim5) 


# In[98]:


synonyms5 = []
for syn in wordnet.synsets("sincere", pos = wordnet.ADJ):
    for l in syn.lemmas():
        synonyms5.append(l.name())
syn5 = list(synonyms5)        
print(syn5)   


# In[99]:


for a in range(0,len(sim5)):
    for b in range(0,len(syn5)):
        try:
            tes=model.wv.similarity(w1=sim5[a],w2=syn5[b])
        except KeyError: 
            pass
        if tes > 0.5:
            print(sim5[a], syn5[b], tes)
        else: 
            pass


# In[102]:


trait6=model.wv.most_similar("real", topn=40)
simtrait6=[]
for i in range(0,len(trait6)):   
    if wordnet.synsets(trait6[i][0], pos = wordnet.ADJ) != []:
        simtrait6.append(trait6[i][0])
    else:
        continue
sim6=list(simtrait6)
print(sim6) 


# In[66]:


synonyms6 = []
for syn in wordnet.synsets("real", pos = wordnet.ADJ):
    for l in syn.lemmas():
        synonyms6.append(l.name())
syn6 = list(synonyms6)        
print(syn6)   


# In[103]:


for a in range(0,len(sim6)):
    for b in range(0,len(syn6)):
        try:
            tes=model.wv.similarity(w1=sim6[a],w2=syn6[b])
        except KeyError: 
            pass
        if tes > 0.5:
            print(sim6[a], syn6[b], tes)
        else: 
            pass


# In[106]:


trait8=model.wv.most_similar("original", topn=20)
simtrait8=[]
for i in range(0,len(trait8)):   
    if wordnet.synsets(trait8[i][0], pos = wordnet.ADJ) != []:
        simtrait8.append(trait8[i][0])
    else:
        continue
sim8=list(simtrait8)
print(sim8) 


# In[107]:


synonyms8 = []
for syn in wordnet.synsets("original", pos = wordnet.ADJ):
    for l in syn.lemmas():
        synonyms8.append(l.name())
syn8 = list(synonyms8)        
print(syn8)   


# In[108]:


for a in range(0,len(sim8)):
    for b in range(0,len(syn8)):
        try:
            tes=model.wv.similarity(w1=sim8[a],w2=syn8[b])
        except KeyError: 
            pass
        if tes > 0.5:
            print(sim8[a], syn8[b], tes)
        else: 
            pass


# In[113]:


trait9=model.wv.most_similar("cheerful", topn=28)
simtrait9=[]
for i in range(0,len(trait9)):   
    if wordnet.synsets(trait9[i][0], pos = wordnet.ADJ) != []:
        simtrait9.append(trait9[i][0])
    else:
        continue
sim9=list(simtrait9)
print(sim9) 


# In[115]:


synonyms9 = []
for syn in wordnet.synsets("cheerful", pos = wordnet.ADJ):
    for l in syn.lemmas():
        synonyms9.append(l.name())
syn9 = list(synonyms9)        
print(syn9)  


# In[116]:


for a in range(0,len(sim9)):
    for b in range(0,len(syn9)):
        try:
            tes=model.wv.similarity(w1=sim9[a],w2=syn9[b])
        except KeyError: 
            pass
        if tes > 0.5:
            print(sim9[a], syn9[b], tes)
        else: 
            pass


# In[119]:


trait10=model.wv.most_similar("friendly", topn=12)
simtrait10=[]
for i in range(0,len(trait10)):   
    if wordnet.synsets(trait10[i][0], pos = wordnet.ADJ) != []:
        simtrait10.append(trait10[i][0])
    else:
        continue
sim10=list(simtrait10)
print(sim10) 


# In[121]:


synonyms10 = []
for syn in wordnet.synsets("friendly", pos = wordnet.ADJ):
    for l in syn.lemmas():
        synonyms10.append(l.name())
syn10 = list(synonyms10)        
print(syn10)  


# In[122]:


for a in range(0,len(sim10)):
    for b in range(0,len(syn10)):
        try:
            tes=model.wv.similarity(w1=sim10[a],w2=syn10[b])
        except KeyError: 
            pass
        if tes > 0.5:
            print(sim10[a], syn10[b], tes)
        else: 
            pass


# # Dimensi Excitement

# In[125]:


trait22=model.wv.most_similar("trendy", topn=29)
simtrait22=[]
for i in range(0,len(trait22)):   
    if wordnet.synsets(trait22[i][0], pos = wordnet.ADJ) != []:
        simtrait22.append(trait22[i][0])
    else:
        continue
sim22=list(simtrait22)
print(sim22) 


# In[126]:


synonyms22 = []
for syn in wordnet.synsets("trendy", pos = wordnet.ADJ):
    for l in syn.lemmas():
        synonyms22.append(l.name())
syn22 = list(synonyms22)        
print(syn22)  


# In[127]:


for a in range(0,len(sim22)):
    for b in range(0,len(syn22)):
        try:
            tes=model.wv.similarity(w1=sim22[a],w2=syn22[b])
        except KeyError: 
            pass
        if tes > 0.5:
            print(sim22[a], syn22[b], tes)
        else: 
            pass


# In[132]:


trait23=model.wv.most_similar("exciting", topn=55)
simtrait23=[]
for i in range(0,len(trait23)):   
    if wordnet.synsets(trait23[i][0], pos = wordnet.ADJ) != []:
        simtrait23.append(trait23[i][0])
    else:
        continue
sim23=list(simtrait23)
print(sim23) 


# In[133]:


synonyms23 = []
for syn in wordnet.synsets("exciting", pos = wordnet.ADJ):
    for l in syn.lemmas():
        synonyms23.append(l.name())
syn23 = list(synonyms23)        
print(syn23)  


# In[135]:


for a in range(0,len(sim23)):
    for b in range(0,len(syn23)):
        try:
            tes=model.wv.similarity(w1=sim23[a],w2=syn23[b])
        except KeyError: 
            pass
        if tes > 0.5:
            print(sim23[a], syn23[b], tes)
        else: 
            pass


# In[139]:


trait25=model.wv.most_similar("cool", topn=26)
simtrait25=[]
for i in range(0,len(trait25)):   
    if wordnet.synsets(trait25[i][0], pos = wordnet.ADJ) != []:
        simtrait25.append(trait25[i][0])
    else:
        continue
sim25=list(simtrait25)
print(sim25) 


# In[140]:


synonyms25 = []
for syn in wordnet.synsets("cool", pos = wordnet.ADJ):
    for l in syn.lemmas():
        synonyms25.append(l.name())
syn25 = list(synonyms25)        
print(syn25)  


# In[141]:


for a in range(0,len(sim25)):
    for b in range(0,len(syn25)):
        try:
            tes=model.wv.similarity(w1=sim25[a],w2=syn25[b])
        except KeyError: 
            pass
        if tes > 0.5:
            print(sim25[a], syn25[b], tes)
        else: 
            pass


# In[143]:


trait26=model.wv.most_similar("young", topn=46)
simtrait26=[]
for i in range(0,len(trait26)):   
    if wordnet.synsets(trait26[i][0], pos = wordnet.ADJ) != []:
        simtrait26.append(trait26[i][0])
    else:
        continue
sim26=list(simtrait26)
print(sim26) 


# In[144]:


synonyms26 = []
for syn in wordnet.synsets("young", pos = wordnet.ADJ):
    for l in syn.lemmas():
        synonyms26.append(l.name())
syn26 = list(synonyms26)        
print(syn26)  


# In[145]:


for a in range(0,len(sim26)):
    for b in range(0,len(syn26)):
        try:
            tes=model.wv.similarity(w1=sim26[a],w2=syn26[b])
        except KeyError: 
            pass
        if tes > 0.5:
            print(sim26[a], syn26[b], tes)
        else: 
            pass


# In[149]:


trait28=model.wv.most_similar("unique", topn=33)
simtrait28=[]
for i in range(0,len(trait28)):   
    if wordnet.synsets(trait28[i][0], pos = wordnet.ADJ) != []:
        simtrait28.append(trait28[i][0])
    else:
        continue
sim28=list(simtrait28)
print(sim28) 


# In[150]:


synonyms28 = []
for syn in wordnet.synsets("unique", pos = wordnet.ADJ):
    for l in syn.lemmas():
        synonyms28.append(l.name())
syn28 = list(synonyms28)        
print(syn28)  


# In[151]:


for a in range(0,len(sim28)):
    for b in range(0,len(syn28)):
        try:
            tes=model.wv.similarity(w1=sim28[a],w2=syn28[b])
        except KeyError: 
            pass
        if tes > 0.5:
            print(sim28[a], syn28[b], tes)
        else: 
            pass


# In[154]:


trait211=model.wv.most_similar("contemporary", topn=31)
simtrait211=[]
for i in range(0,len(trait211)):   
    if wordnet.synsets(trait211[i][0], pos = wordnet.ADJ) != []:
        simtrait211.append(trait211[i][0])
    else:
        continue
sim211=list(simtrait211)
print(sim211) 


# In[155]:


synonyms211 = []
for syn in wordnet.synsets("contemporary", pos = wordnet.ADJ):
    for l in syn.lemmas():
        synonyms211.append(l.name())
syn211 = list(synonyms211)        
print(syn211)  


# In[156]:


for a in range(0,len(sim211)):
    for b in range(0,len(syn211)):
        try:
            tes=model.wv.similarity(w1=sim211[a],w2=syn211[b])
        except KeyError: 
            pass
        if tes > 0.5:
            print(sim211[a], syn211[b], tes)
        else: 
            pass


# # Dimensi Competence

# In[159]:


trait31=model.wv.most_similar("reliable", topn=27)
simtrait31=[]
for i in range(0,len(trait31)):   
    if wordnet.synsets(trait31[i][0], pos = wordnet.ADJ) != []:
        simtrait31.append(trait31[i][0])
    else:
        continue
sim31=list(simtrait31)
print(sim31) 


# In[160]:


synonyms31 = []
for syn in wordnet.synsets("reliable", pos = wordnet.ADJ):
    for l in syn.lemmas():
        synonyms31.append(l.name())
syn31 = list(synonyms31)        
print(syn31) 


# In[161]:


for a in range(0,len(sim31)):
    for b in range(0,len(syn31)):
        try:
            tes=model.wv.similarity(w1=sim31[a],w2=syn31[b])
        except KeyError: 
            pass
        if tes > 0.5:
            print(sim31[a], syn31[b], tes)
        else: 
            pass


# In[166]:


trait33=model.wv.most_similar("secure", topn=38)
simtrait33=[]
for i in range(0,len(trait33)):   
    if wordnet.synsets(trait33[i][0], pos = wordnet.ADJ) != []:
        simtrait33.append(trait33[i][0])
    else:
        continue
sim33=list(simtrait33)
print(sim33) 


# In[167]:


synonyms33 = []
for syn in wordnet.synsets("secure", pos = wordnet.ADJ):
    for l in syn.lemmas():
        synonyms33.append(l.name())
syn33 = list(synonyms33)        
print(syn33) 


# In[168]:


for a in range(0,len(sim33)):
    for b in range(0,len(syn33)):
        try:
            tes=model.wv.similarity(w1=sim33[a],w2=syn33[b])
        except KeyError: 
            pass
        if tes > 0.5:
            print(sim33[a], syn33[b], tes)
        else: 
            pass


# In[170]:


trait35=model.wv.most_similar("technical", topn=39)
simtrait35=[]
for i in range(0,len(trait35)):   
    if wordnet.synsets(trait35[i][0], pos = wordnet.ADJ) != []:
        simtrait35.append(trait35[i][0])
    else:
        continue
sim35=list(simtrait35)
print(sim35) 


# In[171]:


synonyms35 = []
for syn in wordnet.synsets("technical", pos = wordnet.ADJ):
    for l in syn.lemmas():
        synonyms35.append(l.name())
syn35 = list(synonyms35)        
print(syn35) 


# In[172]:


for a in range(0,len(sim35)):
    for b in range(0,len(syn35)):
        try:
            tes=model.wv.similarity(w1=sim35[a],w2=syn35[b])
        except KeyError: 
            pass
        if tes > 0.5:
            print(sim35[a], syn35[b], tes)
        else: 
            pass


# In[175]:


trait36=model.wv.most_similar("corporate", topn=37)
simtrait36=[]
for i in range(0,len(trait36)):   
    if wordnet.synsets(trait36[i][0], pos = wordnet.ADJ) != []:
        simtrait36.append(trait36[i][0])
    else:
        continue
sim36=list(simtrait36)
print(sim36) 


# In[176]:


synonyms36 = []
for syn in wordnet.synsets("corporate", pos = wordnet.ADJ):
    for l in syn.lemmas():
        synonyms36.append(l.name())
syn36 = list(synonyms36)        
print(syn36) 


# In[177]:


for a in range(0,len(sim36)):
    for b in range(0,len(syn36)):
        try:
            tes=model.wv.similarity(w1=sim36[a],w2=syn36[b])
        except KeyError: 
            pass
        if tes > 0.5:
            print(sim36[a], syn36[b], tes)
        else: 
            pass


# In[179]:


trait37=model.wv.most_similar("successful", topn=57)
simtrait37=[]
for i in range(0,len(trait37)):   
    if wordnet.synsets(trait37[i][0], pos = wordnet.ADJ) != []:
        simtrait37.append(trait37[i][0])
    else:
        continue
sim37=list(simtrait37)
print(sim37) 


# In[180]:


synonyms37 = []
for syn in wordnet.synsets("successful", pos = wordnet.ADJ):
    for l in syn.lemmas():
        synonyms37.append(l.name())
syn37 = list(synonyms37)        
print(syn37) 


# In[181]:


for a in range(0,len(sim37)):
    for b in range(0,len(syn37)):
        try:
            tes=model.wv.similarity(w1=sim37[a],w2=syn37[b])
        except KeyError: 
            pass
        if tes > 0.5:
            print(sim37[a], syn37[b], tes)
        else: 
            pass


# # Dimensi Sophistication

# In[183]:


trait42=model.wv.most_similar("glamorous", topn=56)
simtrait42=[]
for i in range(0,len(trait42)):   
    if wordnet.synsets(trait42[i][0], pos = wordnet.ADJ) != []:
        simtrait42.append(trait42[i][0])
    else:
        continue
sim42=list(simtrait42)
print(sim42) 


# In[184]:


synonyms42 = []
for syn in wordnet.synsets("glamorous", pos = wordnet.ADJ):
    for l in syn.lemmas():
        synonyms42.append(l.name())
syn42 = list(synonyms42)        
print(syn42) 


# In[185]:


for a in range(0,len(sim42)):
    for b in range(0,len(syn42)):
        try:
            tes=model.wv.similarity(w1=sim42[a],w2=syn42[b])
        except KeyError: 
            pass
        if tes > 0.5:
            print(sim42[a], syn42[b], tes)
        else: 
            pass


# In[187]:


trait44=model.wv.most_similar("charming", topn=20)
simtrait44=[]
for i in range(0,len(trait44)):   
    if wordnet.synsets(trait44[i][0], pos = wordnet.ADJ) != []:
        simtrait44.append(trait44[i][0])
    else:
        continue
sim44=list(simtrait44)
print(sim44) 


# In[188]:


synonyms44 = []
for syn in wordnet.synsets("charming", pos = wordnet.ADJ):
    for l in syn.lemmas():
        synonyms44.append(l.name())
syn44 = list(synonyms44)        
print(syn44) 


# In[189]:


for a in range(0,len(sim44)):
    for b in range(0,len(syn44)):
        try:
            tes=model.wv.similarity(w1=sim44[a],w2=syn44[b])
        except KeyError: 
            pass
        if tes > 0.5:
            print(sim44[a], syn44[b], tes)
        else: 
            pass


# In[192]:


trait46=model.wv.most_similar("smooth", topn=38)
simtrait46=[]
for i in range(0,len(trait46)):   
    if wordnet.synsets(trait46[i][0], pos = wordnet.ADJ) != []:
        simtrait46.append(trait46[i][0])
    else:
        continue
sim46=list(simtrait46)
print(sim46) 


# In[193]:


synonyms46 = []
for syn in wordnet.synsets("smooth", pos = wordnet.ADJ):
    for l in syn.lemmas():
        synonyms46.append(l.name())
syn46 = list(synonyms46)        
print(syn46) 


# In[194]:


for a in range(0,len(sim46)):
    for b in range(0,len(syn46)):
        try:
            tes=model.wv.similarity(w1=sim46[a],w2=syn46[b])
        except KeyError: 
            pass
        if tes > 0.5:
            print(sim46[a], syn46[b], tes)
        else: 
            pass


# # Dimensi Ruggedness

# In[196]:


trait54=model.wv.most_similar("tough", topn=35)
simtrait54=[]
for i in range(0,len(trait54)):   
    if wordnet.synsets(trait54[i][0], pos = wordnet.ADJ) != []:
        simtrait54.append(trait54[i][0])
    else:
        continue
sim54=list(simtrait54)
print(sim54) 


# In[197]:


synonyms54 = []
for syn in wordnet.synsets("tough", pos = wordnet.ADJ):
    for l in syn.lemmas():
        synonyms54.append(l.name())
syn54 = list(synonyms54)        
print(syn54) 


# In[198]:


for a in range(0,len(sim54)):
    for b in range(0,len(syn54)):
        try:
            tes=model.wv.similarity(w1=sim54[a],w2=syn54[b])
        except KeyError: 
            pass
        if tes > 0.5:
            print(sim54[a], syn54[b], tes)
        else: 
            pass


# # Dimnesi Sustainability

# In[203]:


trait61=model.wv.most_similar("lovely", topn=13)
simtrait61=[]
for i in range(0,len(trait61)):   
    if wordnet.synsets(trait61[i][0], pos = wordnet.ADJ) != []:
        simtrait61.append(trait61[i][0])
    else:
        continue
sim61=list(simtrait61)
print(sim61) 


# In[204]:


synonyms61 = []
for syn in wordnet.synsets("lovely", pos = wordnet.ADJ):
    for l in syn.lemmas():
        synonyms61.append(l.name())
syn61 = list(synonyms61)        
print(syn61) 


# In[205]:


for a in range(0,len(sim61)):
    for b in range(0,len(syn61)):
        try:
            tes=model.wv.similarity(w1=sim61[a],w2=syn61[b])
        except KeyError: 
            pass
        if tes > 0.5:
            print(sim61[a], syn61[b], tes)
        else: 
            pass


# In[218]:


trait64=model.wv.most_similar("giving", topn=46)
simtrait64=[]
for i in range(0,len(trait64)):   
    if wordnet.synsets(trait64[i][0], pos = wordnet.ADJ) != []:
        simtrait64.append(trait64[i][0])
    else:
        continue
sim64=list(simtrait64)
print(sim64) 


# In[219]:


synonyms64 = []
for syn in wordnet.synsets("giving", pos = wordnet.ADJ):
    for l in syn.lemmas():
        synonyms64.append(l.name())
syn64 = list(synonyms64)        
print(syn64) 


# In[220]:


for a in range(0,len(sim64)):
    for b in range(0,len(syn64)):
        try:
            tes=model.wv.similarity(w1=sim64[a],w2=syn64[b])
        except KeyError: 
            pass
        if tes > 0.5:
            print(sim64[a], syn64[b], tes)
        else: 
            pass


# In[222]:


trait65=model.wv.most_similar("fair", topn=45)
simtrait65=[]
for i in range(0,len(trait65)):   
    if wordnet.synsets(trait65[i][0], pos = wordnet.ADJ) != []:
        simtrait65.append(trait65[i][0])
    else:
        continue
sim65=list(simtrait65)
print(sim65) 


# In[223]:


synonyms65 = []
for syn in wordnet.synsets("fair", pos = wordnet.ADJ):
    for l in syn.lemmas():
        synonyms65.append(l.name())
syn65 = list(synonyms65)        
print(syn65) 


# In[224]:


for a in range(0,len(sim65)):
    for b in range(0,len(syn65)):
        try:
            tes=model.wv.similarity(w1=sim65[a],w2=syn65[b])
        except KeyError: 
            pass
        if tes > 0.5:
            print(sim65[a], syn65[b], tes)
        else: 
            pass


# In[233]:


trait67=model.wv.most_similar("aware", topn=39)
simtrait67=[]
for i in range(0,len(trait67)):   
    if wordnet.synsets(trait67[i][0], pos = wordnet.ADJ) != []:
        simtrait67.append(trait67[i][0])
    else:
        continue
sim67=list(simtrait67)
print(sim67) 


# In[236]:


synonyms67 = []
for syn in wordnet.synsets("aware", pos = wordnet.ADJ):
    for l in syn.lemmas():
        synonyms67.append(l.name())
syn67 = list(synonyms67)        
print(syn67) 


# In[237]:


for a in range(0,len(sim67)):
    for b in range(0,len(syn67)):
        try:
            tes=model.wv.similarity(w1=sim67[a],w2=syn67[b])
        except KeyError: 
            pass
        if tes > 0.5:
            print(sim67[a], syn67[b], tes)
        else: 
            pass


# In[243]:


trait610=model.wv.most_similar("peaceful", topn=19)
simtrait610=[]
for i in range(0,len(trait610)):   
    if wordnet.synsets(trait610[i][0], pos = wordnet.ADJ) != []:
        simtrait610.append(trait610[i][0])
    else:
        continue
sim610=list(simtrait610)
print(sim610) 


# In[244]:


synonyms610 = []
for syn in wordnet.synsets("peaceful", pos = wordnet.ADJ):
    for l in syn.lemmas():
        synonyms610.append(l.name())
syn610 = list(synonyms610)        
print(syn610) 


# In[245]:


for a in range(0,len(sim610)):
    for b in range(0,len(syn610)):
        try:
            tes=model.wv.similarity(w1=sim610[a],w2=syn610[b])
        except KeyError: 
            pass
        if tes > 0.5:
            print(sim610[a], syn610[b], tes)
        else: 
            pass


# In[246]:


df['content'] = df['content'].astype("str").astype("string")


# In[247]:


df


# # Dimensi Sincerity

# In[20]:


sum1= 0
itung=df['content'].str.count("involved ")
for itung1 in range(0,len(df)):
    sum1=sum1+itung[itung1]
print("involved = ", sum1 )    


# In[21]:


sum2 = 0
itung=df['content'].str.count("alone ")
for itung1 in range(0,len(df)):
    sum2=sum2+itung[itung1]
print("alone = ", sum2 )    


# In[22]:


sum3 = 0
itung=df['content'].str.count("after ")
for itung1 in range(0,len(df)):
    sum3=sum3+itung[itung1]
print("after = ", sum3 )    


# In[23]:


sum4 = 0
itung=df['content'].str.count("stated ")
for itung1 in range(0,len(df)):
    sum4=sum4+itung[itung1]
print("stated = ", sum4 )    


# In[24]:


sum5 = 0
itung=df['content'].str.count("embarrassing ")
for itung1 in range(0,len(df)):
    sum5=sum5+itung[itung1]
print("embarassing = ", sum5 )    


# In[25]:


sum6 = 0
itung=df['content'].str.count("lost ")
for itung1 in range(0,len(df)):
    sum6=sum6+itung[itung1]
print("lost = ", sum6 )    


# In[26]:


sum7 = 0
itung=df['content'].str.count("together ")
for itung1 in range(0,len(df)):
    sum7=sum7+itung[itung1]
print("together = ", sum7 )    


# In[27]:


sum8 = 0
itung=df['content'].str.count("angry ")
for itung1 in range(0,len(df)):
    sum8=sum8+itung[itung1]
print("angry = ", sum8 )    


# In[28]:


sum9 = 0
itung=df['content'].str.count("through ")
for itung1 in range(0,len(df)):
    sum9=sum9+itung[itung1]
print("through = ", sum9 )    


# In[29]:


sum10 = 0
itung=df['content'].str.count("continued ")
for itung1 in range(0,len(df)):
    sum10=sum10+itung[itung1]
print("continued = ", sum10 )    


# In[30]:


sum11 = 0
itung=df['content'].str.count("honest ")
for itung1 in range(0,len(df)):
    sum11=sum11+itung[itung1]
print("honest = ", sum11 )    


# In[31]:


sum12 = 0
itung=df['content'].str.count("reliable ")
for itung1 in range(0,len(df)):
    sum12=sum12+itung[itung1]
print("reliable = ", sum12 )    


# In[32]:


sum13 = 0
itung=df['content'].str.count("true ")
for itung1 in range(0,len(df)):
    sum13=sum13+itung[itung1]
print("true = ", sum13 )    


# In[33]:


sum14 = 0
itung=df['content'].str.count("good ")
for itung1 in range(0,len(df)):
    sum14=sum14+itung[itung1]
print("good = ", sum14 )    


# In[34]:


sum15 = 0
itung=df['content'].str.count("fair ")
for itung1 in range(0,len(df)):
    sum15=sum15+itung[itung1]
print("fair = ", sum15 )    


# #Trait Honest

# In[35]:


traithonest = sum1+sum2+sum3+sum4+sum5+sum6+sum7+sum8+sum9+sum10+sum11+sum12+sum13+sum14+sum15
print(traithonest)


# In[36]:


sum16 = 0
itung=df['content'].str.count("female ")
for itung1 in range(0,len(df)):
    sum16=sum16+itung[itung1]
print("female = ", sum16 )    


# In[37]:


sum17 = 0
itung=df['content'].str.count("addressed ")
for itung1 in range(0,len(df)):
    sum17=sum17+itung[itung1]
print("addressed = ", sum17 )    


# In[38]:


sum18 = 0
itung=df['content'].str.count("assisted ")
for itung1 in range(0,len(df)):
    sum18=sum18+itung[itung1]
print("assisted = ", sum18 )    


# In[39]:


sum19 = 0
itung=df['content'].str.count("required ")
for itung1 in range(0,len(df)):
    sum19=sum19+itung[itung1]
print("required = ", sum19 )    


# In[40]:


sum20 = 0
itung=df['content'].str.count("telling ")
for itung1 in range(0,len(df)):
    sum20=sum20+itung[itung1]
print("telling = ", sum20 )    


# In[41]:


sum21 = 0
itung=df['content'].str.count("greek ")
for itung1 in range(0,len(df)):
    sum21=sum21+itung[itung1]
print("greek = ", sum21 )    


# In[42]:


sum22 = 0
itung=df['content'].str.count("scheduled ")
for itung1 in range(0,len(df)):
    sum22=sum22+itung[itung1]
print("scheduled = ", sum22 )   


# In[43]:


sum23 = 0
itung=df['content'].str.count("spoken ")
for itung1 in range(0,len(df)):
    sum23=sum23+itung[itung1]
print("spoken = ", sum23 )   


# In[44]:


sum24 = 0
itung=df['content'].str.count("sorry ")
for itung1 in range(0,len(df)):
    sum24=sum24+itung[itung1]
print("sorry = ", sum24 )   


# In[45]:


sum25 = 0
itung=df['content'].str.count("individual ")
for itung1 in range(0,len(df)):
    sum25=sum25+itung[itung1]
print("individual = ", sum25 )   


# In[46]:


sum26 = 0
itung=df['content'].str.count("sincere ")
for itung1 in range(0,len(df)):
    sum26=sum26+itung[itung1]
print("sincere = ", sum26 )   


# In[47]:


sum27 = 0
itung=df['content'].str.count("solemn ")
for itung1 in range(0,len(df)):
    sum27=sum27+itung[itung1]
print("solemn = ", sum27 )   


# #Trait sincere

# In[48]:


traitsincere = sum16+sum17+sum18+sum19+sum20+sum21+sum22+sum23+sum24+sum25+sum26+sum27
print(traitsincere)


# In[49]:


sum28 = 0
itung=df['content'].str.count("solid ")
for itung1 in range(0,len(df)):
    sum28=sum28+itung[itung1]
print("solid = ", sum28 )   


# In[50]:


sum29 = 0
itung=df['content'].str.count("unique ")
for itung1 in range(0,len(df)):
    sum29=sum29+itung[itung1]
print("unique = ", sum29 )   


# In[51]:


sum30 = 0
itung=df['content'].str.count("advertised ")
for itung1 in range(0,len(df)):
    sum30=sum30+itung[itung1]
print("advertised = ", sum30 )   


# In[52]:


sum30 = 0
itung=df['content'].str.count("advertised ")
for itung1 in range(0,len(df)):
    sum30=sum30+itung[itung1]
print("advertised = ", sum30 )   


# In[53]:


sum31 = 0
itung=df['content'].str.count("palatial ")
for itung1 in range(0,len(df)):
    sum31=sum31+itung[itung1]
print("palatial = ", sum31 )   


# In[54]:


sum32 = 0
itung=df['content'].str.count("positive ")
for itung1 in range(0,len(df)):
    sum32=sum32+itung[itung1]
print("positive = ", sum32 )   


# In[55]:


sum33 = 0
itung=df['content'].str.count("tremendous ")
for itung1 in range(0,len(df)):
    sum33=sum33+itung[itung1]
print("tremendous = ", sum33 )   


# In[56]:


sum34 = 0
itung=df['content'].str.count("highest ")
for itung1 in range(0,len(df)):
    sum34=sum34+itung[itung1]
print("highest = ", sum34 )   


# In[57]:


sum35 = 0
itung=df['content'].str.count("extraordinary ")
for itung1 in range(0,len(df)):
    sum35=sum35+itung[itung1]
print("extraordinary = ", sum35 )   


# In[58]:


sum36 = 0
itung=df['content'].str.count("geared ")
for itung1 in range(0,len(df)):
    sum36=sum36+itung[itung1]
print("geared = ", sum36 )   


# In[59]:


sum37 = 0
itung=df['content'].str.count("spoiled ")
for itung1 in range(0,len(df)):
    sum37=sum37+itung[itung1]
print("spoiled = ", sum37 )   


# In[60]:


sum38 = 0
itung=df['content'].str.count("real ")
for itung1 in range(0,len(df)):
    sum38=sum38+itung[itung1]
print("real = ", sum38 )   


# In[61]:


sum39 = 0
itung=df['content'].str.count("existent ")
for itung1 in range(0,len(df)):
    sum39=sum39+itung[itung1]
print("existent = ", sum39 )   


# In[62]:


sum40 = 0
itung=df['content'].str.count("actual ")
for itung1 in range(0,len(df)):
    sum40=sum40+itung[itung1]
print("actual = ", sum40 )   


# In[63]:


sum41 = 0
itung=df['content'].str.count("genuine ")
for itung1 in range(0,len(df)):
    sum41=sum41+itung[itung1]
print("genuine = ", sum41 )   


# In[64]:


sum42 = 0
itung=df['content'].str.count("substantial ")
for itung1 in range(0,len(df)):
    sum42=sum42+itung[itung1]
print("substantial = ", sum42 ) 


# In[65]:


sum43 = 0
itung=df['content'].str.count("material ")
for itung1 in range(0,len(df)):
    sum43=sum43+itung[itung1]
print("material = ", sum43 ) 


# #trait real

# In[66]:


traitreal=sum28+sum29+sum30+sum31+sum32+sum33+sum34+sum35+sum36+sum37+sum38+sum39+sum40+sum41+sum42+sum43
print(traitreal)


# In[67]:


sum44 = 0
itung=df['content'].str.count("golden ")
for itung1 in range(0,len(df)):
    sum44=sum44+itung[itung1]
print("golden = ", sum44 ) 


# In[68]:


sum45 = 0
itung=df['content'].str.count("built ")
for itung1 in range(0,len(df)):
    sum45=sum45+itung[itung1]
print("built = ", sum45 ) 


# In[69]:


sum46 = 0
itung=df['content'].str.count("unbelievable ")
for itung1 in range(0,len(df)):
    sum46=sum46+itung[itung1]
print("unbelievable = ", sum46 ) 


# In[70]:


sum47 = 0
itung=df['content'].str.count("private ")
for itung1 in range(0,len(df)):
    sum47=sum47+itung[itung1]
print("private = ", sum47 ) 


# In[71]:


sum48 = 0
itung=df['content'].str.count("weather ")
for itung1 in range(0,len(df)):
    sum48=sum48+itung[itung1]
print("weather = ", sum48 ) 


# In[72]:


sum49 = 0
itung=df['content'].str.count("exclusive ")
for itung1 in range(0,len(df)):
    sum49=sum49+itung[itung1]
print("exclusive = ", sum49 ) 


# In[73]:


sum50 = 0
itung=df['content'].str.count("closing ")
for itung1 in range(0,len(df)):
    sum50=sum50+itung[itung1]
print("closing = ", sum50 ) 


# In[74]:


sum51 = 0
itung=df['content'].str.count("blind ")
for itung1 in range(0,len(df)):
    sum51=sum51+itung[itung1]
print("blind = ", sum50 ) 


# In[75]:


sum52 = 0
itung=df['content'].str.count("asleep ")
for itung1 in range(0,len(df)):
    sum52=sum52+itung[itung1]
print("asleep = ", sum52 ) 


# In[76]:


sum53 = 0
itung=df['content'].str.count("older ")
for itung1 in range(0,len(df)):
    sum53=sum53+itung[itung1]
print("older = ", sum53 ) 


# In[77]:


sum54 = 0
itung=df['content'].str.count("original ")
for itung1 in range(0,len(df)):
    sum54=sum54+itung[itung1]
print("original = ", sum54 ) 


# In[78]:


#Trait original


# In[79]:


traitoriginal = sum44+sum45+sum46+sum47+sum48+sum49+sum50+sum51+sum52+sum53+sum54
print(traitoriginal)


# In[80]:


sum55 = 0
itung=df['content'].str.count("smiling ")
for itung1 in range(0,len(df)):
    sum55=sum55+itung[itung1]
print("smiling = ", sum55 ) 


# In[81]:


sum56 = 0
itung=df['content'].str.count("hospitable ")
for itung1 in range(0,len(df)):
    sum56=sum56+itung[itung1]
print("hospitable = ", sum56 ) 


# In[82]:


sum57 = 0
itung=df['content'].str.count("responsive ")
for itung1 in range(0,len(df)):
    sum57=sum57+itung[itung1]
print("responsive = ", sum57 ) 


# In[83]:


sum58 = 0
itung=df['content'].str.count("thorough ")
for itung1 in range(0,len(df)):
    sum58=sum58+itung[itung1]
print("thorough = ", sum58 ) 


# In[84]:


sum59 = 0
itung=df['content'].str.count("patient ")
for itung1 in range(0,len(df)):
    sum59=sum59+itung[itung1]
print("patient = ", sum59 ) 


# In[85]:


sum60 = 0
itung=df['content'].str.count("matt ")
for itung1 in range(0,len(df)):
    sum60=sum60+itung[itung1]
print("matt = ", sum60 ) 


# In[86]:


sum61 = 0
itung=df['content'].str.count("caring ")
for itung1 in range(0,len(df)):
    sum61=sum61+itung[itung1]
print("caring = ", sum61 )


# In[87]:


sum62 = 0
itung=df['content'].str.count("personable ")
for itung1 in range(0,len(df)):
    sum62=sum62+itung[itung1]
print("personable = ", sum62 )


# In[88]:


sum63 = 0
itung=df['content'].str.count("informative ")
for itung1 in range(0,len(df)):
    sum63=sum63+itung[itung1]
print("informative = ", sum63 )


# In[89]:


sum64 = 0
itung=df['content'].str.count("thoughtful ")
for itung1 in range(0,len(df)):
    sum64=sum64+itung[itung1]
print("thoughtful = ", sum64 )


# In[90]:


sum65 = 0
itung=df['content'].str.count("cheerful ")
for itung1 in range(0,len(df)):
    sum65=sum65+itung[itung1]
print("cheerful = ", sum65 )


# In[91]:


sum66 = 0
itung=df['content'].str.count("upbeat ")
for itung1 in range(0,len(df)):
    sum66=sum66+itung[itung1]
print("upbeat = ", sum66 )


# In[92]:


#trait cheerful


# In[93]:


traitcheerful = sum55+sum56+sum57+sum58+sum59+sum60+sum61+sum62+sum63+sum64+sum65+sum66
print(traitcheerful)


# In[94]:


sum67 = 0
itung=df['content'].str.count("accommodating ")
for itung1 in range(0,len(df)):
    sum67=sum67+itung[itung1]
print("accommodating = ", sum67 )


# In[95]:


sum68 = 0
itung=df['content'].str.count("courteous ")
for itung1 in range(0,len(df)):
    sum68=sum68+itung[itung1]
print("courteous = ", sum68 )


# In[96]:


sum69 = 0
itung=df['content'].str.count("attentive ")
for itung1 in range(0,len(df)):
    sum69=sum69+itung[itung1]
print("attentive = ", sum69 )


# In[97]:


sum70 = 0
itung=df['content'].str.count("helpful ")
for itung1 in range(0,len(df)):
    sum70=sum70+itung[itung1]
print("helpful = ", sum70 )


# In[98]:


sum71 = 0
itung=df['content'].str.count("professional ")
for itung1 in range(0,len(df)):
    sum71=sum71+itung[itung1]
print("professional = ", sum71 )


# In[99]:


sum72 = 0
itung=df['content'].str.count("welcoming ")
for itung1 in range(0,len(df)):
    sum72=sum72+itung[itung1]
print("welcoming = ", sum72 )


# In[100]:


sum73 = 0
itung=df['content'].str.count("pleasant ")
for itung1 in range(0,len(df)):
    sum73=sum73+itung[itung1]
print("pleasant = ", sum73 )


# In[101]:


sum74 = 0
itung=df['content'].str.count("polite ")
for itung1 in range(0,len(df)):
    sum74=sum74+itung[itung1]
print("polite = ", sum74 )


# In[102]:


sum75 = 0
itung=df['content'].str.count("efficient ")
for itung1 in range(0,len(df)):
    sum75=sum75+itung[itung1]
print("efficient = ", sum75 )


# In[103]:


sum76 = 0
itung=df['content'].str.count("knowledgable ")
for itung1 in range(0,len(df)):
    sum76=sum76+itung[itung1]
print("knowledgable = ", sum76 )


# In[104]:


sum77 = 0
itung=df['content'].str.count("friendly ")
for itung1 in range(0,len(df)):
    sum77=sum77+itung[itung1]
print("friendly = ", sum77 )


# In[105]:


sum78 = 0
itung=df['content'].str.count("favorable ")
for itung1 in range(0,len(df)):
    sum78=sum78+itung[itung1]
print("favorable = ", sum78 )


# In[106]:


sum79 = 0
itung=df['content'].str.count("well-disposed ")
for itung1 in range(0,len(df)):
    sum79=sum79+itung[itung1]
print("well disposed = ", sum79 )


# In[107]:


#trait friendly


# In[108]:


traitfriendly = sum67+sum68+sum69+sum70+sum71+sum72+sum73+sum74+sum75+sum76+sum77+sum78+sum79
print(traitfriendly)


# # Dimensi Excitement

# In[109]:


sume1 = 0
itung=df['content'].str.count("most ")
for itung1 in range(0,len(df)):
    sume1=sume1+itung[itung1]
print("most = ", sume1 )


# In[110]:


sume2 = 0
itung=df['content'].str.count("prestigious ")
for itung1 in range(0,len(df)):
    sume2=sume2+itung[itung1]
print("prestigious = ", sume2 )


# In[111]:


sume3 = 0
itung=df['content'].str.count("classical ")
for itung1 in range(0,len(df)):
    sume3=sume3+itung[itung1]
print("classical = ", sume3 )


# In[112]:


sume4 = 0
itung=df['content'].str.count("former ")
for itung1 in range(0,len(df)):
    sume4=sume4+itung[itung1]
print("former = ", sume4 )


# In[113]:


sume5 = 0
itung=df['content'].str.count("chic ")
for itung1 in range(0,len(df)):
    sume5=sume5+itung[itung1]
print("chic = ", sume5 )


# In[114]:


sume6 = 0
itung=df['content'].str.count("firm ")
for itung1 in range(0,len(df)):
    sume6=sume6+itung[itung1]
print("firm = ", sume6 )


# In[115]:


sume7 = 0
itung=df['content'].str.count("global ")
for itung1 in range(0,len(df)):
    sume7=sume7+itung[itung1]
print("global = ", sume7 )


# In[116]:


sume8 = 0
itung=df['content'].str.count("preferred ")
for itung1 in range(0,len(df)):
    sume8=sume8+itung[itung1]
print("preferred = ", sume8 )


# In[117]:


sume9 = 0
itung=df['content'].str.count("picky ")
for itung1 in range(0,len(df)):
    sume9=sume9+itung[itung1]
print("picky = ", sume9 )


# In[118]:


sume10 = 0
itung=df['content'].str.count("tops ")
for itung1 in range(0,len(df)):
    sume10=sume10+itung[itung1]
print("tops = ", sume10 )


# In[119]:


sume11 = 0
itung=df['content'].str.count("trendy ")
for itung1 in range(0,len(df)):
    sume11=sume11+itung[itung1]
print("trendy = ", sume11 )


# In[120]:


#trait trendy


# In[121]:


traittrendy = sume1+sume2+sume3+sume4+sume5+sume6+sume7+sume8+sume9+sume10+sume11
print(traittrendy)


# In[122]:


sume12 = 0
itung=df['content'].str.count("rich ")
for itung1 in range(0,len(df)):
    sume12=sume12+itung[itung1]
print("rich = ", sume12 )


# In[123]:


sume13 = 0
itung=df['content'].str.count("seasonal ")
for itung1 in range(0,len(df)):
    sume13=sume13+itung[itung1]
print("seasonal = ", sume13 )


# In[124]:


sume14 = 0
itung=df['content'].str.count("marvelous ")
for itung1 in range(0,len(df)):
    sume14=sume14+itung[itung1]
print("marverlous = ", sume14 )


# In[125]:


sume15 = 0
itung=df['content'].str.count("oriental ")
for itung1 in range(0,len(df)):
    sume15=sume15+itung[itung1]
print("oriental = ", sume15 )


# In[126]:


sume16 = 0
itung=df['content'].str.count("significant ")
for itung1 in range(0,len(df)):
    sume16=sume16+itung[itung1]
print("significant = ", sume16 )


# In[127]:


sume17 = 0
itung=df['content'].str.count("sad ")
for itung1 in range(0,len(df)):
    sume17=sume17+itung[itung1]
print("sad = ", sume17 )


# In[128]:


sume18 = 0
itung=df['content'].str.count("pet ")
for itung1 in range(0,len(df)):
    sume18=sume18+itung[itung1]
print("pet = ", sume18 )


# In[129]:


sume19 = 0
itung=df['content'].str.count("pro ")
for itung1 in range(0,len(df)):
    sume19=sume19+itung[itung1]
print("pro = ", sume19 )


# In[130]:


sume20 = 0
itung=df['content'].str.count("authentic ")
for itung1 in range(0,len(df)):
    sume20=sume20+itung[itung1]
print("authentic = ", sume20 )


# In[131]:


sume21 = 0
itung=df['content'].str.count("picky ")
for itung1 in range(0,len(df)):
    sume21=sume21+itung[itung1]
print("picky = ", sume21 )


# In[132]:


sume22 = 0
itung=df['content'].str.count("exciting ")
for itung1 in range(0,len(df)):
    sume22=sume22+itung[itung1]
print("exciting = ", sume22 )


# In[133]:


#trait exciting


# In[134]:


traitexciting=sume12+sume13+sume14+sume15+sume16+sume17+sume18+sume19+sume20+sume21+sume22
print(traitexciting)


# In[135]:


sume23 = 0
itung=df['content'].str.count("liked ")
for itung1 in range(0,len(df)):
    sume23=sume23+itung[itung1]
print("liked = ", sume23 )


# In[136]:


sume24 = 0
itung=df['content'].str.count("fancy ")
for itung1 in range(0,len(df)):
    sume24=sume24+itung[itung1]
print("fancy = ", sume24 )


# In[137]:


sume25 = 0
itung=df['content'].str.count("dark ")
for itung1 in range(0,len(df)):
    sume25=sume25+itung[itung1]
print("dark = ", sume25 )


# In[138]:


sume26 = 0
itung=df['content'].str.count("functional ")
for itung1 in range(0,len(df)):
    sume26=sume26+itung[itung1]
print("functional = ", sume26 )


# In[139]:


sume27 = 0
itung=df['content'].str.count("dim ")
for itung1 in range(0,len(df)):
    sume27=sume27+itung[itung1]
print("dim = ", sume27 )


# In[140]:


sume28 = 0
itung=df['content'].str.count("affordable ")
for itung1 in range(0,len(df)):
    sume28=sume28+itung[itung1]
print("affordable = ", sume28 )


# In[141]:


sume29 = 0
itung=df['content'].str.count("outdated ")
for itung1 in range(0,len(df)):
    sume29=sume29+itung[itung1]
print("outdated = ", sume29 )


# In[142]:


sume30 = 0
itung=df['content'].str.count("congested ")
for itung1 in range(0,len(df)):
    sume30=sume30+itung[itung1]
print("congested = ", sume30 )


# In[143]:


sume31 = 0
itung=df['content'].str.count("crowded ")
for itung1 in range(0,len(df)):
    sume31=sume31+itung[itung1]
print("crowded = ", sume31 )


# In[144]:


sume32 = 0
itung=df['content'].str.count("plentiful ")
for itung1 in range(0,len(df)):
    sume32=sume32+itung[itung1]
print("plentiful = ", sume32 )


# In[145]:


sume33 = 0
itung=df['content'].str.count("cool ")
for itung1 in range(0,len(df)):
    sume33=sume33+itung[itung1]
print("cool = ", sume33 )


# In[146]:


#trait cool


# In[147]:


traitcool=sume23+sume24+sume25+sume26+sume27+sume28+sume29+sume30+sume31+sume32+sume33
print(traitcool)


# In[148]:


sume34 = 0
itung=df['content'].str.count("realized ")
for itung1 in range(0,len(df)):
    sume34=sume34+itung[itung1]
print("realized = ", sume34 )


# In[149]:


sume35 = 0
itung=df['content'].str.count("gifted ")
for itung1 in range(0,len(df)):
    sume35=sume35+itung[itung1]
print("gifted = ", sume35 )


# In[150]:


sume36 = 0
itung=df['content'].str.count("adult ")
for itung1 in range(0,len(df)):
    sume36=sume36+itung[itung1]
print("adult = ", sume36 )


# In[151]:


sume37 = 0
itung=df['content'].str.count("medical ")
for itung1 in range(0,len(df)):
    sume37=sume37+itung[itung1]
print("medical = ", sume37 )


# In[152]:


sume38 = 0
itung=df['content'].str.count("together ")
for itung1 in range(0,len(df)):
    sume38=sume38+itung[itung1]
print("together = ", sume38 )


# In[153]:


sume39 = 0
itung=df['content'].str.count("stuffed ")
for itung1 in range(0,len(df)):
    sume39=sume39+itung[itung1]
print("stuffed = ", sume39 )


# In[154]:


sume40 = 0
itung=df['content'].str.count("confirmed ")
for itung1 in range(0,len(df)):
    sume40=sume40+itung[itung1]
print("confirmed = ", sume40 )


# In[155]:


sume41 = 0
itung=df['content'].str.count("played ")
for itung1 in range(0,len(df)):
    sume41=sume41+itung[itung1]
print("played = ", sume41 )


# In[156]:


sume42 = 0
itung=df['content'].str.count("noticed ")
for itung1 in range(0,len(df)):
    sume42=sume42+itung[itung1]
print("noticed = ", sume42 )


# In[157]:


sume43 = 0
itung=df['content'].str.count("engaged ")
for itung1 in range(0,len(df)):
    sume43=sume43+itung[itung1]
print("engaged = ", sume43 )


# In[158]:


sume44 = 0
itung=df['content'].str.count("young ")
for itung1 in range(0,len(df)):
    sume44=sume40+itung[itung1]
print("young = ", sume44 )


# In[159]:


sume45 = 0
itung=df['content'].str.count("new ")
for itung1 in range(0,len(df)):
    sume45=sume45+itung[itung1]
print("new = ", sume45 )


# In[160]:


#trait young


# In[161]:


traityoung=sume34+sume35+sume36+sume37+sume38+sume39+sume40+sume41+sume42+sume43+sume44+sume45
print(traityoung)


# In[162]:


sume46 = 0
itung=df['content'].str.count("satisfied ")
for itung1 in range(0,len(df)):
    sume46=sume46+itung[itung1]
print("satisfied = ", sume46 )


# In[163]:


sume47 = 0
itung=df['content'].str.count("highest ")
for itung1 in range(0,len(df)):
    sume47=sume47+itung[itung1]
print("highest = ", sume47 )


# In[164]:


sume48 = 0
itung=df['content'].str.count("tremendous ")
for itung1 in range(0,len(df)):
    sume48=sume48+itung[itung1]
print("tremendous = ", sume48 )


# In[165]:


sume49 = 0
itung=df['content'].str.count("solid ")
for itung1 in range(0,len(df)):
    sume49=sume49+itung[itung1]
print("solid = ", sume49 )


# In[166]:


sume50 = 0
itung=df['content'].str.count("extraordinary ")
for itung1 in range(0,len(df)):
    sume50=sume50+itung[itung1]
print("extraordinary = ", sume50 )


# In[167]:


sume51 = 0
itung=df['content'].str.count("positive ")
for itung1 in range(0,len(df)):
    sume51=sume51+itung[itung1]
print("positive = ", sume51 )


# In[168]:


sume52 = 0
itung=df['content'].str.count("delightful ")
for itung1 in range(0,len(df)):
    sume52=sume52+itung[itung1]
print("delightful = ", sume52 )


# In[169]:


sume53 = 0
itung=df['content'].str.count("absolute ")
for itung1 in range(0,len(df)):
    sume53=sume53+itung[itung1]
print("absolute = ", sume53 )


# In[170]:


sume54 = 0
itung=df['content'].str.count("familiar ")
for itung1 in range(0,len(df)):
    sume54=sume54+itung[itung1]
print("familiar = ", sume54 )


# In[171]:


sume55 = 0
itung=df['content'].str.count("amazed ")
for itung1 in range(0,len(df)):
    sume55=sume55+itung[itung1]
print("amazed = ", sume55 )


# In[172]:


sume56 = 0
itung=df['content'].str.count("alone ")
for itung1 in range(0,len(df)):
    sume56=sume56+itung[itung1]
print("alone = ", sume56 )


# In[173]:


sume57 = 0
itung=df['content'].str.count("unique ")
for itung1 in range(0,len(df)):
    sume57=sume57+itung[itung1]
print("unique = ", sume57 )


# In[174]:


sume58 = 0
itung=df['content'].str.count("unparalleled ")
for itung1 in range(0,len(df)):
    sume58=sume58+itung[itung1]
print("unparalleled = ", sume58 )


# In[175]:


#trait unique


# In[176]:


traitunique=sume46+sume47+sume48+sume49+sume50+sume51+sume52+sume53+sume54+sume55+sume56+sume57+sume58
print(traitunique)


# In[177]:


sume59 = 0
itung=df['content'].str.count("thick ")
for itung1 in range(0,len(df)):
    sume59=sume59+itung[itung1]
print("thick = ", sume59 )


# In[178]:


sume60 = 0
itung=df['content'].str.count("plush ")
for itung1 in range(0,len(df)):
    sume60=sume60+itung[itung1]
print("plush = ", sume60 )


# In[179]:


sume61 = 0
itung=df['content'].str.count("designed ")
for itung1 in range(0,len(df)):
    sume61=sume61+itung[itung1]
print("designed = ", sume61 )


# In[180]:


sume62 = 0
itung=df['content'].str.count("living ")
for itung1 in range(0,len(df)):
    sume62=sume62+itung[itung1]
print("living = ", sume62 )


# In[181]:


sume63 = 0
itung=df['content'].str.count("attractive ")
for itung1 in range(0,len(df)):
    sume63=sume63+itung[itung1]
print("attractive = ", sume63 )


# In[182]:


sume64 = 0
itung=df['content'].str.count("effective ")
for itung1 in range(0,len(df)):
    sume64=sume64+itung[itung1]
print("effective = ", sume64 )


# In[183]:


sume65 = 0
itung=df['content'].str.count("premium ")
for itung1 in range(0,len(df)):
    sume65=sume65+itung[itung1]
print("premium = ", sume65 )


# In[184]:


sume66 = 0
itung=df['content'].str.count("exterior ")
for itung1 in range(0,len(df)):
    sume66=sume66+itung[itung1]
print("exterior = ", sume66 )


# In[185]:


sume67 = 0
itung=df['content'].str.count("extensive ")
for itung1 in range(0,len(df)):
    sume67=sume67+itung[itung1]
print("extensive = ", sume67 )


# In[186]:


sume68 = 0
itung=df['content'].str.count("lush ")
for itung1 in range(0,len(df)):
    sume68=sume68+itung[itung1]
print("lush = ", sume68 )


# In[187]:


sume69 = 0
itung=df['content'].str.count("contemporary ")
for itung1 in range(0,len(df)):
    sume69=sume69+itung[itung1]
print("contemporary = ", sume69 )


# In[188]:


#trait contemporary


# In[189]:


traitcontemporary=sume59+sume60+sume61+sume62+sume63+sume64+sume65+sume66+sume67+sume68+sume69
print(traitcontemporary)


# # Dimensi Competence

# In[190]:


sumc1 = 0
itung=df['content'].str.count("self ")
for itung1 in range(0,len(df)):
    sumc1=sumc1+itung[itung1]
print("self = ", sumc1 )


# In[191]:


sumc2 = 0
itung=df['content'].str.count("horrible ")
for itung1 in range(0,len(df)):
    sumc2=sumc2+itung[itung1]
print("horrible = ", sumc2 )


# In[192]:


sumc3 = 0
itung=df['content'].str.count("serious ")
for itung1 in range(0,len(df)):
    sumc3=sumc3+itung[itung1]
print("serious = ", sumc3 )


# In[193]:


sumc4 = 0
itung=df['content'].str.count("strong ")
for itung1 in range(0,len(df)):
    sumc4=sumc4+itung[itung1]
print("strong = ", sumc4 )


# In[194]:


sumc5 = 0
itung=df['content'].str.count("dingy ")
for itung1 in range(0,len(df)):
    sumc5=sumc5+itung[itung1]
print("dingy = ", sumc5 )


# In[195]:


sumc6 = 0
itung=df['content'].str.count("weird ")
for itung1 in range(0,len(df)):
    sumc6=sumc6+itung[itung1]
print("weird = ", sumc6 )


# In[196]:


sumc7 = 0
itung=df['content'].str.count("terrible ")
for itung1 in range(0,len(df)):
    sumc7=sumc7+itung[itung1]
print("terrible = ", sumc7 )


# In[197]:


sumc8 = 0
itung=df['content'].str.count("bare ")
for itung1 in range(0,len(df)):
    sumc8=sumc8+itung[itung1]
print("bare = ", sumc8 )


# In[198]:


sumc9 = 0
itung=df['content'].str.count("uncomfortable ")
for itung1 in range(0,len(df)):
    sumc9=sumc9+itung[itung1]
print("uncomfortable = ", sumc9 )


# In[199]:


sumc10 = 0
itung=df['content'].str.count("over ")
for itung1 in range(0,len(df)):
    sumc10=sumc10+itung[itung1]
print("uncomfortable = ", sumc10 )


# In[200]:


sumc11 = 0
itung=df['content'].str.count("reliable ")
for itung1 in range(0,len(df)):
    sumc11=sumc11+itung[itung1]
print("reliable = ", sumc11 )


# In[201]:


sumc12 = 0
itung=df['content'].str.count("honest ")
for itung1 in range(0,len(df)):
    sumc12=sumc12+itung[itung1]
print("honest = ", sumc12 )


# In[202]:


sumc13 = 0
itung=df['content'].str.count("true ")
for itung1 in range(0,len(df)):
    sumc13=sumc13+itung[itung1]
print("true = ", sumc13 )


# In[203]:


sumc14 = 0
itung=df['content'].str.count("authentic ")
for itung1 in range(0,len(df)):
    sumc14=sumc14+itung[itung1]
print("authentic = ", sumc14 )


# In[204]:


#Trait reliable


# In[205]:


traitreliable=sumc1+sumc2+sumc3+sumc4+sumc5+sumc6+sumc7+sumc8+sumc9+sumc10+sumc11+sumc12+sumc13+sumc14
print(traitreliable)


# In[206]:


sumc15 = 0
itung=df['content'].str.count("continued ")
for itung1 in range(0,len(df)):
    sumc15=sumc15+itung[itung1]
print("continued = ", sumc15 )


# In[207]:


sumc16 = 0
itung=df['content'].str.count("unexpected ")
for itung1 in range(0,len(df)):
    sumc16=sumc16+itung[itung1]
print("unexpected = ", sumc16 )


# In[208]:


sumc17 = 0
itung=df['content'].str.count("giving ")
for itung1 in range(0,len(df)):
    sumc17=sumc17+itung[itung1]
print("giving = ", sumc17 )


# In[209]:


sumc18 = 0
itung=df['content'].str.count("complaining ")
for itung1 in range(0,len(df)):
    sumc18=sumc18+itung[itung1]
print("complaining = ", sumc18 )


# In[210]:


sumc19 = 0
itung=df['content'].str.count("noted ")
for itung1 in range(0,len(df)):
    sumc19=sumc19+itung[itung1]
print("noted = ", sumc19 )


# In[211]:


sumc20 = 0
itung=df['content'].str.count("neither ")
for itung1 in range(0,len(df)):
    sumc20=sumc20+itung[itung1]
print("neither = ", sumc20 )


# In[212]:


sumc21 = 0
itung=df['content'].str.count("convinced ")
for itung1 in range(0,len(df)):
    sumc21=sumc21+itung[itung1]
print("convinced = ", sumc21 )


# In[213]:


sumc22 = 0
itung=df['content'].str.count("dropping ")
for itung1 in range(0,len(df)):
    sumc22=sumc22+itung[itung1]
print("dropping = ", sumc22 )


# In[214]:


sumc23 = 0
itung=df['content'].str.count("sorted ")
for itung1 in range(0,len(df)):
    sumc23=sumc23+itung[itung1]
print("sorted = ", sumc23 )


# In[215]:


sumc24 = 0
itung=df['content'].str.count("bothered ")
for itung1 in range(0,len(df)):
    sumc24=sumc24+itung[itung1]
print("bothered = ", sumc24 )


# In[216]:


sumc25 = 0
itung=df['content'].str.count("secure ")
for itung1 in range(0,len(df)):
    sumc25=sumc25+itung[itung1]
print("secure = ", sumc25 )


# In[217]:


sumc26 = 0
itung=df['content'].str.count("strong ")
for itung1 in range(0,len(df)):
    sumc26=sumc26+itung[itung1]
print("strong = ", sumc26 )


# In[218]:


sumc27 = 0
itung=df['content'].str.count("good ")
for itung1 in range(0,len(df)):
    sumc27=sumc27+itung[itung1]
print("good = ", sumc27 )


# In[219]:


sumc28 = 0
itung=df['content'].str.count("safe ")
for itung1 in range(0,len(df)):
    sumc28=sumc28+itung[itung1]
print("safe = ", sumc28 )


# In[220]:


#trait secure


# In[221]:


traitsecure = sumc15+sumc16+sumc17+sumc18+sumc19+sumc20+sumc21+sumc22+sumc23+sumc24+sumc24+sumc25+sumc26+sumc27+sumc28
print(traitsecure)


# In[222]:


sumc29 = 0
itung=df['content'].str.count("trying ")
for itung1 in range(0,len(df)):
    sumc29=sumc29+itung[itung1]
print("trying = ", sumc29 )


# In[223]:


sumc30 = 0
itung=df['content'].str.count("annoyed ")
for itung1 in range(0,len(df)):
    sumc30=sumc30+itung[itung1]
print("annoyed = ", sumc30 )


# In[224]:


sumc31 = 0
itung=df['content'].str.count("fixed ")
for itung1 in range(0,len(df)):
    sumc31=sumc31+itung[itung1]
print("fixed = ", sumc31 )


# In[225]:


sumc32 = 0
itung=df['content'].str.count("frustrated ")
for itung1 in range(0,len(df)):
    sumc32=sumc32+itung[itung1]
print("frustrated = ", sumc32 )


# In[226]:


sumc33 = 0
itung=df['content'].str.count("granted ")
for itung1 in range(0,len(df)):
    sumc33=sumc33+itung[itung1]
print("granted = ", sumc33 )


# In[227]:


sumc34 = 0
itung=df['content'].str.count("difficult ")
for itung1 in range(0,len(df)):
    sumc34=sumc34+itung[itung1]
print("difficult = ", sumc34 )


# In[228]:


sumc35 = 0
itung=df['content'].str.count("off ")
for itung1 in range(0,len(df)):
    sumc35=sumc35+itung[itung1]
print("off = ", sumc35 )


# In[229]:


sumc36 = 0
itung=df['content'].str.count("connected ")
for itung1 in range(0,len(df)):
    sumc36=sumc36+itung[itung1]
print("connected = ", sumc36 )


# In[230]:


sumc37 = 0
itung=df['content'].str.count("pointed ")
for itung1 in range(0,len(df)):
    sumc37=sumc37+itung[itung1]
print("pointed = ", sumc37 )


# In[231]:


sumc38 = 0
itung=df['content'].str.count("pop ")
for itung1 in range(0,len(df)):
    sumc38=sumc38+itung[itung1]
print("pop = ", sumc38 )


# In[232]:


sumc39 = 0
itung=df['content'].str.count("technical ")
for itung1 in range(0,len(df)):
    sumc39=sumc39+itung[itung1]
print("technical = ", sumc39 )


# In[233]:


sumc40 = 0
itung=df['content'].str.count("expert ")
for itung1 in range(0,len(df)):
    sumc40=sumc40+itung[itung1]
print("expert = ", sumc40 )


# In[234]:


#trait technical


# In[235]:


traittechnical=sumc29+sumc30+sumc31+sumc32+sumc33+sumc34+sumc35+sumc36+sumc37+sumc38+sumc39+sumc40
print(traittechnical)


# In[236]:


sumc41 = 0
itung=df['content'].str.count("learned ")
for itung1 in range(0,len(df)):
    sumc41=sumc41+itung[itung1]
print("learned = ", sumc41 )


# In[237]:


sumc42 = 0
itung=df['content'].str.count("frequent ")
for itung1 in range(0,len(df)):
    sumc42=sumc42+itung[itung1]
print("frequent = ", sumc42 )


# In[238]:


sumc43 = 0
itung=df['content'].str.count("gone ")
for itung1 in range(0,len(df)):
    sumc43=sumc43+itung[itung1]
print("gone = ", sumc43 )


# In[239]:


sumc44 = 0
itung=df['content'].str.count("cutting ")
for itung1 in range(0,len(df)):
    sumc44=sumc44+itung[itung1]
print("cutting = ", sumc44 )


# In[240]:


sumc45 = 0
itung=df['content'].str.count("unacceptable ")
for itung1 in range(0,len(df)):
    sumc45=sumc45+itung[itung1]
print("unacceptable = ", sumc45 )


# In[241]:


sumc46 = 0
itung=df['content'].str.count("oriental ")
for itung1 in range(0,len(df)):
    sumc46=sumc46+itung[itung1]
print("oriental = ", sumc46 )


# In[242]:


sumc47 = 0
itung=df['content'].str.count("competitive ")
for itung1 in range(0,len(df)):
    sumc47=sumc47+itung[itung1]
print("competitive = ", sumc47 )


# In[243]:


sumc48 = 0
itung=df['content'].str.count("worst ")
for itung1 in range(0,len(df)):
    sumc48=sumc48+itung[itung1]
print("worst = ", sumc48 )


# In[244]:


sumc49 = 0
itung=df['content'].str.count("significant ")
for itung1 in range(0,len(df)):
    sumc49=sumc49+itung[itung1]
print("significant = ", sumc49 )


# In[245]:


sumc50 = 0
itung=df['content'].str.count("aware ")
for itung1 in range(0,len(df)):
    sumc50=sumc50+itung[itung1]
print("aware = ", sumc50 )


# In[246]:


sumc51 = 0
itung=df['content'].str.count("corporate ")
for itung1 in range(0,len(df)):
    sumc51=sumc51+itung[itung1]
print("corporate = ", sumc51 )


# In[247]:


sumc52 = 0
itung=df['content'].str.count("bodied ")
for itung1 in range(0,len(df)):
    sumc52=sumc52+itung[itung1]
print("bodied = ", sumc52 )


# In[248]:


sumc53 = 0
itung=df['content'].str.count("embodied ")
for itung1 in range(0,len(df)):
    sumc53=sumc53+itung[itung1]
print("embodied = ", sumc53 )


# In[249]:


sumc54 = 0
itung=df['content'].str.count("incorporated ")
for itung1 in range(0,len(df)):
    sumc54=sumc54+itung[itung1]
print("incorporated = ", sumc54 )


# In[250]:


#trait corporate


# In[251]:


traitcorporate = sumc41+sumc42+sumc43+sumc44+sumc45+sumc46+sumc47+sumc48+sumc49+sumc50+sumc51+sumc52+sumc53+sumc54
print(traitcorporate)


# In[252]:


sumc55 = 0
itung=df['content'].str.count("remarkable ")
for itung1 in range(0,len(df)):
    sumc55=sumc55+itung[itung1]
print("remarkable = ", sumc55 )


# In[253]:


sumc56 = 0
itung=df['content'].str.count("grateful ")
for itung1 in range(0,len(df)):
    sumc56=sumc56+itung[itung1]
print("grateful = ", sumc56 )


# In[254]:


sumc57 = 0
itung=df['content'].str.count("possible ")
for itung1 in range(0,len(df)):
    sumc57=sumc57+itung[itung1]
print("possible = ", sumc57 )


# In[255]:


sumc58 = 0
itung=df['content'].str.count("executed ")
for itung1 in range(0,len(df)):
    sumc58=sumc58+itung[itung1]
print("executed = ", sumc58 )


# In[256]:


sumc59 = 0
itung=df['content'].str.count("delighted ")
for itung1 in range(0,len(df)):
    sumc59=sumc59+itung[itung1]
print("delighted = ", sumc59 )


# In[257]:


sumc60 = 0
itung=df['content'].str.count("spoiled ")
for itung1 in range(0,len(df)):
    sumc60=sumc60+itung[itung1]
print("spoiled = ", sumc60 )


# In[258]:


sumc61 = 0
itung=df['content'].str.count("learned ")
for itung1 in range(0,len(df)):
    sumc61=sumc61+itung[itung1]
print("learned = ", sumc61 )


# In[259]:


sumc62 = 0
itung=df['content'].str.count("hasty ")
for itung1 in range(0,len(df)):
    sumc62=sumc62+itung[itung1]
print("hasty = ", sumc62 )


# In[260]:


sumc63 = 0
itung=df['content'].str.count("meet ")
for itung1 in range(0,len(df)):
    sumc63=sumc63+itung[itung1]
print("meet = ", sumc63 )


# In[261]:


sumc64 = 0
itung=df['content'].str.count("settled ")
for itung1 in range(0,len(df)):
    sumc64=sumc64+itung[itung1]
print("settled = ", sumc64 )


# In[262]:


sumc65 = 0
itung=df['content'].str.count("successful ")
for itung1 in range(0,len(df)):
    sumc65=sumc65+itung[itung1]
print("successful = ", sumc65 )


# In[263]:


#trait successful


# In[264]:


traitsuccessful = sumc55+sumc56+sumc57+sumc58+sumc59+sumc60+sumc61+sumc62+sumc63+sumc64+sumc65
print(traitsuccessful)


# # Dimensi Sophistication

# In[265]:


sums1 = 0
itung=df['content'].str.count("countless ")
for itung1 in range(0,len(df)):
    sums1=sums1+itung[itung1]
print("countless = ", sums1 )


# In[266]:


sums2 = 0
itung=df['content'].str.count("favorite ")
for itung1 in range(0,len(df)):
    sums2=sums2+itung[itung1]
print("favorite = ", sums2 )


# In[267]:


sums3 = 0
itung=df['content'].str.count("world ")
for itung1 in range(0,len(df)):
    sums3=sums3+itung[itung1]
print("world = ", sums3 )


# In[268]:


sums4 = 0
itung=df['content'].str.count("base ")
for itung1 in range(0,len(df)):
    sums4=sums4+itung[itung1]
print("base = ", sums4 )


# In[269]:


sums5 = 0
itung=df['content'].str.count("based ")
for itung1 in range(0,len(df)):
    sums5=sums5+itung[itung1]
print("based = ", sums5 )


# In[270]:


sums6 = 0
itung=df['content'].str.count("traveled ")
for itung1 in range(0,len(df)):
    sums6=sums6+itung[itung1]
print("traveled = ", sums6 )


# In[271]:


sums7 = 0
itung=df['content'].str.count("iconic ")
for itung1 in range(0,len(df)):
    sums7=sums7+itung[itung1]
print("iconic = ", sums7 )


# In[272]:


sums8 = 0
itung=df['content'].str.count("extravagant ")
for itung1 in range(0,len(df)):
    sums8=sums8+itung[itung1]
print("extravagant = ", sums8 )


# In[273]:


sums9 = 0
itung=df['content'].str.count("annual ")
for itung1 in range(0,len(df)):
    sums9=sums9+itung[itung1]
print("annual = ", sums9 )


# In[274]:


sums10 = 0
itung=df['content'].str.count("different ")
for itung1 in range(0,len(df)):
    sums10=sums10+itung[itung1]
print("different = ", sums10 )


# In[275]:


sums11 = 0
itung=df['content'].str.count("glamorous ")
for itung1 in range(0,len(df)):
    sums11=sums11+itung[itung1]
print("glamorous = ", sums11 )


# In[276]:


#trait glamorous


# In[277]:


traitglamourous=sums1+sums2+sums3+sums4+sums5+sums6+sums7+sums8+sums9+sums10+sums11
print(traitglamourous)


# In[278]:


sums12 = 0
itung=df['content'].str.count("sophisticated ")
for itung1 in range(0,len(df)):
    sums12=sums12+itung[itung1]
print("sophisticated = ", sums12 )


# In[279]:


sums13 = 0
itung=df['content'].str.count("exterior ")
for itung1 in range(0,len(df)):
    sums13=sums13+itung[itung1]
print("exterior = ", sums13 )


# In[280]:


sums14 = 0
itung=df['content'].str.count("exquisite ")
for itung1 in range(0,len(df)):
    sums14=sums14+itung[itung1]
print("exquisite = ", sums14 )


# In[281]:


sums15 = 0
itung=df['content'].str.count("attractive ")
for itung1 in range(0,len(df)):
    sums15=sums15+itung[itung1]
print("attractive = ", sums15 )


# In[282]:


sums16 = 0
itung=df['content'].str.count("impressive ")
for itung1 in range(0,len(df)):
    sums16=sums16+itung[itung1]
print("impressive = ", sums16 )


# In[283]:


sums17 = 0
itung=df['content'].str.count("historical ")
for itung1 in range(0,len(df)):
    sums17=sums17+itung[itung1]
print("historical = ", sums17 )


# In[284]:


sums18 = 0
itung=df['content'].str.count("inviting ")
for itung1 in range(0,len(df)):
    sums18=sums18+itung[itung1]
print("inviting = ", sums18 )


# In[285]:


sums19 = 0
itung=df['content'].str.count("suspect ")
for itung1 in range(0,len(df)):
    sums19=sums19+itung[itung1]
print("suspect = ", sums19 )


# In[286]:


sums20 = 0
itung=df['content'].str.count("brilliant ")
for itung1 in range(0,len(df)):
    sums20=sums20+itung[itung1]
print("brilliant = ", sums20 )


# In[287]:


sums21 = 0
itung=df['content'].str.count("famed ")
for itung1 in range(0,len(df)):
    sums21=sums21+itung[itung1]
print("famed = ", sums21 )


# In[288]:


sums22 = 0
itung=df['content'].str.count("charming ")
for itung1 in range(0,len(df)):
    sums22=sums22+itung[itung1]
print("charming = ", sums22 )


# In[289]:


sums23 = 0
itung=df['content'].str.count("magic ")
for itung1 in range(0,len(df)):
    sums23=sums23+itung[itung1]
print("magic = ", sums23 )


# In[290]:


sums24 = 0
itung=df['content'].str.count("magical ")
for itung1 in range(0,len(df)):
    sums24=sums24+itung[itung1]
print("magical = ", sums24 )


# In[291]:


sums25 = 0
itung=df['content'].str.count("witching ")
for itung1 in range(0,len(df)):
    sums25=sums25+itung[itung1]
print("witching = ", sums25 )


# In[292]:


#trait charming


# In[293]:


traitcharming=sums12+sums13+sums14+sums15+sums16+sums17+sums18+sums19+sums20+sums21+sums22+sums23+sums24+sums25
print(traitcharming)


# In[294]:


sums26 = 0
itung=df['content'].str.count("seamless ")
for itung1 in range(0,len(df)):
    sums26=sums26+itung[itung1]
print("seamless = ", sums26 )


# In[295]:


sums27 = 0
itung=df['content'].str.count("swift ")
for itung1 in range(0,len(df)):
    sums27=sums27+itung[itung1]
print("swift = ", sums27 )


# In[296]:


sums28 = 0
itung=df['content'].str.count("sandy ")
for itung1 in range(0,len(df)):
    sums28=sums28+itung[itung1]
print("sandy = ", sums28 )


# In[297]:


sums29 = 0
itung=df['content'].str.count("frosty ")
for itung1 in range(0,len(df)):
    sums29=sums29+itung[itung1]
print("frosty = ", sums29 )


# In[298]:


sums30 = 0
itung=df['content'].str.count("informative ")
for itung1 in range(0,len(df)):
    sums30=sums30+itung[itung1]
print("informative = ", sums30 )


# In[299]:


sums31 = 0
itung=df['content'].str.count("surly ")
for itung1 in range(0,len(df)):
    sums31=sums31+itung[itung1]
print("surly = ", sums31 )


# In[300]:


sums32 = 0
itung=df['content'].str.count("matt ")
for itung1 in range(0,len(df)):
    sums32=sums32+itung[itung1]
print("matt = ", sums32 )


# In[301]:


sums33 = 0
itung=df['content'].str.count("thorough ")
for itung1 in range(0,len(df)):
    sums33=sums33+itung[itung1]
print("thorough = ", sums33 )


# In[302]:


sums34 = 0
itung=df['content'].str.count("understaffed ")
for itung1 in range(0,len(df)):
    sums34=sums34+itung[itung1]
print("understaffed = ", sums34 )


# In[303]:


sums35 = 0
itung=df['content'].str.count("personable ")
for itung1 in range(0,len(df)):
    sums35=sums35+itung[itung1]
print("personable = ", sums35 )


# In[304]:


sums36 = 0
itung=df['content'].str.count("smooth ")
for itung1 in range(0,len(df)):
    sums36=sums36+itung[itung1]
print("smooth = ", sums36 )


# In[305]:


sums37 = 0
itung=df['content'].str.count("bland ")
for itung1 in range(0,len(df)):
    sums37=sums37+itung[itung1]
print("bland = ", sums37 )


# In[306]:


sums38 = 0
itung=df['content'].str.count("fluent ")
for itung1 in range(0,len(df)):
    sums38=sums38+itung[itung1]
print("fluent = ", sums38 )


# In[307]:


sums39 = 0
itung=df['content'].str.count("quiet ")
for itung1 in range(0,len(df)):
    sums39=sums39+itung[itung1]
print("quiet = ", sums39 )


# In[308]:


sums40 = 0
itung=df['content'].str.count("surly ")
for itung1 in range(0,len(df)):
    sums40=sums40+itung[itung1]
print("still = ", sums40 )


# In[309]:


sums41 = 0
itung=df['content'].str.count("tranquil ")
for itung1 in range(0,len(df)):
    sums41=sums41+itung[itung1]
print("tranquil = ", sums41 )


# In[310]:


#trait smooth


# In[311]:


traitsmooth=sums26+sums27+sums28+sums29+sums30+sums31+sums32+sums33+sums34+sums34+sums35+sums36+sums37+sums38+sums39+sums40+sums41
print(traitsmooth)


# # Dimensi Ruggedness

# In[312]:


sumsr1 = 0
itung=df['content'].str.count("indifferent ")
for itung1 in range(0,len(df)):
    sumsr1=sumsr1+itung[itung1]
print("indifferent = ", sumsr1 )


# In[313]:


sumsr2 = 0
itung=df['content'].str.count("personalized ")
for itung1 in range(0,len(df)):
    sumsr2=sumsr2+itung[itung1]
print("personalized = ", sumsr2 )


# In[314]:


sumsr3 = 0
itung=df['content'].str.count("careful ")
for itung1 in range(0,len(df)):
    sumsr3=sumsr3+itung[itung1]
print("careful = ", sumsr3 )


# In[315]:


sumsr4 = 0
itung=df['content'].str.count("accurate ")
for itung1 in range(0,len(df)):
    sumsr4=sumsr4+itung[itung1]
print("accurate = ", sumsr4 )


# In[316]:


sumsr5 = 0
itung=df['content'].str.count("boss ")
for itung1 in range(0,len(df)):
    sumsr5=sumsr5+itung[itung1]
print("boss = ", sumsr5 )


# In[317]:


sumsr6 = 0
itung=df['content'].str.count("friendlier ")
for itung1 in range(0,len(df)):
    sumsr6=sumsr6+itung[itung1]
print("friendlier = ", sumsr6 )


# In[318]:


sumsr7 = 0
itung=df['content'].str.count("appropriate ")
for itung1 in range(0,len(df)):
    sumsr7=sumsr7+itung[itung1]
print("appropriate = ", sumsr7 )


# In[319]:


sumsr8 = 0
itung=df['content'].str.count("certain ")
for itung1 in range(0,len(df)):
    sumsr8=sumsr8+itung[itung1]
print("certain = ", sumsr8 )


# In[320]:


sumsr9 = 0
itung=df['content'].str.count("slight ")
for itung1 in range(0,len(df)):
    sumsr9=sumsr9+itung[itung1]
print("slight = ", sumsr9 )


# In[321]:


sumsr10 = 0
itung=df['content'].str.count("game ")
for itung1 in range(0,len(df)):
    sumsr10=sumsr10+itung[itung1]
print("game = ", sumsr10 )


# In[322]:


sumsr11 = 0
itung=df['content'].str.count("tough ")
for itung1 in range(0,len(df)):
    sumsr11=sumsr11+itung[itung1]
print("tough = ", sumsr11 )


# In[323]:


sumsr12 = 0
itung=df['content'].str.count("rugged ")
for itung1 in range(0,len(df)):
    sumsr12=sumsr12+itung[itung1]
print("rugged = ", sumsr12 )


# In[324]:


sumsr13 = 0
itung=df['content'].str.count("bad ")
for itung1 in range(0,len(df)):
    sumsr13=sumsr13+itung[itung1]
print("bad = ", sumsr13 )


# In[325]:


sumsr14 = 0
itung=df['content'].str.count("hard ")
for itung1 in range(0,len(df)):
    sumsr14=sumsr14+itung[itung1]
print("hard = ", sumsr14 )


# In[326]:


sumsr15 = 0
itung=df['content'].str.count("baffling ")
for itung1 in range(0,len(df)):
    sumsr15=sumsr15+itung[itung1]
print("baffling = ", sumsr15 )


# In[327]:


sumsr16 = 0
itung=df['content'].str.count("elusive ")
for itung1 in range(0,len(df)):
    sumsr16=sumsr16+itung[itung1]
print("elusive = ", sumsr16 )


# In[328]:


sumsr17 = 0
itung=df['content'].str.count("problematic ")
for itung1 in range(0,len(df)):
    sumsr17=sumsr17+itung[itung1]
print("problematic = ", sumsr17 )


# In[329]:


#trait tough


# In[330]:


traittough=sumsr1+sumsr2+sumsr3+sumsr4+sumsr5+sumsr6+sumsr7+sumsr8+sumsr9+sumsr10+sumsr11+sumsr12+sumsr13+sumsr14+sumsr15+sumsr16+sumsr17
print(traittough)


# # Dimensi Sustainability

# In[331]:


sumss1 = 0
itung=df['content'].str.count("gorgeous ")
for itung1 in range(0,len(df)):
    sumss1=sumss1+itung[itung1]
print("gorgeous = ", sumss1 )


# In[332]:


sumss2 = 0
itung=df['content'].str.count("beautiful ")
for itung1 in range(0,len(df)):
    sumss2=sumss2+itung[itung1]
print("beautiful = ", sumss2 )


# In[333]:


sumss3 = 0
itung=df['content'].str.count("decorated ")
for itung1 in range(0,len(df)):
    sumss3=sumss3+itung[itung1]
print("decorated = ", sumss3 )


# In[334]:


sumss4 = 0
itung=df['content'].str.count("immaculate ")
for itung1 in range(0,len(df)):
    sumss4=sumss4+itung[itung1]
print("immaculate = ", sumss4 )


# In[335]:


sumss5 = 0
itung=df['content'].str.count("incredible ")
for itung1 in range(0,len(df)):
    sumss5=sumss5+itung[itung1]
print("incredible = ", sumss5 )


# In[336]:


sumss6 = 0
itung=df['content'].str.count("spotless ")
for itung1 in range(0,len(df)):
    sumss6=sumss6+itung[itung1]
print("spotless = ", sumss6 )


# In[337]:


sumss7 = 0
itung=df['content'].str.count("stunning ")
for itung1 in range(0,len(df)):
    sumss7=sumss7+itung[itung1]
print("stunning = ", sumss7 )


# In[338]:


sumss8 = 0
itung=df['content'].str.count("spacious ")
for itung1 in range(0,len(df)):
    sumss8=sumss8+itung[itung1]
print("spacious = ", sumss8 )


# In[339]:


sumss9 = 0
itung=df['content'].str.count("nice ")
for itung1 in range(0,len(df)):
    sumss9=sumss9+itung[itung1]
print("nice = ", sumss9 )


# In[340]:


sumss10 = 0
itung=df['content'].str.count("spectacular ")
for itung1 in range(0,len(df)):
    sumss10=sumss10+itung[itung1]
print("spectacular = ", sumss10 )


# In[341]:


sumss11 = 0
itung=df['content'].str.count("lovely ")
for itung1 in range(0,len(df)):
    sumss11=sumss11+itung[itung1]
print("lovely = ", sumss11 )


# In[342]:


#trait lovely


# In[343]:


traitlovely=sumss1+sumss2+sumss3+sumss4+sumss5+sumss6+sumss7+sumss8+sumss9+sumss10+sumss11
print(traitlovely)


# In[344]:


sumss12 = 0
itung=df['content'].str.count("adorable ")
for itung1 in range(0,len(df)):
    sumss12=sumss12+itung[itung1]
print("adorable = ", sumss12 )


# In[345]:


sumss13 = 0
itung=df['content'].str.count("expert ")
for itung1 in range(0,len(df)):
    sumss13=sumss13+itung[itung1]
print("expert = ", sumss13 )


# In[346]:


sumss14 = 0
itung=df['content'].str.count("continued ")
for itung1 in range(0,len(df)):
    sumss14=sumss14+itung[itung1]
print("continued = ", sumss14 )


# In[347]:


sumss15 = 0
itung=df['content'].str.count("kindly ")
for itung1 in range(0,len(df)):
    sumss15=sumss15+itung[itung1]
print("kindly = ", sumss15 )


# In[348]:


sumss16 = 0
itung=df['content'].str.count("stressful ")
for itung1 in range(0,len(df)):
    sumss16=sumss16+itung[itung1]
print("stressful = ", sumss16 )


# In[349]:


sumss17 = 0
itung=df['content'].str.count("assisted ")
for itung1 in range(0,len(df)):
    sumss17=sumss17+itung[itung1]
print("assisted = ", sumss17 )


# In[350]:


sumss18 = 0
itung=df['content'].str.count("sorry ")
for itung1 in range(0,len(df)):
    sumss18=sumss18+itung[itung1]
print("sorry = ", sumss18 )


# In[351]:


sumss19 = 0
itung=df['content'].str.count("addressed ")
for itung1 in range(0,len(df)):
    sumss19=sumss19+itung[itung1]
print("addressed = ", sumss19 )


# In[352]:


sumss20 = 0
itung=df['content'].str.count("telling ")
for itung1 in range(0,len(df)):
    sumss20=sumss20+itung[itung1]
print("telling = ", sumss20 )


# In[353]:


sumss21 = 0
itung=df['content'].str.count("miserable ")
for itung1 in range(0,len(df)):
    sumss21=sumss21+itung[itung1]
print("miserable = ", sumss21 )


# In[354]:


sumss22 = 0
itung=df['content'].str.count("direct ")
for itung1 in range(0,len(df)):
    sumss22=sumss22+itung[itung1]
print("direct = ", sumss22 )


# In[355]:


sumss23 = 0
itung=df['content'].str.count("big ")
for itung1 in range(0,len(df)):
    sumss23=sumss23+itung[itung1]
print("big = ", sumss23 )


# In[356]:


sumss24 = 0
itung=df['content'].str.count("bountiful ")
for itung1 in range(0,len(df)):
    sumss24=sumss24+itung[itung1]
print("bountiful = ", sumss24 )


# In[357]:


sumss25 = 0
itung=df['content'].str.count("handsome ")
for itung1 in range(0,len(df)):
    sumss25=sumss25+itung[itung1]
print("handsome = ", sumss25 )


# In[358]:


sumss26 = 0
itung=df['content'].str.count("giving ")
for itung1 in range(0,len(df)):
    sumss26=sumss26+itung[itung1]
print("giving = ", sumss26 )


# In[359]:


sumss27 = 0
itung=df['content'].str.count("liberal ")
for itung1 in range(0,len(df)):
    sumss27=sumss27+itung[itung1]
print("liberal = ", sumss27 )


# In[360]:


#trait giving


# In[361]:


traitgiving= sumss12+sumss13+sumss14+sumss15+sumss16+sumss17+sumss18+sumss19+sumss20+sumss21+sumss22+sumss23+sumss24+sumss25+sumss26+sumss27
print(traitgiving)


# In[362]:


sumss28 = 0
itung=df['content'].str.count("nightly ")
for itung1 in range(0,len(df)):
    sumss28=sumss28+itung[itung1]
print("nightly = ", sumss28 )


# In[363]:


sumss29 = 0
itung=df['content'].str.count("strong ")
for itung1 in range(0,len(df)):
    sumss29=sumss29+itung[itung1]
print("strong = ", sumss29 )


# In[364]:


sumss30 = 0
itung=df['content'].str.count("same ")
for itung1 in range(0,len(df)):
    sumss30=sumss30+itung[itung1]
print("same = ", sumss30 )


# In[365]:


sumss31 = 0
itung=df['content'].str.count("inclusive ")
for itung1 in range(0,len(df)):
    sumss31=sumss31+itung[itung1]
print("inclusive = ", sumss31 )


# In[366]:


sumss32 = 0
itung=df['content'].str.count("own ")
for itung1 in range(0,len(df)):
    sumss32=sumss32+itung[itung1]
print("own = ", sumss32 )


# In[367]:


sumss33 = 0
itung=df['content'].str.count("acceptable ")
for itung1 in range(0,len(df)):
    sumss33=sumss33+itung[itung1]
print("acceptable = ", sumss33 )


# In[368]:


sumss34 = 0
itung=df['content'].str.count("nuts ")
for itung1 in range(0,len(df)):
    sumss34=sumss34+itung[itung1]
print("nuts = ", sumss34 )


# In[369]:


sumss35 = 0
itung=df['content'].str.count("considered ")
for itung1 in range(0,len(df)):
    sumss35=sumss35+itung[itung1]
print("considered = ", sumss35 )


# In[370]:


sumss36 = 0
itung=df['content'].str.count("surprising ")
for itung1 in range(0,len(df)):
    sumss36=sumss36+itung[itung1]
print("surprising = ", sumss36 )


# In[371]:


sumss37 = 0
itung=df['content'].str.count("increased ")
for itung1 in range(0,len(df)):
    sumss37=sumss37+itung[itung1]
print("increased = ", sumss37 )


# In[372]:


sumss38 = 0
itung=df['content'].str.count("fair ")
for itung1 in range(0,len(df)):
    sumss38=sumss38+itung[itung1]
print("fair = ", sumss38 )


# In[373]:


sumss39 = 0
itung=df['content'].str.count("reasonable ")
for itung1 in range(0,len(df)):
    sumss39=sumss39+itung[itung1]
print("reasonable = ", sumss39 )


# In[374]:


sumss40 = 0
itung=df['content'].str.count("bonnie ")
for itung1 in range(0,len(df)):
    sumss40=sumss40+itung[itung1]
print("bonnie = ", sumss40 )


# In[375]:


sumss41 = 0
itung=df['content'].str.count("average ")
for itung1 in range(0,len(df)):
    sumss41=sumss41+itung[itung1]
print("average = ", sumss41 )


# In[376]:


sumss42 = 0
itung=df['content'].str.count("mediocre ")
for itung1 in range(0,len(df)):
    sumss42=sumss42+itung[itung1]
print("mediocre = ", sumss42 )


# In[377]:


sumss43 = 0
itung=df['content'].str.count("clean ")
for itung1 in range(0,len(df)):
    sumss43=sumss43+itung[itung1]
print("clean = ", sumss43 )


# In[378]:


sumss44 = 0
itung=df['content'].str.count("honest ")
for itung1 in range(0,len(df)):
    sumss44=sumss44+itung[itung1]
print("honest = ", sumss44 )


# In[379]:


#trait fair


# In[380]:


traitfair=sumss28+sumss29+sumss30+sumss31+sumss32+sumss33+sumss34+sumss35+sumss36+sumss37+sumss38+sumss39+sumss40+sumss41+sumss42+sumss43+sumss44
print(traitfair)


# In[381]:


sumss45 = 0
itung=df['content'].str.count("same ")
for itung1 in range(0,len(df)):
    sumss45=sumss45+itung[itung1]
print("same = ", sumss45 )


# In[382]:


sumss46 = 0
itung=df['content'].str.count("seasonal ")
for itung1 in range(0,len(df)):
    sumss46=sumss46+itung[itung1]
print("seasonal = ", sumss46 )


# In[383]:


sumss47 = 0
itung=df['content'].str.count("angry ")
for itung1 in range(0,len(df)):
    sumss47=sumss47+itung[itung1]
print("angry = ", sumss47 )


# In[384]:


sumss48 = 0
itung=df['content'].str.count("minimum ")
for itung1 in range(0,len(df)):
    sumss48=sumss48+itung[itung1]
print("minimum = ", sumss48 )


# In[385]:


sumss49 = 0
itung=df['content'].str.count("longer ")
for itung1 in range(0,len(df)):
    sumss49=sumss49+itung[itung1]
print("longer = ", sumss49 )


# In[386]:


sumss50 = 0
itung=df['content'].str.count("considered ")
for itung1 in range(0,len(df)):
    sumss50=sumss50+itung[itung1]
print("considered = ", sumss50 )


# In[387]:


sumss52 = 0
itung=df['content'].str.count("concerned ")
for itung1 in range(0,len(df)):
    sumss52=sumss52+itung[itung1]
print("concerned = ", sumss52 )


# In[388]:


sumss53 = 0
itung=df['content'].str.count("cut ")
for itung1 in range(0,len(df)):
    sumss53=sumss53+itung[itung1]
print("cut = ", sumss53 )


# In[389]:


sumss54 = 0
itung=df['content'].str.count("needless ")
for itung1 in range(0,len(df)):
    sumss54=sumss54+itung[itung1]
print("needless = ", sumss54 )


# In[390]:


sumss55 = 0
itung=df['content'].str.count("significant ")
for itung1 in range(0,len(df)):
    sumss55=sumss55+itung[itung1]
print("significant = ", sumss55 )


# In[391]:


sumss56 = 0
itung=df['content'].str.count("aware ")
for itung1 in range(0,len(df)):
    sumss56=sumss56+itung[itung1]
print("aware = ", sumss56 )


# In[392]:


sumss57 = 0
itung=df['content'].str.count("mindful ")
for itung1 in range(0,len(df)):
    sumss57=sumss57+itung[itung1]
print("mindful = ", sumss57 )


# In[393]:


#trait aware


# In[394]:


traitaware=sumss45+sumss46+sumss47+sumss48+sumss49+sumss50+sumss52+sumss53+sumss54+sumss55+sumss56+sumss57
print(traitaware)


# In[395]:


sumss59 = 0
itung=df['content'].str.count("fab ")
for itung1 in range(0,len(df)):
    sumss59=sumss59+itung[itung1]
print("fab = ", sumss59 )


# In[396]:


sumss60 = 0
itung=df['content'].str.count("classy ")
for itung1 in range(0,len(df)):
    sumss60=sumss60+itung[itung1]
print("classy = ", sumss60 )


# In[397]:


sumss61 = 0
itung=df['content'].str.count("sumptuous ")
for itung1 in range(0,len(df)):
    sumss61=sumss61+itung[itung1]
print("sumptuous = ", sumss61 )


# In[398]:


sumss62 = 0
itung=df['content'].str.count("stylish ")
for itung1 in range(0,len(df)):
    sumss62=sumss62+itung[itung1]
print("stylish = ", sumss62 )


# In[399]:


sumss63 = 0
itung=df['content'].str.count("tidy ")
for itung1 in range(0,len(df)):
    sumss63=sumss63+itung[itung1]
print("tidy = ", sumss63 )


# In[400]:


sumss64 = 0
itung=df['content'].str.count("tasteful ")
for itung1 in range(0,len(df)):
    sumss64=sumss64+itung[itung1]
print("tasteful = ", sumss64 )


# In[401]:


sumss65 = 0
itung=df['content'].str.count("inviting ")
for itung1 in range(0,len(df)):
    sumss65=sumss65+itung[itung1]
print("inviting = ", sumss65 )


# In[402]:


sumss66 = 0
itung=df['content'].str.count("opulent ")
for itung1 in range(0,len(df)):
    sumss66=sumss66+itung[itung1]
print("opulent = ", sumss66 )


# In[403]:


sumss67 = 0
itung=df['content'].str.count("shiny ")
for itung1 in range(0,len(df)):
    sumss67=sumss67+itung[itung1]
print("shiny = ", sumss67 )


# In[404]:


sumss68 = 0
itung=df['content'].str.count("impressive ")
for itung1 in range(0,len(df)):
    sumss68=sumss68+itung[itung1]
print("impressive = ", sumss68 )


# In[405]:


sumss69 = 0
itung=df['content'].str.count("peaceful ")
for itung1 in range(0,len(df)):
    sumss69=sumss69+itung[itung1]
print("peaceful = ", sumss69 )


# In[406]:


#trait peaceful


# In[407]:


traitpeaceful=sumss59+sumss60+sumss61+sumss62+sumss63+sumss64+sumss65+sumss66+sumss67+sumss68+sumss69
print(traitpeaceful)


# # Rekapitulasi Frekuensi  

# In[417]:


# Dimensi sincerity
honest = traithonest/318839
print("honest = ", honest)
sincere = traitsincere/318839
print("sincere = ", sincere)
real = traitreal/318839
print("real = ", real)
original = traitoriginal/318839
print("original = ", original)
cheerful = traitcheerful/318839
print("cheerful = ", cheerful)
friendly = traitfriendly/318839
print("friendly = ", friendly)
totalsincerity=honest+sincere+real+original+cheerful+friendly
print(totalsincerity)


# In[418]:


# Dimensi Excitement
trendy = traittrendy/318839
print("trendy = ", trendy)
exciting = traitexciting/318839
print("exciting = ", exciting)
cool = traitcool/318839
print("cool = ", cool)
young = traityoung/318839
print("young = ", young)
unique = traitunique/318839
print("unique = ", unique)
contemporary = traitcontemporary/318839
print("contemporary = ", contemporary)
totalexcitement=trendy+exciting+cool+young+unique+contemporary
print(totalexcitement)


# In[419]:


# Dimensi Competence
reliable = traitreliable/318839
print("reliable = ", reliable)
secure = traitsecure/318839
print("secure = ", secure)
technical = traittechnical/318839
print("technical = ", technical)
corporate = traitcorporate/318839
print("corporate = ", corporate)
successful = traitsuccessful/318839
print("successful = ", successful)
totalcompetence=reliable+secure+technical+corporate+successful
print(totalcompetence)


# In[420]:


# Dimensi Sophistication
glamorous = traitglamourous/318839
print("glamorous = ", glamorous)
charming = traitcharming/318839
print("charming = ", charming)
smooth = traitsmooth/318839
print("smooth = ", smooth)
totalsophistication=glamorous+charming+smooth
print(totalsophistication)


# In[422]:


# Dimensi Ruggeddness
tough = traittough/318839
print("tough = ", tough)
totalruggeddnes=tough
print(totalruggeddnes)


# In[423]:


# Dimensi Sustainability
lovely = traitlovely/318839
print("lovely = ", lovely)
giving = traitgiving/318839
print("giving = ", giving)
fair = traitfair/318839
print("fair = ", fair)
aware = traitaware/318839
print("aware = ", aware)
peaceful = traitpeaceful/318839
print("peaceful = ", peaceful)
totalsustainability=lovely+giving+fair+aware+peaceful
print(totalsustainability)


# In[416]:


total1 = 0
itung=df['content'].str
itung3=itung.split()
for i in range(0,len(df)):
    total1=total1+len(itung3[i])
print("total = ", total1 )  


# In[111]:


# - itung semua frekuensi
# - rekap per dimensi
# - itung jumlah seluruh kata semuanya buat persentase


# In[3]:


import plotly.express as px
import pandas as pd
df = pd.DataFrame(dict(
    r=[0.0155, 0.0122, 0.00843, 0.00369, 0.0012, 0.02],
    theta=[' sincerity','excitement','competence',
           'sophistication', 'ruggeddnes', 'sustainability']))
fig = px.line_polar(df, r='r', theta='theta', line_close=True)
fig.show()


# In[ ]:




