#%% import libraries
import enum
import gensim
from nltk import stem
import sklearn
import nltk
import pandas
import spacy
import pandas as pd
import re

#%%
path = "./data/savedrecs.tsv"

# load WoS data 
df = pd.read_csv(open(path, 'r',  encoding='UTF-8'), sep = "\t", header = 0, encoding = 'utf-8', index_col = False)

# %% Get reference from the WoS website for 
df_refs = pd.read_html("https://images.webofknowledge.com/images/help/WOS/hs_wos_fieldtags.html")[1]
df_refs.columns = ["alias", "description"]
df_refs.set_index("alias", inplace = True)
df_refs['col_name'] = df_refs.description.map(lambda x: "_".join(re.sub(r"[^a-z ]", "",x.lower()).split(" ")[:2]))

# %% set properly named columns based on above
col_names = []
for col in df.columns:
    col_name = df_refs.loc[col]["col_name"]
    print(col, col_name)
    col_names.append(col_name)
 
df.columns = col_names
# %% Subsetting column names

required_columns = list(df.count()[df.count() > 150].index)

df[required_columns].info()

# %%
df = df[required_columns]

df["text"] = df[['document_title','abstract', 'author_keywords', 'keywords_plus']].fillna(" ").apply(lambda x: " ".join(x), axis = 1)
# %%
def cleanse(text):
    return re.sub(r"[^a-z]", " ", text.lower())

corpus = df['text'].map(cleanse).values
# %%
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter

# porter stemmer
ps = PorterStemmer()

def stem_tokenise(corpus):
    corpus_clean = []
    for doc in corpus:
        tokens = []
        for word in word_tokenize(doc):
            if word not in stopwords.words('english'):
                stemmed_word = ps.stem(word)
                tokens.append(stemmed_word)
                
        corpus_clean.append(tokens)
    
    return corpus_clean
#%%
corpus_clean = stem_tokenise(corpus)
print(corpus_clean[:2])

#%%
words = []

for w in corpus_clean:
    words += w
    
 # %%
def build_vocab_dictionaries(words, limit):
    c = Counter(words)
    w2i = {}
    i2w = {}
    for idx, (word, count) in enumerate(c.most_common(limit)):
        w2i[word] = idx
        i2w[idx] = word
        
    return w2i, i2w

w2i, i2w = build_vocab_dictionaries(words, 10000)     

# %%
