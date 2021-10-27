#%% Libraries
from nltk import corpus, text
import pandas as pd
import numpy as np
import re
from pandas import DataFrame
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter


#%%
def load_woc_data(path: str) -> DataFrame:
    # load WoS data 
    df = pd.read_csv(open(path, 'r',  encoding='UTF-8'), sep = "\t", header = 0, encoding = 'utf-8', index_col = False)

    # Get reference from the WoS website for 
    df_refs = pd.read_html("https://images.webofknowledge.com/images/help/WOS/hs_wos_fieldtags.html")[1]
    df_refs.columns = ["alias", "description"]
    df_refs.set_index("alias", inplace = True)
    df_refs['col_name'] = df_refs.description.map(lambda x: "_".join(re.sub(r"[^a-z ]", "",x.lower()).split(" ")[:2]))

    # set properly named columns based on above
    col_names = []
    for col in df.columns:
        col_name = df_refs.loc[col]["col_name"]
        print(col, col_name)
        col_names.append(col_name)
    
    df.columns = col_names
    #Subsetting column names

    required_columns = list(df.count()[df.count() > 150].index)

    df[required_columns].info()
    df = df[required_columns]
    df["text"] = df[['document_title','abstract', 'author_keywords', 'keywords_plus']].fillna(" ").\
                    apply(lambda x: " ".join(x), axis = 1)

    return df, df_refs

# %%
class text_tokeniser():

    def cleanse(text):
        return re.sub(r"[^a-z]", " ", text.lower())

    def __init__(self, fn_text_processing = cleanse,  fn_stemmer = PorterStemmer(),
                 stopwords = stopwords.words('english')):
        self.text_processing = fn_text_processing
        self.stemmer = PorterStemmer()
        self.min_length = 3
        self.stopwords = stopwords
        self.counter = Counter()

    # porter stemmer
    def load(self, corpus):
        self.corpus = []
        
        self.w2i = {}
        self.i2w = {}

        self.w2i['UNK'] = -1
        self.w2i[-1] = 'UNK'

        for text in corpus:
            doc = self.text_processing(text)
            tokens = []
            for word in word_tokenize(doc):
                if word not in stopwords.words('english') and len(word) >= self.min_length:
                    if self.stemmer:
                        word = self.stemmer.stem(word)
                    self.counter[word] +=1
                    tokens.append(word)
            
            self.corpus.append(tokens)

        for ind, key in enumerate(self.counter):
            self.w2i[key] = ind
            self.i2w[ind] = key

        self.word_vectors = []
        for c in self.corpus:
            self.word_vectors.append([self.w2i[w] for w in c])

        # set vocab_size
        self.vocab_size = len(self.w2i)

    # get top most of the words
    def get_most_common(self, n):
        return self.counter.most_common(n)


