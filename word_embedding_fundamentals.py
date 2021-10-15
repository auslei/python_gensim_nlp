#%% import libraries
from nltk import stem
import sklearn
import pandas as pd
import re
import numpy as np
from tensorflow.python.keras.backend import reverse, sigmoid

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

def stem_tokenise(corpus, min_length):
    corpus_clean = []
    for doc in corpus:
        tokens = []
        for word in word_tokenize(doc):
            if word not in stopwords.words('english') and len(word)>=min_length:
                stemmed_word = ps.stem(word)
                tokens.append(stemmed_word)
                
        corpus_clean.append(tokens)
    
    return corpus_clean
#%%
corpus_clean = stem_tokenise(corpus, 3)
print(corpus_clean[:2])

#%%
words = []

for w in corpus_clean:
    words += w
    
#words = ["this", "is", "a", "test"]

 # %%
def build_vocab_dictionaries(words, limit):
    c = Counter(words)
    w2i = {}
    i2w = {}
    for idx, (word, _) in enumerate(c.most_common(limit)):
        w2i[word] = idx
        i2w[idx] = word
    
    w2i["UNK"] = -1
    i2w[-1] = "UNK"
    return w2i, i2w

w2i, i2w = build_vocab_dictionaries(words, 10000)     

# %% Skipgram
def build_training_data(words, w2i, window_size):
    N = len(words)
    X = []
    Y = []
    for i in range(N):
        # get the left right window and put context words against the other within the window
        nbr_inds = list(range(max(0, i - window_size), i)) + \
                   list(range(i + 1, min(N, i + window_size + 1)))
        for j in nbr_inds:
            X.append(w2i[words[i]])
            Y.append(w2i[words[j]])

    X = np.array(X)
    Y = np.array(Y)

    return X.reshape(-1, 1), Y.reshape(-1, 1)


X, Y = build_training_data(words, w2i, 5)
# %%
vocab_size = len(i2w)
m = Y.shape[1]

# %%
# turn Y into one hot encoding
from sklearn.preprocessing import OneHotEncoder

X_one_hot = np.zeros((m, vocab_size)) 
Y_one_hot = np.zeros((m, vocab_size))          # this is to just build a matrix of zeors (col - word index, row - Y)
X_one_hot[np.arange(m), X.flatten()] = 1 
Y_one_hot[np.arange(m), Y.flatten()] = 1       # this is to basically use index and set anything in Y for every row to 1

# %%

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import sparse_categorical_crossentropy

#%%
vector_size = 50

model = keras.Sequential([
    Input(shape = [m, vocab_size]),
    Dense(vocab_size, activation = keras.activations.sigmoid),
    Dense(vector_size, activation = keras.activations.sigmoid),
    Dense(vocab_size, activation = keras.activations.softmax)
]
)

model.summary()

model.compile(loss = keras.losses.sparse_categorical_crossentropy , optimizer = "sgd")

history = model.fit(X_one_hot, Y_one_hot, validation_batch_size=0.3, epochs = 30, batch_size = 32,)


# %%
word_vec = model.get_weights()[2]
# %%
word_vec.shape
from sklearn.metrics.pairwise import cosine_similarity

def find_similar_word(word, woi, word_vecs):
    i = woi[word]
    word_vec = word_vecs[i]
    similarity = cosine_similarity([word_vec], word_vecs)
    return np.sort(similarity)[::-1], np.argsort(similarity)[::-1]

# %%
sim, arg = find_similar_word("use", w2i, word_vec)

for i in arg.flatten()[:10]:
    print(i2w[i])
# %%
