#%% import libraries
from nltk import stem
import pandas as pd
import re
import numpy as np
import tqdm
import tensorflow as tf
from tensorflow import keras

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

#%% vectorise sentence
def vectorise_words(words):
    return [w2i[w] for w in words]

word_vector = vectorise_words(words)


#%% generate positive skip grams:
from tensorflow import keras
import tensorflow as tf

"""
TERMINOLOGY 

While a bag-of-words model predicts a word given the neighboring context, 
a skip-gram model predicts the context (or neighbors) of a word, given the 
word itself. The model is trained on skip-grams, which are n-grams that allow
tokens to be skipped (see the diagram below for an example). The context of 
a word can be represented through a set of skip-gram pairs of 
(target_word, context_word) where context_word appears in the neighboring 
context of target_word.
"""

example_sequence = word_vector[:10]

window_size = 2
positive_skip_grams, _ = keras.preprocessing.sequence.skipgrams(
      example_sequence,
      vocabulary_size=len(w2i),
      window_size=window_size,
      negative_samples=0)

# illustation only
print([i2w[w] for w in example_sequence])
print([(i2w[w], i2w[c]) for w,c in positive_skip_grams])
print(len(positive_skip_grams)) 

#%% negative sampling (for 1)
# Get target and context words for one positive skip-gram.
target_word, context_word = positive_skip_grams[0]
#target_word = [t for t, _ in positive_skip_grams]
#context_word = [c for _, c in positive_skip_grams]

# Set the number of negative samples per positive context.
num_ns = 4
SEED = 42

context_class = tf.reshape(tf.constant(context_word, dtype="int64"), (1, 1))

negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
    true_classes = context_class,  # class that should be sampled as 'positive'
    num_true = 1,  # each positive skip-gram has 1 positive context class
    num_sampled = num_ns,  # number of negative context words to sample
    unique = True,  # all the negative samples should be unique
    range_max= len(w2i),  # pick index of the samples from [0, vocab_size]
    seed = SEED,  # seed for reproducibility
    name = "negative_sampling"  # name of this operation
)
print(negative_sampling_candidates)
print([i2w[index.numpy()] for index in negative_sampling_candidates])

#%% Geenrating just one training data
# Add a dimension so you can use concatenation (on the next step).
negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)

# Concat positive context word with negative sampled words.
context = tf.concat([context_class, negative_sampling_candidates], 0)

# Label first context word as 1 (positive) followed by num_ns 0s (negative).
label = tf.constant([1] + [0]*num_ns, dtype="int64")

#%%
print(f"target_index    : {target}")
print(f"target_word     : {i2w[target_word]}")
print(f"context_indices : {context}")
print(f"context_words   : {[i2w[c.numpy()] for c in context]}")
print(f"label           : {label}")

#%%

sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(size=100)
print(sampling_table)

#%% combine everything

# Generates skip-gram pairs with negative sampling for a list of sequences
# (int-encoded sentences) based on window size, number of negative samples
# and vocabulary size.
def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
  # Elements of each training example are appended to these lists.
  targets, contexts, labels = [], [], []

  # Build the sampling table for vocab_size tokens.
  sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

  # Iterate over all sequences (sentences) in dataset.
  for sequence in tqdm.tqdm(sequences):

    # Generate positive skip-gram pairs for a sequence (sentence).
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sequence,
          vocabulary_size=vocab_size,
          sampling_table=sampling_table,
          window_size=window_size,
          negative_samples=0)

    # Iterate over each positive skip-gram pair to produce training examples
    # with positive context word and negative samples.
    for target_word, context_word in positive_skip_grams:
        
        context_class = tf.expand_dims(tf.constant([context_word], dtype="int64"), 1)
        negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
            true_classes=context_class,
            num_true=1,
            num_sampled=num_ns,
            unique=True,
            range_max=vocab_size,
            seed=SEED,
            name="negative_sampling")

        # Build context and label vectors (for one target word)
        negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)

        context = tf.concat([context_class, negative_sampling_candidates], 0)
        label = tf.constant([1] + [0]*num_ns, dtype="int64")

        # Append each element from the training example to global lists.
        targets.append(target_word)
        contexts.append(context)
        labels.append(label)

  return targets, contexts, labels

## generate training data (for tf go back to:
# https://colab.research.google.com/drive/1YDx370G07SOiVvsZyT5fDfgBGgeu1NSn#scrollTo=nbu8PxPSnVY2&uniqifier=1)

#%%
targets, contexts, labels = generate_training_data([word_vector], 2, 4, len(w2i), SEED)

targets = np.array(targets)
contexts = np.array(contexts)[:,:,0]
labels = np.array(labels)

print('\n')
print(f"targets.shape: {targets.shape}")
print(f"contexts.shape: {contexts.shape}")
print(f"labels.shape: {labels.shape}")



#%%
BATCH_SIZE = 1024
BUFFER_SIZE = 10000
dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print(dataset)

#%%
AUTOTUNE = tf.data.AUTOTUNE
dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
print(dataset)

#%% W2v MODEL
from keras import layers

class Word2Vec(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim):
      
    super(Word2Vec, self).__init__()
    self.target_embedding = layers.Embedding(vocab_size,
                                             embedding_dim,
                                             input_length=1,
                                             name="w2v_embedding")
    
    self.context_embedding = layers.Embedding(vocab_size,
                                              embedding_dim,
                                              input_length = num_ns + 1)

  def call(self, pair):
    target, context = pair
    # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
    # context: (batch, context)
    if len(target.shape) == 2:
      target = tf.squeeze(target, axis=1)
    # target: (batch,)
    word_emb = self.target_embedding(target)
    # word_emb: (batch, embed)
    context_emb = self.context_embedding(context)
    # context_emb: (batch, context, embed)
    dots = tf.einsum('be,bce->bc', word_emb, context_emb)
    # dots: (batch, context)
    return dots


#%% loss function
def custom_loss(x_logit, y_true):
      return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)


#%% logs
vocab_size = len(w2i)
embedding_dim = 128
word2vec = Word2Vec(vocab_size, embedding_dim)
word2vec.compile(optimizer='adam',
                 loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

#%% training
word2vec.fit(dataset, epochs=20, callbacks=[tensorboard_callback])




#%%
import io

weights = word2vec.get_layer('w2v_embedding').get_weights()[0]

out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

for index, word in enumerate(w2i):
  if index == 0:
    continue  # skip 0, it's padding.
  vec = weights[index]
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
  out_m.write(word + "\n")
  
out_v.close()
out_m.close()

#%% some functions
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
    



#+++++++++++++++++++ OLD CODE ++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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

# %%
