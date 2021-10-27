#%%
import process_data as p
import numpy as np
import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.backend import reverse, sigmoid

#%%
df, df_refs = p.load_woc_data("./data/savedrecs.tsv")
t = p.text_tokeniser()
t.load(df.text)

#%%
t.counter.most_common(10)
w2i = t.w2i
i2w = t.i2w

#%% generate positive skip grams:
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

example_sequence = t.word_vectors[0][:10]

window_size = 2
positive_skip_grams, _ = keras.preprocessing.sequence.skipgrams(
      example_sequence,
      vocabulary_size=len(w2i),
      window_size=window_size,
      negative_samples=0)

# illustation only
print([(i, i2w[i]) for i in example_sequence])
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

# print out the samples
print(f"target_word: {target_word}, {i2w[target_word]}, context_word:{context_word}, {i2w[context_word]}")
print("Negative sampled words", [(index.numpy(), i2w[index.numpy()]) for index in negative_sampling_candidates])



#%% Geenrating just one training data
# Add a dimension so you can use concatenation (on the next step).
negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)

# Concat positive context word with negative sampled words.
context = tf.concat([context_class, negative_sampling_candidates], 0)

# Label first context word as 1 (positive) followed by num_ns 0s (negative).
label = tf.constant([1] + [0]*num_ns, dtype="int64")

#%%
print(f"target_index    : {target_word}")
print(f"target_word     : {i2w[target_word]}")
print(f"context_indices : {context}")
print(f"context_words   : {[i2w[c.numpy()] for c in context]}")
print(f"label           : {label}")

#%%

sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(size=100)
print(sampling_table)