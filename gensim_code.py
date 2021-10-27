#%%
import process_data as p
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import Word2Vec


#%%
df, df_refs = p.load_woc_data("./data/savedrecs.tsv")
t = p.text_tokeniser()
t.stemmer = None
t.load(df.text)


#%%
model = Word2Vec(sentences=t.corpus, vector_size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")

# %%
vector = model.wv['model']
model.wv.most_similar('bim', topn=10)
# %%
model = Word2Vec.load("word2vec.model")
word_vectors = [np.mean([model.wv[w] for w in c], axis = 0) for c in t.corpus]

# %%
import numpy as np
word_vectors = np.array(word_vectors)

# %%
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
kmeans = KMeans(n_clusters=8, random_state=0).fit(word_vectors)
dbscan = DBSCAN(eps = 0.01) 



# %%
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
p = pca.fit_transform(word_vectors)
# %%
from matplotlib import colors, pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(10, 10)

sns.scatterplot(x = p[:, 0], y = p[:, 1], hue = kmeans.labels_, style = kmeans.labels_, ax = ax)
# %%

# %%
