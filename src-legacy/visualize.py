from sklearn.decomposition import PCA
import matplotlib.pyplot as plot

import gensim
from gensim.models import Word2Vec as wv 

import numpy as np
import random

def plot_pca(xs):
  plot.figure()
  print(PCA(n_components=2).fit(xs).explained_variance_ratio_)
  xs = PCA(n_components=2).fit_transform(xs)

  plot.scatter(xs[:, 0], xs[:, 1])
  plot.draw()

def main():
  file_prefix = '/usr/share/lang-corpus/word2vec'
  en_mdl = wv.load_word2vec_format('%s/%s' % 
      ( file_prefix, 'GoogleNews-vectors-negative300.bin'), binary=True)  
  de_mdl = wv.load_word2vec_format('%s/%s' %
      ( file_prefix, 'de-vectors.bin'), binary=True)
  print('finished loading models')

  pairs = open('../data/en-de-available-number-train.dict', 'r')

  srcs_raw = []
  trgs_raw = []

  for pair in pairs:
    if random.random() < 0.92:
      continue

    src = pair.split()[0]
    trg = pair.split()[1]

    srcs_raw.append(en_mdl[src])
    trgs_raw.append(de_mdl[trg])

  print(len(srcs_raw))
  print(len(trgs_raw))

  srcs = np.array(srcs_raw)
  trgs = np.array(trgs_raw)

  plot_pca(srcs)
  plot_pca(trgs)

  plot.show()

  print('finished plotting')

if __name__ == '__main__':
  main()

