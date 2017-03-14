import gensim
from gensim.models import Word2Vec as wv 

import numpy as np

def main():
  raw = []
  with open('../data/ground_tm.txt', 'r') as f:
    for line in f:
      row = line.split()
      row = [ float(e) for e in row ]
      raw.append(row)

  tm = np.matrix(raw)
  print(tm)
  print('finished loading translatio nmatrix')

  file_prefix = '/usr/share/lang-corpus/word2vec'
  en_mdl = wv.load_word2vec_format('%s/%s' % 
      ( file_prefix, 'GoogleNews-vectors-negative300.bin'), binary=True)  
  en_mdl.init_sims(replace=True)
  print('finished en model')

  de_mdl = wv.load_word2vec_format('%s/%s' %
      ( file_prefix, 'de-vectors.bin'), binary=True)
  de_mdl.init_sims(replace=True)
  print('finished loading de model')

  output_f = open('../output/predicted-translations-ground.txt', 'w')
  with open('../data/en-de-available.dict', 'r') as f:
    for line in f:
      tokens = line.split()

      src_dict = tokens[0]
      target_dict = tokens[1]

      src_arr = en_mdl[src_dict]
      src_arr.shape=(300,1)

      target_vec = np.matmul(tm, src_arr)
      target_vec.shape=(300,1)
      strength = np.dot(de_mdl[target_dict], target_vec) / (300 * 300)
      print(src_dict, target_dict, target_vec[:10], strength)

if __name__ == '__main__':
  main()
