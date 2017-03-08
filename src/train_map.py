import gensim
from gensim.models import Word2Vec as wv 

import numpy as np
import pandas as pd

import tensorflow as tf

CONFIDENCE_THRESHOLD = 0.15

# inspired from 
# https://github.com/n8686025/word2vec-translation-matrix/blob/master/jp-en-translation.py
def train(en_mdl, de_mdl, pair_filename):
  source_training_set = []
  target_training_set = []
  with open(pair_filename, 'r') as pairs:
    for line in pairs:
      pair = line.split()
      if float(pair[2]) < CONFIDENCE_THRESHOLD:
        continue

      src_arr = en_mdl[pair[0]]
      src_arr.shape = (300, 1)
      trg_arr = de_mdl[pair[1]]
      trg_arr.shape = (300, 1)
      source_training_set.append(src_arr)
      target_training_set.append(trg_arr)

  x = tf.placeholder(tf.float32, shape=(300, 1), name='x')
  y = tf.placeholder(tf.float32, shape=(300, 1), name='y')

  W = tf.Variable(tf.random_uniform([300, 300], -1.0, 1.0), name='W')
  y_hat = tf.matmul(W, x, name='y_hat')

  loss = tf.reduce_mean(tf.square(tf.sub(y_hat, y)), name='loss')
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = optimizer.minimize(loss)

  #regualization parameter

  saver = tf.train.Saver()

  W_out = None

  # training
  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    print('Sample vector pair...', source_training_set[0][:20], target_training_set[0][:20])

    for epoch in range(50):
      for i in range(len(source_training_set)):
        model_loss = sess.run(train, feed_dict={
                                  x: source_training_set[i],
                                  y: target_training_set[i]
                                })

        W_out = sess.run(W)
        if i % 200 == 0:
          print(epoch, i, 'Training %s' % pair_filename, W_out)

    saver.save(sess, 'tmp/model.ckpt')

  # performance test set, also vary epochs

  '''
  # loading
  with tf.Session() as sess:
    saver.restore(sess, '../saved-models/model.ckpt')
    W_out = sess.run(W)
  '''

  print('Completed translation matrix on %s.' % pair_filename)
  return W_out

def evaluate(en_mdl, de_mdl, matrix, name):
  output_f = open('../output/predicted-translations-%s.txt' % name, 'w')
  with open('../data/en-de-available-number2.dict', 'r') as f:
    for line in f:
      tokens = line.split()
      src_dict = tokens[0]
      target_dict = tokens[1]
      prob = tokens[2]

      src_arr = en_mdl[src_dict]
      src_arr.shape=(300, 1)

      target_vec = np.matmul(matrix, src_arr)
      target_vec.shape=(300,)

      out_str = '%s %s %s %s\n' % (src_dict, de_mdl.similar_by_vector(target_vec), target_dict, prob)
      output_f.write(out_str) 
  with open('../data/en-de-available-animal2.dict', 'r') as f:
    for line in f:
      tokens = line.split()
      src_dict = tokens[0]
      target_dict = tokens[1]
      prob = tokens[2]

      src_arr = en_mdl[src_dict]
      src_arr.shape=(300, 1)

      target_vec = np.matmul(matrix, src_arr)
      target_vec.shape=(300,)

      out_str = '%s %s %s %s\n' % (src_dict, de_mdl.similar_by_vector(target_vec), target_dict, prob)
      output_f.write(out_str) 
  
def main():
  file_prefix = '/usr/share/lang-corpus/word2vec'
  en_mdl = wv.load_word2vec_format('%s/%s' % 
      ( file_prefix, 'GoogleNews-vectors-negative300.bin'), binary=True)  
  en_mdl.init_sims(replace=True)
  print('finished en model')

  de_mdl = wv.load_word2vec_format('%s/%s' %
      ( file_prefix, 'de-vectors.bin'), binary=True)
  de_mdl.init_sims(replace=True)
  print('finished loading de model')

  overall_mat = train(en_mdl, de_mdl, '../data/en-de-available2.dict')
  animal_mat = train(en_mdl, de_mdl, '../data/en-de-available-animal2.dict')
  num_mat = train(en_mdl, de_mdl, '../data/en-de-available-number2.dict')

  print(overall_mat)
  print(animal_mat)
  print(num_mat)

  print('Writing translations to file...')
  evaluate(en_mdl, de_mdl, overall_mat, 'overall')
  evaluate(en_mdl, de_mdl, animal_mat, 'animal')
  evaluate(en_mdl, de_mdl, num_mat, 'number')
  print('Done writing translations to file...')
    

if __name__ == '__main__':
  main()
