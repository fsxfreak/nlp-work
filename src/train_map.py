import tensorflow as tf
import numpy as np
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors

import time, math, random, string
print(random.__file__)

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r (%r, %r) %2.2f sec' % \
              (method.__name__, args, kw, te-ts))
        return result
    return timed

class Trainer(object):
  _SPACE_DIR = '/usr/share/lang-corpus/word2vec/'
  _SRC_SPACE_FILENAME = _SPACE_DIR + 'GoogleNews-vectors-negative300-SLIM.bin'
  _TRG_SPACE_FILENAME = _SPACE_DIR + 'de-vectors.bin' 

  _VEC_SHAPE = (1,300)
  _MAP_SHAPE = (300,300)
  _NUM_NEG = 10# k, Lazaridou et. al, 2015

  @timeit
  def __init__(self, train_filename, test_filename):
    self.src_mdl = KeyedVectors.load_word2vec_format(self._SRC_SPACE_FILENAME, binary=True)
    self.trg_mdl = KeyedVectors.load_word2vec_format(self._TRG_SPACE_FILENAME, binary=True)

    self.set_labels(train_filename, test_filename)
    self.build_graph()

  @timeit
  def _load_labels(self, filename):
    # TODO don't ignore first row
    data_raw = pd.read_csv(filename, sep=' ')
    data_raw.columns = ['src', 'trg']

    data = [] 
    for src_raw, trg_raw in list(zip(data_raw.src, data_raw.trg)):
      src_vec = self.src_mdl[src_raw]
      trg_vec = self.trg_mdl[trg_raw]

      src_vec.shape = self._VEC_SHAPE
      trg_vec.shape = self._VEC_SHAPE
      data.append(((src_raw, src_vec), (trg_raw, trg_vec)))

    return data

  def set_labels(self, train_filename, test_filename):
    self.train_labels = self._load_labels(train_filename)
    self.test_labels = self._load_labels(test_filename)

  '''
    Builds the Tensorflow computation graph. Should call this every time
    Trainer.set_labels() is called (but not entirely necessary).
  '''
  def build_graph(self):
    x = tf.placeholder(tf.float32, shape=self._VEC_SHAPE, name='x')
    y = tf.placeholder(tf.float32, shape=self._VEC_SHAPE, name='y')

    W = tf.get_variable('W', shape=self._MAP_SHAPE, dtype=tf.float32,
                      regularizer=tf.contrib.layers.l2_regularizer(0.1))

    y_hat = tf.matmul(x, W, name='y_hat')

    '''
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = (tf.reduce_sum(tf.squared_difference(y_hat, y),name='loss') 
              + 0.01 * sum(reg_losses))
    '''
    negs = [ tf.placeholder(tf.float32, 
                            shape=self._VEC_SHAPE, 
                            name='y_%d' % i) for i in range(self._NUM_NEG) ]

    def dist(t1, t2):
      return tf.divide(tf.acos(
                       tf.divide(tf.reduce_sum(tf.multiply(t1, t2)), 
                       tf.multiply(tf.norm(t1), tf.norm(t2)))
                       ),
                       tf.constant(math.pi))


    gamma = tf.constant(0.75)
    loss = tf.reduce_sum(tf.stack(
            [ tf.maximum(tf.zeros(shape=()), tf.add(gamma,
                         tf.subtract(dist(y_hat, y), 
                                     dist(y_hat, neg)))) for neg in negs ]))
    minimizer = (tf.train.GradientDescentOptimizer(0.005)
                         .minimize(loss))
    sess = tf.Session()

    self.tf_g = {
      'x' : x, 'y' : y, 'W' : W, 'y_hat' : y_hat, 'negs' : negs,
      #'reg_losses' : reg_losses,
      'loss' : loss,
      'minimizer' : minimizer,
      'sess' : sess
    }
    self.tf_g['sess'].run(tf.global_variables_initializer())

    for t in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
      print(t.name)

  def train_step(self):
    for src, trg in self.train_labels:
      src_vec = src[1]
      trg_vec = trg[1]

      feed_dict = {
          self.tf_g['x'] : src_vec,
          self.tf_g['y'] : trg_vec
          }

      W = self.tf_g['sess'].run(self.tf_g['W'])
      # cannot obtain from Session.run() because have not fed the new src_vec yet
      y_hat = np.matmul(src_vec, W).T
      y_hat.shape = (300,)

      def cos_vec(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

      scores = []
      # randomly sample 4x num vectors for the negative examples
      for e in range(self._NUM_NEG * 4):
        seed = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(15))
        neg_poss = self.trg_mdl.seeded_vector(seed)

        score = cos_vec(y_hat, neg_poss) - cos_vec(trg_vec, neg_poss) 
        scores.append((score, neg_poss))
      scores = sorted(scores, key=lambda tup: tup[0], reverse=True)

      for i, neg in enumerate(self.tf_g['negs']):
        feed_dict[neg] = self.trg_mdl[scores[i][0]]
        _, loss = self.tf_g['sess'].run(
                         [ self.tf_g['minimizer'], self.tf_g['loss'] ],
                         feed_dict=feed_dict)
    return loss

  def evaluate(self):
    W = self.tf_g['sess'].run(self.tf_g['W'])

    total_similarity = 0.0
    for src, trg in self.test_labels:
      src_vec = src[1]
      src_word = src[0]
      trg_word = trg[0]

      trg_vec_pred = np.matmul(src_vec, W).T
      trg_vec_pred.shape = (300,)

      print('Golden translation: %s -> %s' % (src_word, trg_word))

      translations = self.trg_mdl.similar_by_vector(trg_vec_pred, topn=5)
      for i, x in enumerate(translations):
        print('\t %d: %s' % (i, x))

      gold_trg_vec = self.trg_mdl[trg_word]
      similarity = (np.dot(trg_vec_pred, gold_trg_vec) /
                      (np.linalg.norm(trg_vec_pred) * np.linalg.norm(gold_trg_vec)))
      print('\tSimilarity to golden: %.5f' % similarity)
      total_similarity = total_similarity + similarity

    print('Translation quality: %.5f' % (total_similarity / len(self.test_labels)))

def main():
  '''
  print('Testing animal on overall train')
  trainer = Trainer('../data/en-de-available-overall-train.dict',
                    '../data/en-de-available-number-test.dict')
  for x in range(10):
    trainer.train_step()
  trainer.evaluate()
  '''

  print('Testing number on number train')
  trainer = Trainer('../data/en-de-available-number-train.dict',
                    '../data/en-de-available-number-test.dict')
  old_loss = trainer.train_step()
  print('loss', old_loss)
  while True:
    new_loss = trainer.train_step()
    print('loss', new_loss)
    if abs(old_loss - new_loss) < 0.001:
      break
    old_loss = new_loss
  trainer.evaluate()

if __name__ == '__main__':
  main()
