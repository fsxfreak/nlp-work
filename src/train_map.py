import tensorflow as tf
import numpy as np
import pandas as pd
from gensim.models import Word2Vec as wv

class Trainer(object):
  _SRC_SPACE_FILENAME = '/usr/share/lang-corpus/word2vec/GoogleNews-vectors-negative300.bin'
  _TRG_SPACE_FILENAME = '/usr/share/lang-corpus/word2vec/de-vectors.bin' 

  def __init__(self, train_filename, test_filename):
    #self.src_mdl = wv.load_word2vec_format(_SRC_SPACE_FILENAME, binary=True)
    #self.trg_mdl = wv.load_word2vec_format(_TRG_SPACE_FILENAME, binary=True)

    self.set_labels(train_filename, test_filename)
    self.build_graph()

  def _load_labels(self, filename):
    data = pd.read_csv(filename, sep=' ')
    data.columns = ['src', 'trg']

    return data

  def set_labels(train_filename, test_filename):
    self.train_labels = self._load_labels(train_filename)
    self.test_labels = self._load_labels(test_filename)

  '''
    Builds the Tensorflow computation graph. Should call this every time
    Trainer.set_labels() is called (but not entirely necessary).
  '''
  def build_graph(self):
    self.tf_graph = tf.Graph()

    with tf_graph.as_default():
        x = tf.placeholder(tf.float32, shape=(1,300), name='x')
        y = tf.placeholder(tf.float32, shape=(1,300), name='y')

        W = tf.Variable(tf.random_uniform(shape=(300,300), -1.0, 1.0), name='W')

        y_hat = tf.matmul(x, W, name='y_hat')

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        print(reg_losses)
        loss = tf.reduce_mean(tf.squared_difference(y_hat, y),name='loss')
             + 0.01 * sum(reg_losses)


        self.tf_train_step = tf.train.GradientDescentOptimizer(0.10)
                                     .minimize(loss)
        self.tf_sess = tf.Session()
        self.tf_sess.run(tf.global_variables_initializer())

        assert loss.graph is self.tf_graph

  def train_step():
    pass

  def evaluate(self):
    return 0

def main():
  trainer = Trainer('simple.dict'))

if __name__ == '__main__':
  main()
