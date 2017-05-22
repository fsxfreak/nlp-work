import gensim
from gensim.models.keyedvectors import KeyedVectors

FILE_PREFIX = '/usr/share/lang-corpus/word2vec'
SRC_MDL = 'GoogleNews-vectors-negative300-SLIM.bin'
TRG_MDL = 'de-vectors.bin'

def lower_vocab(mdl_file):
  mdl = KeyedVectors.load_word2vec_format('%s/%s' % (FILE_PREFIX, mdl_file), binary=True)  
  lower_vocab = { e.lower() : e for e in mdl.vocab }
  return lower_vocab

def main():
  lower_src = lower_vocab(SRC_MDL)
  lower_trg = lower_vocab(TRG_MDL)

  out_f = open('../data/en-de-available.dict', 'w')
  # does not ever find the phrases for which there are underscores.
  with open('../data/en-de.dict', 'r') as f:
    for line in f:
      words = line.split('|||')

      src = words[0].strip().lower()
      trg = words[1].strip().lower()

      if src not in lower_src or trg not in lower_trg:
        continue
      
      out_f.write('%s %s\n' % (lower_src[src], lower_trg[trg]))

if __name__ == '__main__':
  main()
