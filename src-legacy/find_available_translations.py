import gensim
from gensim.models import Word2Vec as wv 

def main():
  file_prefix = '/usr/share/lang-corpus/word2vec'
  en_mdl = wv.load_word2vec_format('%s/%s' % 
      ( file_prefix, 'GoogleNews-vectors-negative300.bin'), binary=True)  
  en_mdl.init_sims(replace=True)
  de_mdl = wv.load_word2vec_format('%s/%s' %
      ( file_prefix, 'de-vectors.bin'), binary=True)
  de_mdl.init_sims(replace=True)

  lower_dict_en_vocab = { e.lower() : e for e in en_mdl.vocab }
  lower_dict_de_vocab = { e.lower() : e for e in de_mdl.vocab }

  out_f = open('../data/en-de-available.dict', 'w')
  # does not ever find the phrases for which there are underscores.
  with open('../data/en-de.dict', 'r') as f:
    for line in f:
      words = line.split('|||')

      src = words[0].strip().lower()
      trg = words[1].strip().lower()

      if src not in lower_dict_en_vocab or trg not in lower_dict_de_vocab:
        continue
      
      out_f.write('%s %s\n' % (lower_dict_en_vocab[src], lower_dict_de_vocab[trg]))

if __name__ == '__main__':
  main()
