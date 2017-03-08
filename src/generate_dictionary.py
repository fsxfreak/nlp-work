

def find_ml(prob_filename):
  '''
  Returns the maximum likelihood translation src-target pairs for each src word 
  
  Example:
    [ (0, 10, 0.239077),
      (2, 2, 0.764512),
      (3, 15, 0.351879),
      (4, 7, 0.366189),
      (5, 4, 0.435935),
      (6, 5, 0.89829),
      (7, 6, 0.444987),
      (8, 144, 0.276489),
      (9, 16, 0.598603),
      (10, 56, 0.675921) ]
  '''
  ml = []
  with open(prob_filename, 'r') as pf:
    iterpf = iter(pf)
    first_line = next(iterpf)
    
    first_vals = first_line.split()
    prev_src = int(first_vals[0])
    max_trg = int(first_vals[1])
    max_prob = float(first_vals[2])

    for line in iterpf:
      vals = line.split()
  
      src = int(vals[0])
      trg = int(vals[1])
      prob = float(vals[2])

      if src == prev_src:
        if prob > max_prob:
          max_trg = trg
          max_prob = prob
      else:
        ml.append((prev_src, max_trg, max_prob))

        max_trg = trg
        max_prob = prob

      prev_src = src

  return ml

def parse_vcb(vcb_filename):
  vocab = []
  with open(vcb_filename, 'r') as vf:
    for line in vf:
      vals = line.split()

      num = int(vals[0])
      word = vals[1]

      if num < len(vocab):
        vocab[num] = word
      else:
        while num > len(vocab):
          vocab.append('')
        vocab.append(word)
        
  return vocab
def main():
  src_vocab = parse_vcb('../data/tst.src.vcb')
  trg_vocab = parse_vcb('../data/tst.trg.vcb')
  print('Done parsing vcb files...')

  mls = find_ml('../data/t3.final')
  print('Finished finding maximum likelihood translations...')

  with open('../data/en-de.dict', 'w') as out:
    for ml in mls:
      src_i, trg_i, prob = ml
      out.write('%s %s %f\n' % (src_vocab[src_i], trg_vocab[trg_i], prob))

  print ('Finished generating dictionary file.')
      
if __name__ == '__main__':
  main()
