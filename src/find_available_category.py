from nltk.corpus import wordnet as wn

numbers = wn.synset('definite_quantity.n.01')
animals = wn.synset('animal.n.01')
def is_number(synset):
  synsets = list(synset.closure(lambda s : s.hypernyms(), depth=64))
  return numbers in synsets

def is_animal(synset):
  synsets = list(synset.closure(lambda s : s.hypernyms(), depth=64))
  return animals in synsets

def main():
  numbers_file = open('../data/en-de-available-number.dict', 'w')
  animals_file = open('../data/en-de-available-animal.dict', 'w')
  with open('../data/en-de-available.dict', 'r') as f:
    for line in f:
      words = line.split()

      src = words[0]
      target = words[1]

      synsets = wn.synsets(src.lower().strip())
      for synset in synsets:
        if is_number(synset):
          output = '%s %s' %(src.strip(), target.strip())
          numbers_file.write('%s\n' % output)
        if is_animal(synset):
          output = '%s %s' %(src.strip(), target.strip())
          animals_file.write('%s\n' % output)

if __name__ == '__main__':
  main()
