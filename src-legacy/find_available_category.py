from nltk.corpus import wordnet as wn
import random

numbers = [
    wn.synset('number.n.02'),
    wn.synset('quantity.n.03'),
    wn.synset('measure.n.03'),
    wn.synset('number.n.04'),
    wn.synset('numeral.n.01'),
    wn.synset('definite_quantity.n.01')
]
animals = wn.synset('animal.n.01')
def is_number(synset):
  synsets = list(synset.closure(lambda s : s.hypernyms(), depth=3))
  for number in numbers:
    if number in synsets:
      return True
  return False

def is_animal(synset):
  synsets = list(synset.closure(lambda s : s.hypernyms(), depth=5))
  return animals in synsets

def test_or_train(output, test_file, train_file):
  TEST_PROB = 0.2
  if random.random() < TEST_PROB:
    test_file.write('%s\n' % output)
  else:
    train_file.write('%s\n' % output)

def main():
  numbers_test = open('../data/en-de-available-number-test.dict', 'w')
  animals_test = open('../data/en-de-available-animal-test.dict', 'w')
  overall_test = open('../data/en-de-available-overall-test.dict', 'w')

  numbers_train = open('../data/en-de-available-number-train.dict', 'w')
  animals_train = open('../data/en-de-available-animal-train.dict', 'w')
  overall_train = open('../data/en-de-available-overall-train.dict', 'w')

  with open('../data/en-de-available.dict', 'r') as f:
    for line in f:
      words = line.split()

      src = words[0].strip()
      target = words[1].strip()

      test_or_train('%s %s' % (src, target), overall_test, overall_train)

      synsets = wn.synsets(src.lower())
      for synset in synsets:
        if is_number(synset):
          output = '%s %s' %(src, target)
          test_or_train(output, numbers_test, numbers_train)
        if is_animal(synset):
          output = '%s %s' %(src, target)
          test_or_train(output, animals_test, animals_train)

if __name__ == '__main__':
  main()
