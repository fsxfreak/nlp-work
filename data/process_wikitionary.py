import sys, re

def main():
  try:
    filename = sys.argv[1]
  except IndexError:
    filename = 'en-de.dict.raw'

  references = 0
  with open(filename, 'r') as f:
    pattern = re.compile(r"(?P<src>.*)\s[{].*::\s(?P<trg>.*)")
    for idx, line in enumerate(f):
      if 'SEE:' in line:
        pass
      match = pattern.match(line)
      try:
        src = match.group('src')
        targets = match.group('trg')
      except AttributeError:
        continue

  print(references)
if __name__ == '__main__':
  main()
