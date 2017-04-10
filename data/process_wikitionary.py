import sys, re

def main():
  try:
    filename = sys.argv[1]
  except IndexError:
    filename = 'en-de.dict.raw'

  references = {}

  errors = open('en-de.dict.errors', 'w')
  translations = open('en-de.dict', 'w')

  with open(filename, 'r') as f:
    pattern_see = re.compile(r"(?P<src>.*)\s[{].*SEE[:]\s(?P<trg>.*)\s::")
    pattern = re.compile(r"(?P<src>.*)\s[{].*::\s(?P<trg>.*)")

    for idx, line in enumerate(f):
      if '{proverb}' in line:
        continue

      if 'SEE:' in line:
        match = pattern_see.match(line)
        try:
          src = match.group('src')
          targets = match.group('trg')
          #print(src, 'targets:', targets)
        except AttributeError:
          continue
      else:
        match = pattern.match(line)
        try:
          src = match.group('src')
          targets = match.group('trg')

          # clean text in brackets
          targets = re.sub(r'\[[^\]]*\]', '', targets)
          targets = re.sub(r'\([^\)]*\)', '', targets)
          targets = re.sub(r'\{[^\}]*\}', '', targets)

          targets_split = targets.split(',')
          for target in targets_split:
            target = target.strip()
            
            if target:
              translations.write('%s ||| %s' % (src, target.strip()))
              translations.write('\n')
        except AttributeError:
          errors.write(line)
          continue
if __name__ == '__main__':
  main()
