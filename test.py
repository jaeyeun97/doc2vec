import os, glob
from smart_open import smart_open
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import tokenize as _tokenize
from itertools import tee, chain

dataDir = 'aclImdb'

if not os.path.isdir(dataDir):
    raise Exception('Download Data')

folders = ['train', 'test']
sentiments = {'pos': 1, 'neg': -1}

docs = list()
for d in folders:
    for s in sentiments.keys():
        files = glob.glob("{}/{}/{}/*.txt".format(dataDir, d, s))
        l = len(files)
        print("{}/{}: {} files".format(d, s, l))
        for f in files:
            with smart_open(f, 'rb') as f:
                words = list(tokenize(f.read().decode('utf-8'), lowercase=True, deacc=True))
                docs.append(TaggedDocument(words, sentiments[s]))

# return generator of lines for a file
def read(f):
    with smart_open(f, 'rb') as f:
        for l in f:
            yield l.decode('utf-8')
    # return smart_open(f, 'rb').read().decode('utf-8')

# Generator of generators
def flatMap(ls, func=lambda x: x):
    return (func(i) for l in ls for i in l)

# generator of lines -> generator of tokens
def tokenize(lines):
    return (_tokenize(line, lowercase=True, deacc=True) for line in lines)

# s and list of filenames generator
fileLists = ((s, glob.glob("{}/{}/{}/*.txt".format(dataDir, d, s))) for s in sentiments for d in folders)

# s and filename generator
files = ((s,f) for s, files in fileLists for f in files)

# gen of gen of tokens
tokenLists = ((s, tokenize(read(f))) for s, f in files)

# docs
docs = list(TaggedDocument(list(tl), s) for s, tl in tokenLists)
