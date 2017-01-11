# import linguamind.linalg as la
# import linguamind.nn as nn
# import linguamind.nlp as nlp

# seed = la.Seed(1)

# mat = la.Matrix(6,10)
# mat.uniform(seed)
# mat -= float(0.5)
# mat /= float(10)

# assert mat[0][0] == 0.04002685472369194
# assert mat[5][9] == -0.006083679385483265

# path = '/Users/amberedmundson/Laboratory/clones/word2vec/text.txt'

# v = nlp.Vocab()
# t = nlp.Text(path,v)

# v2 = nlp.Vocab(t.vocab)
# v2.sort(0)
# v2.InitUnigramTable()
# t.ChangeVocab(v2)

# assert t.sentences[0][0] == 2
# assert t.sentences[0][1] == 1
# assert t.sentences[0][2] == 4

# vocab_size = v2.size
# dim = 10

# seed = la.Seed(1)

# syn0 = nn.SparseLinearInput(vocab_size,dim)
# syn0.weights.uniform(seed)
# syn0.weights -= 0.5
# syn0.weights /= dim

# syn1 = nn.SparseLinearOutput(dim,vocab_size)
# syn1.weights.zero()

# layers = list()
# layers.append(syn0)
# layers.append(syn1)
# layers.append(nn.Relu(vocab_size))

# mlp = nn.Sequential(layers)
# criterion = nn.MSECriterion()
# optim = nn.StochasticGradient(mlp,criterion)

# assert syn0.weights[0][0] == 0.04002685472369194

# cbow = nn.CBOW(t.sentences,v2,5,5)
# cbow.next()
# mlp.forward(cbow.getInputIndicesReference(),cbow.getOutputIndicesReference())
# out = syn0.getOutput().get()
# assert out[0] == -0.0799407958984375
# assert out[1] == -0.02757568471133709
# assert out[9] == 0.02928466722369194
# assert len(out) == 10


import linguamind.linalg as la
import linguamind.nn as nn
import linguamind.nlp as nlp

from collections import Counter
import random

path ='/Users/amberedmundson/Laboratory/datasets/harryPotter/harryPotterAsciiCleaned.txt'
# path = '/Users/amberedmundson/Laboratory/clones/word2vec/text.txt'

v = nlp.Vocab()
t = nlp.Text(path,v)

v2 = nlp.Vocab(t.vocab)
v2.sort(0)
t.ChangeVocab(v2)

v2.InitUnigramTable()

vocab_size = v2.size
dim = 10

seed = la.Seed(1)

syn0 = nn.SparseLinearInput(vocab_size,dim)
syn0.weights.uniform(seed)
syn0.weights -= 0.5
syn0.weights /= dim

syn1neg = nn.SparseLinearOutput(dim,vocab_size)
syn1neg.weights.zero()

layers = list()
layers.append(syn0)
layers.append(syn1neg)
# layers.append(nn.Relu(vocab_size))

mlp = nn.Sequential(layers)
criterion = nn.MSECriterion()
optim = nn.StochasticGradient(mlp,criterion)

v2.getUnigramValue(45000000)
cbow = nn.CBOW(t.sentences,v2,5,5)
cbow.next()
mlp.forward(cbow.getInputIndicesReference(),cbow.getOutputIndicesReference())

assert cbow.getOutputIndicesReference()[0] == 10
assert cbow.getOutputIndicesReference()[1] == 2
assert cbow.getOutputIndicesReference()[2] == 1269
assert cbow.getOutputIndicesReference()[3] == 5953
assert cbow.getOutputIndicesReference()[4] == 1098
assert cbow.getOutputIndicesReference()[5] == 43

