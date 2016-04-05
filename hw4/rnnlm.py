from pycnn import *
import random

LAYERS = 1
INPUT_DIM = 50
HIDDEN_DIM = 50
VOCAB_SIZE = 0

from collections import defaultdict
from itertools import count
import sys
import util

class RNNLanguageModel:
    def __init__(self, model, LAYERS, INPUT_DIM, HIDDEN_DIM,
                 VOCAB_SIZE, builder=SimpleRNNBuilder):
        self.m = model
        self.builder = builder(LAYERS, INPUT_DIM, HIDDEN_DIM, model)

        model.add_lookup_parameters("lookup", (VOCAB_SIZE, INPUT_DIM))
        model.add_parameters("R", (VOCAB_SIZE, HIDDEN_DIM))
        model.add_parameters("bias", (VOCAB_SIZE))

    def BuildLMGraph(self, sent):
        renew_cg()

        # Initialize the decoder with the avergae of input word vectors
        vec = lookup(self.m["lookup"], int(sent[0]))
        for word in sent[1:]:
            vec += lookup(self.m["lookup"], int(word))
        vec /= len(sent)
        init = [vec, tanh(vec)]  # init the cell and hidden layer of the RNN
        init_state = self.builder.initial_state(init)

        R = parameter(self.m["R"])
        bias = parameter(self.m["bias"])
        errs = [] # will hold expressions
        es=[]
        state = init_state
        for (cw,nw) in zip(sent,sent[1:]):
            x_t = lookup(self.m["lookup"], int(cw))
            state = state.add_input(x_t)
            y_t = state.output()
            r_t = bias + (R * y_t)
            err = pickneglogsoftmax(r_t, int(nw))
            errs.append(err)
        nerr = esum(errs)
        return nerr

if __name__ == '__main__':
    train = util.CorpusReader(sys.argv[1])
    vocab = util.Vocab.from_corpus(train)
    VOCAB_SIZE = vocab.size()

    model = Model()
    sgd = AdadeltaTrainer(model)
    lm = RNNLanguageModel(model, LAYERS, INPUT_DIM, HIDDEN_DIM,
                          VOCAB_SIZE, builder=LSTMBuilder)

    train = list(train)
    for ITER in xrange(100):
        random.shuffle(train)
        loss = 0.0
        for i,sent in enumerate(train):
            isent = [vocab.w2i[w] for w in sent]
            errs = lm.BuildLMGraph(isent)
            loss += errs.scalar_value()
            errs.backward()
            sgd.update(1.0)
        print "ITER",ITER,loss
