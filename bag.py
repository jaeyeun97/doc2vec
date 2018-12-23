import numpy as np

def getNgramTokens(n, tokens):
    l = list()
    for i in range(n):
        index = -1 * n + i + 1
        if index == 0:
            tl = tokens[i:]
        else:
            tl = tokens[i:index]
        l.append(tl)
    return list(zip(*l))


class BagOfFrequency(object):
    def __init__(self, grams, tokens):
        self.grams = grams
        self.tokenMap = self.generateTokenMap(grams, tokens) 

    @staticmethod
    def generateTokenMap(grams, tokens):
        tokenMap = dict()
        for i in grams:
            for token in getNgramTokens(i, tokens):
                if token not in tokenMap:
                    tokenMap[token] = 0
                tokenMap[token] += 1
        return tokenMap

    def getTokenCount(self, t):
        if t in self.tokenMap:
            return self.tokenMap[t]
        else:
            return 0


class BagOfPresence(object):
    def __init__(self, grams, tokens):
        self.grams = grams
        self.tokenMap = self.generateTokenSet(grams, tokens) 

    @staticmethod
    def generateTokenSet(grams, tokens):
        tokenSet = set()
        for i in grams:
            for token in getNgramTokens(i, tokens):
                tokenSet.add(token)
        return tokenSet

    @staticmethod
    def generateTokenMap(grams, tokens):
        return dict((token, 1) for token in BagOfPresence.generateTokenSet(grams, tokens))

    def getTokenCount(self, t):
        return 1 if t in self.tokenMap else 0 
