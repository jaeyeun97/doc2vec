import sys
import numpy as np
from sklearn.feature_extraction import DictVectorizer

sys.path.append('/home/jaeyeun/Cambridge/nlp/thundersvm/python')
from thundersvmScikit import SVC


class Doc2VecSVM(SVC):
    def __init__(self, doc2vec, **kwargs):
        self.doc2vec = doc2vec
        kwargs['kernel'] = 'linear'
        super().__init__(**kwargs)
    
    def fit(self, X, y, **kwargs): 
        X = self.infer_vector(X)
        return super().fit(X, y, **kwargs)

    def infer_vector(self, X):
        return [self.doc2vec.infer_vector(xs) for xs in X]

    def predict(self, X):
        X = self.infer_vector(X)
        return super().predict(X)


class NaiveSVM(SVC):
    def __init__(self, BagClass, grams, cutoff, **kwargs):
        self.BagClass = BagClass
        self.grams = grams
        self.cutoff = cutoff
        self.vectorizer = DictVectorizer(sparse=True, dtype=int)
        kwargs['kernel'] = 'linear'
        super().__init__(**kwargs)
    
    def fit(self, X, y, **kwargs): 
        # X is a list of list of tokens
        bags = [self.BagClass(self.grams, ts) for ts in X] 
        features = frozenset(t for bag in bags for t, v in bag.tokenMap.items() if v >= self.cutoff)
        T = [{t: bag.getTokenCount(t) for t in features} for bag in bags]
        X = self.vectorizer.fit_transform(T)
        return super().fit(X, y, **kwargs)

    def infer_vector(self, X):
        tokenMaps = list(self.BagClass.generateTokenMap(self.grams, xs) for xs in X)
        return self.vectorizer.transform(tokenMaps)

    def predict(self, X):
        X = self.infer_vector(X)
        return super().predict(X)
