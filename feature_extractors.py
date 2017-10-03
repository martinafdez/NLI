"""Feature Extractors

This module contains any classes to be passed to _FeatureUnion_ of the
Pipeline. For simplicity, all the estimators should inherit from
_AbstractFeatureExtractors_ which already takes care of the _fit_ and
_transform_ methods and should implement the _get_features_ method.

"""

import numpy as np
import string
from collections import defaultdict

from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag
from nltk.util import ngrams
from nltk import word_tokenize
from nltk.tag import pos_tag
from nltk.parse.stanford import StanfordParser
from nltk.parse.stanford import StanfordDependencyParser

parser = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
from nltk.tree import Tree

funct_words = set()
data_file = open('Function_word_list.txt', 'r')
lines = data_file.readlines()
for line in lines:
    row = line.rstrip()
    funct_words.add(row)

dependency_parser = StanfordDependencyParser()


class AbstractFeatureExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [list(self.get_features(doc)) for doc in X]

    def get_features(self, doc):
        """Extract features from a document.

        Args:
            doc (dict): A (text, id) dictionary

        Returns:
            float
        """
        raise NotImplementedError


class MeasureLexDiv(AbstractFeatureExtractor):

    def __init__(self, feats="lexdiv"):
        self.feats = feats

    def get_features(self, doc):
        tokens = word_tokenize(doc['text'])
        lexdiv = len(np.unique(tokens)) / len(tokens)
        yield lexdiv

class GetpreBoundary(AbstractFeatureExtractor):

    def __init__(self, speech_feats):
        self.speech_feats = speech_feats

    def get_features(self, doc):
        subset = self.speech_feats.loc[
            self.speech_feats['fileID'] == doc['id']]
        speechvec = subset["preBoundary"].tolist()
        if not speechvec:
            speechvec = [0]
        print(doc['id'], np.sum(speechvec))
        yield np.mean(speechvec)

class GetspeechRate(AbstractFeatureExtractor):

    def __init__(self, speech_feats):
        self.speech_feats = speech_feats

    def get_features(self, doc):
        subset = self.speech_feats.loc[
            self.speech_feats['fileID'] == doc['id']]
        speechvec = subset["speechRate"].tolist()
        if not speechvec:
            speechvec = [0]
        print(doc['id'], np.mean(speechvec))
        yield np.mean(speechvec)


class GetSpeechFeatures(AbstractFeatureExtractor):

    def __init__(self, speech_feats):
        self.speech_feats = speech_feats

    def get_features(self, doc):
        subset = self.speech_feats.loc[
            self.speech_feats['fileID'] == doc['id']]
        speechvec = subset["ey.F1"].tolist()
        if not speechvec:
            speechvec = [0]
        print(doc['id'], np.nanmean(speechvec))
        yield np.nanmean(speechvec)


class NLTKPreprocessor(AbstractFeatureExtractor):
    """
    Transforms input data by using NLTK tokenization, lemmatization, and
    other normalization and filtering techniques.
    """

    def __init__(self, feats='WordNgram', stopwords=None, punct=None, lower=True, strip=True):
        """
        Instantiates the preprocessor, which make load corpora, models, or do
        other time-intenstive NLTK data loading.
        """
        self.feats = feats
        self.lower = lower
        self.strip = strip
        self.stopwords = set(stopwords) if stopwords else set(sw.words('english'))
        self.punct = set(punct) if punct else set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()



    def get_features(self, doc):
        for sent in sent_tokenize(doc['text']):
            sent = sent.lower()
            if self.feats == 'WordNgram':
                tokens = word_tokenize(sent)
                for n in range(1, 3):
                    if len(tokens) < n:
                        sent_ngrams = ngrams(tokens, len(tokens))
                    else:
                        sent_ngrams = ngrams(tokens, n)
                    for ngram in sent_ngrams:
                        yield ngram
            elif self.feats == 'CharNgram':
                chrs = [c for c in sent]
                for n in range(1, 7):
                    sent_ngrams = ngrams(chrs, n)
                    for ngram in sent_ngrams:
                        yield ngram
            elif self.feats == 'PosNgram':
                token = word_tokenize(sent)
                tagged = pos_tag(token)  # doctest: +SKIP
                tags = []
                for tagtoken in tagged:
                    tags.append(tagtoken[1])
                for n in range(1, 5):
                    taggrams = ngrams(tags, n)
                    for ngram in taggrams:
                        yield ngram
            elif self.feats == 'ProdRules':
                parse = list(parser.raw_parse(sent))
                parse2 = [''.join(str(tree)) for tree in parse]
                parse3 = ''.join(parse2)
                ptree = Tree.fromstring(parse3)
                for rule in ptree.productions():
                    yield rule
            elif self.feats == 'FunctWordsSkipgram':
                skip = []
                tokens = wordpunct_tokenize(sent)
                for token in tokens:
                    if token in funct_words:
                        skip.append(token)
                        skipgrams = ngrams(skip, 2)
                        for ngram in skipgrams:
                            yield ngram
            elif self.feats == "ContentSkipGram":
                skip = []
                tokens = wordpunct_tokenize(sent)
                for token in tokens:
                    if token not in funct_words:
                        skip.append(token)
                        skipgrams = ngrams(skip, 2)
                        for ngram in skipgrams:
                            yield ngram
            elif self.feats == 'FunctWordCount':
                frequency = defaultdict(int)
                tokens = wordpunct_tokenize(sent)
                for token in tokens:
                    if token in funct_words:
                        frequency[token] += 1
                functwordfreqs = []
                for funct_word in funct_words:
                    functwordfreqs.append(frequency[funct_word])
                return functwordfreqs
            elif self.feats == 'OverallFunctWordCount':
                functcount = 0
                tokens = wordpunct_tokenize(sent)
                for token in tokens:
                    if token in funct_words:
                        functcount += 1
                        #                print(functcount)
                return functcount
            elif self.feats == 'Dependency':
                result = dependency_parser.raw_parse(sent)
                for dep in result:
                    triples = list(dep.triples())
                    for triple in triples:
                        trip = triple[0][1] + '.' + triple[1] + '.' + triple[2][1]
                        yield trip