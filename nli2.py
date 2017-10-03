#!/usr/bin/env python

import os
import re

import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report as clsr
from sklearn.model_selection import train_test_split as tts
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler




from nltk.corpus.reader.plaintext import CategorizedPlaintextCorpusReader

from feature_extractors import MeasureLexDiv, GetSpeechFeatures, NLTKPreprocessor, GetpreBoundary, GetspeechRate


CORPUS = "/Users/martinafernandez/Dropbox/BULATS/"
DOC_PATTERN = r'[A-Z]\d/\w+/.*\.txt'
CAT_PATTERN = r'[A-Z]\d/(\w+)/.*\.txt'

def identity(arg):
    """
    Simple identity function works as a passthrough.
    """
    return arg


def build_and_evaluate(X, y, clf, speech_feats, verbose=True):
    labels = LabelEncoder()
    y = labels.fit_transform(y)
    X_train, X_test, y_train, y_test = tts(
        X, y, test_size=0.2, random_state=42)
    model = Pipeline([
        ('features', FeatureUnion([
            ('word_ngram_tf-idf', Pipeline([
                ('preprocessor', NLTKPreprocessor(feats='WordNgram')),
                ('vectorizer', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)),
                ('best', TruncatedSVD(n_components=50)),
            ])),
            ('char_ngram_tf-idf', Pipeline([
                ('preprocessor', NLTKPreprocessor(feats='CharNgram')),
                ('vectorizer', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)),
                ('best', TruncatedSVD(n_components=50)),
            ])),
            ('POS-Ngrams', Pipeline([
                ('preprocessor', NLTKPreprocessor(feats='PosNgram')),
                ('vectorizer', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)),
                ('best', TruncatedSVD(n_components=50)),
            ])),
            ('PreBoundarySpeech', GetpreBoundary(speech_feats)),
            ('speechRate', GetspeechRate(speech_feats)),
       #     ('SpeechFeatures', GetSpeechFeatures(speech_feats)),
            ])),
            ('Scaler', StandardScaler()),
            ('classifier', clf)])


    #mo.fit(X_train, y_train)
    X_r = model.fit(X_train).transform(X)
    pca = model.named_steps['classifier']
    print('explained variance ratio (first two components): %s'
          % str(pca.explained_variance_ratio_))

    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange', 'red', 'green', 'purple']
    lw = 2

    for color, i, target_name in zip(colors, range(0, 6), labels.classes_):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of BULATS dataset')
    plt.show()

    return model


if __name__ == "__main__":
    PATH = "model.pickle"
    # Loading speech features
    speech = pd.read_csv("/Users/martinafernandez/UROP2017/SpeechFeatures/allspeechfeats.csv")

    if not os.path.exists(PATH):
        nli = CategorizedPlaintextCorpusReader(CORPUS,
                                               DOC_PATTERN,
                                               cat_pattern=CAT_PATTERN)
        # since `nli` already has all the information (text and ids)
        # you don't need to iterate over it multiple times so
        # construct `X` and `y` in one go.
        X = []
        y = []
        for fileid in nli.fileids():
            X.append({'text': nli.raw(fileid),
                      'id': fileid.split('/')[-1].split('.')[0]})
            y.append(nli.categories(fileid)[0])
        clf = PCA(n_components=2)
        model = build_and_evaluate(X, y, clf, speech)
