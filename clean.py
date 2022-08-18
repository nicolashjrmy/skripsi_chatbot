from lib2to3.pgen2 import token
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy as np
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def kumpulankata(tokenized_sentence, semuakata):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(semuakata), dtype=np.float32)
    for idx, w in enumerate(semuakata):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

