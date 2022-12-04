from logging import exception
import pandas as pd
import nltk
import json
from nltk.stem.lancaster import LancasterStemmer
import re


# load the json file
with open('intents.json') as f:
    data = json.load(f)


def preprocess(text):
    """function to lowercase,remove punctuation, tokenize and stemming"""
    lsm = LancasterStemmer()
    text = text.lower()
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    lemm_tokens = nltk.word_tokenize(lsm.stem(text))
    return lemm_tokens


# building the lists required for training

labels = []
training_sentences = [pattern for intent in data['intents'] for pattern in intent['patterns']]
training_labels = [intent['tag'] for intent in data['intents'] for pattern in intent['patterns']]
responses = [intent['responses'] for intent in data['intents']]
labels = [intent['tag'] if intent['tag'] not in labels else exception for intent in data['intents']]
num_classes = len(labels)

pattern_words = [preprocess(pattern) for intent in data['intents'] for pattern in intent['patterns']]
flattened_pattern_words = [i for j in pattern_words for i in j]
flattened_pattern_words = sorted(list(set(flattened_pattern_words)))