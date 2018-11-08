# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import unicodedata
import re
import time
import json
from gensim.models import KeyedVectors, Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
nltk.download('wordnet')
nltk.download('stopwords')

def strip_accents(text):
    try:
        text = unicode(text, 'utf-8')
    except NameError: # unicode is a default on python 3 
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)

def cleaning_text(text):
    #Input: list of sentences
    #Output: list of sentences
    output = [re.sub('\W+', ' ', strip_accents(x.lower())) for x in text]
    output = [re.sub('  ', ' ', x) for x in output]
    return output

def remove_stopwords(text):
    filtered_text = []
    for i in range(len(text)):
        word_tokens = word_tokenize(text[i]) 
        filtered_text.append([w for w in word_tokens if not w in stop_words])
    return filtered_text

# Processing
def vectorize(filtered_text, model, Length, frequencies_dict=None):
    print('Vectorisation...')
    start = time.time()
    X = []
    index = []
    i = 0
    for sent in filtered_text:
        sentence_vec = np.zeros((Length,))
        for x in sent:
            while True:
                try:
                    vec = model[x]
                    break
                except KeyError:
                    vec = np.zeros((Length,))
                    break
                
            while True:    
                if frequencies_dict != None:
                    try:
                        weight = frequencies_dict[x]
                        break
                    except KeyError:
                        weight = 0
                        break
                else:
                    weight = 1
                    break
            sentence_vec += weight*vec
        if (sentence_vec != np.zeros((Length,))).all():
            X.append(sentence_vec)
            index.append(i)
        i += 1
    print("Time in %fs" % (time.time()-start))
    return X, index

def svm_classification(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = SGDClassifier(loss='log')#MultinomialNB()
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    print('Training score: %.2f' % (accuracy_score(y_train, y_pred_train)))
    print('Testing score: %.2f' % (accuracy_score(y_test, y_pred_test)))
    print('F1 score: %.2f' % (f1_score(y_test, y_pred_test, average='weighted')))
    
    cm = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=list(set(y_test)), yticklabels=list(set(y_pred_test)))
    return clf

def test(sentence, model, clf, thres, frequencies_dict=None):
    sentence = re.sub('\W+', ' ', strip_accents(sentence.lower()))
    sentence = re.sub('  ', ' ', sentence)
    tokens = [x for x in sentence.split() if x not in stop_words]
    vector = []
    for word in tokens:
        if frequencies_dict != None:
            try:
                weight = frequencies_dict[word]
            except KeyError:
                weight = 1
        else:
            weight = 1
        try:
            vector.append(model.get_vector(word)*weight)
        except KeyError:
            vector.append(np.zeros(model.get_vector('a').shape))
    vector = np.mean(vector, axis=0)
    y_pred_proba = clf.predict_proba(vector.reshape(1, -1))
    print('Sentence: ', sentence)
    print('Tokens: ', tokens)
    if np.max(y_pred_proba) > thres:
        print('Predict label: ', clf.predict(vector.reshape(1, -1)))
    else:
        print('Probability below threshold=0.3')
def main(input_path):
    ### Import data and pre-trained model
    df = pd.read_csv(input_path, sep=';', names=['sentence', 'label'])

    stop_words = set(stopwords.words('french'))
    stop_words.update(['a', 'les', 'oui', 'non', 'merci', 'ok', 'bonjour', 'donc'])
    model = KeyedVectors.load_word2vec_format("./Clustering_Python/Model/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin", binary=True)

    ### Cleaning data
    clean_text = cleaning_text(list(df.sentence))
    filtered_text = remove_stopwords(clean_text) #filtered_text = tokenized sentences
    with open('./wordFrequencies.json', 'r', encoding='utf-8') as f:
        word_frequencies = json.load(f)

    ### Sentence vectorisation
    X, index = vectorize(filtered_text, model, len(model['a'])) #filtered_text
    y = list(df.label[index])

    ### SVM Classification
    clf = svm_classification(X, y)

    ### Prediction on sentence test
    sent1 = "J'ai besoin d'un service de nettoyage"
    sent2 = "J'ai besoin de nettoyer mes vêtements"
    sent3 = "J'ai besoin de repasser ma chemise"
    sent4 = "J'ai besoin de garder mes enfants"

    test(sent1, model, clf, 0.3)
    test(sent2, model, clf, 0.3)
    test(sent3, model, clf, 0.3)
    test(sent4, model, clf, 0.3)

input_path = 'path_to_your_file.csv'
main(input_path)
