import nltk
import random
import pickle
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


documents_f = open("pickled_algos/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()




word_features5k_f = open("pickled_algos/word_features5k.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features



open_file = open("pickled_algos/originalnaivebayes5k.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()


open_file = open("pickled_algos/MNB_classifier5k.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()



open_file = open("pickled_algos/BernoulliNB_classifier5k.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()


open_file = open("pickled_algos/LogisticRegression_classifier5k.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()


open_file = open("pickled_algos/LinearSVC_classifier5k.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()


open_file = open("pickled_algos/SGDC_classifier5k.pickle", "rb")
SGDC_classifier = pickle.load(open_file)
open_file.close()

open_file = open("pickled_algos/SVC_classifier5k.pickle", "rb")
SVC_classifier = pickle.load(open_file)
open_file.close()




voted_classifier = VoteClassifier(
                                  classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  SGDC_classifier,
                                  SVC_classifier)




def review(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)



print(review("I'm really surprised what a great phone this is for $250."
"The battery life is the best I have ever seen, barely hitting 40% after a day"
" of frequent use including Bluetooth audio and navigation. The processor is fast,"
" no lag in apps. The fingerprint sensor works great. I really like the EMUI, much better "
"notification system than the standard Android 6. Camera, reception, display - no regrets here either."
"Also worth mentioning: Unlike my previous Samsung, Motorola and LG phones, you can actually uninstall/remove"
             " all bloatware apps that come with the phone."
+"Definitely recommend this thing!"))
Vanalyzer = SentimentIntensityAnalyzer()
vs = Vanalyzer.polarity_scores("I have had this for a couple months. Used it to load Win7 on a Acer Netbook. Worked just great.")
print(vs)