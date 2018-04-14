from django.shortcuts import render
from django.http import HttpResponse

####################################################model import##############################################################
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


documents_f = open("reviewapp/pickled_algos/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()




word_features5k_f = open("reviewapp/pickled_algos/word_features5k.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features



open_file = open("reviewapp/pickled_algos/originalnaivebayes5k.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()


open_file = open("reviewapp/pickled_algos/MNB_classifier5k.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()



open_file = open("reviewapp/pickled_algos/BernoulliNB_classifier5k.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()


open_file = open("reviewapp/pickled_algos/LogisticRegression_classifier5k.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()


open_file = open("reviewapp/pickled_algos/LinearSVC_classifier5k.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()


open_file = open("reviewapp/pickled_algos/SGDC_classifier5k.pickle", "rb")
SGDC_classifier = pickle.load(open_file)
open_file.close()

open_file = open("reviewapp/pickled_algos/SVC_classifier5k.pickle", "rb")
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




def reviewScore(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)

#######################################################################################################################


def index(request):
    reviews = {}
    file = open("reviewapp/marketReview.txt",'r')
    while True:
        review= file.readline()
        review = review[:len(review)-1]
        Rating =file.readline()
        Rating = Rating[:len(Rating)-1]
        if not Rating or not review:break
        reviews[review]=Rating
    sumP = 0
    count = 0
    for key,value in reviews.items():
        sumP+=float(str(value))
        count+=1
    if count !=0:
        sumP = sumP/count
    sumP = round(sumP,2)
    return render(request,'reviewapp/index.html',{'a':reviews,'product':sumP})


def runalgo(request):
    reviews = {}
    file = open("reviewapp/marketReview.txt", 'r')
    while True:
        review = file.readline()
        review = review[:len(review) - 1]
        Rating = file.readline()
        Rating = Rating[:len(Rating) - 1]
        if not Rating or not review: break
        reviews[review] = Rating

    sumP =0
    sumS=0
    sumD =0
    countP=0
    countS=0
    countD = 0
    for key, value in reviews.items():
        catagory,conf = reviewScore(key)
        if(catagory=='pro'):
            print("Product  : "+key)
            sumP += float(str(value))
            countP += 1
        elif(catagory=='sel'):
            print("seller  : " + key)
            sumS += float(str(value))
            countS += 1
        elif (catagory == 'del'):
            print("Delivery  : " + key)
            sumD += float(str(value))
            countD += 1
    if countP!=0:
        sumP = sumP/countP
    if countS!=0:
        sumS = sumS / countS
    if countD!=0:
        sumD = sumD / countD
    print(countP,sumP)
    sumP = round(sumP ,2)
    sumS = round(sumS ,2)
    sumD = round(sumD ,2)

    return render(request,'reviewapp/index.html',{'a':reviews,'product':sumP , 'seller':sumS , 'delivery':sumD})






print(reviewScore("I have had this for a couple months. Used it to load Win7 on a Acer Netbook. Worked just great."))
Vanalyzer = SentimentIntensityAnalyzer()
vs = Vanalyzer.polarity_scores("I have had this for a couple months. Used it to load Win7 on a Acer Netbook. Worked just great.")
print(vs)