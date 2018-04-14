import json
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def findword(w):
     return re.compile(r'\b({0})\b'.format(w), re.IGNORECASE).search


analyzer = SentimentIntensityAnalyzer()

reviewreader = open("Electronics.json").read()
#reviewreader = open("Toys_and_Games.json").read()
#reviewreader = open("Automotive.json").read()

#file = open("Reviews/deliveryReviewsE.txt",'w')
file = open("Reviews/E.txt",'w')

#seller delivery

for p in reviewreader.split('\n'):
    if (p !=""):
        p = json.loads(p)
        for key,value in p.items():
            if key =="reviewText" and findword("")(value):
                score = analyzer.polarity_scores(value)["compound"]
                if score >0.05 or score < -0.05:
                    file.write(value+"\n")

file.close()




