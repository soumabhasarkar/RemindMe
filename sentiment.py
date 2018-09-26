from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier


train = [('I love this sandwich.', 'pos'),         
         ('this is an amazing place!', 'pos'),
         ('I feel very good about these beers.', 'pos'),
         ('this is my best work.', 'pos'),
         ("what an awesome view", 'pos'),         
         ('I do not like this restaurant', 'neg'),
         ('I am tired of this stuff.', 'neg'),
         ("I can't deal with this", 'neg'),
         ('he is my sworn enemy!', 'neg'),
         ('my boss is horrible.', 'neg')]
cl = NaiveBayesClassifier(train)
prob_dist = cl.prob_classify("I feel happy this morning.")
print(prob_dist)
##res = cl.classify("This is an amazing library!")
##analysis = TextBlob("Why this taking so long")
##if analysis.sentiment.polarity > 0:
##    print("Pos")
##elif analysis.sentiment.polarity == 0:
##    print("nue")
##else:
##    print("neg")
##
##print(res)
##
##
##testimonial = TextBlob("Textblob is amazingly simple to use. What great fun!")
