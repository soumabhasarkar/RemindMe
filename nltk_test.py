import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
stemmer = LancasterStemmer()

training_data = []
training_data.append({"class":"greeting", "sentence":"how are you?"})
training_data.append({"class":"greeting", "sentence":"how is your day?"})
##training_data.append({"class":"greeting", "sentence":"good day"})
##training_data.append({"class":"greeting", "sentence":"how is it going today?"})
##
##training_data.append({"class":"goodbye", "sentence":"have a nice day"})
##training_data.append({"class":"goodbye", "sentence":"see you later"})
##training_data.append({"class":"goodbye", "sentence":"have a nice day"})
##training_data.append({"class":"goodbye", "sentence":"talk to you soon"})
##
##training_data.append({"class":"sandwich", "sentence":"make me a sandwich"})
##training_data.append({"class":"sandwich", "sentence":"can you make a sandwich?"})
##training_data.append({"class":"sandwich", "sentence":"having a sandwich today?"})
##training_data.append({"class":"sandwich", "sentence":"what's for lunch?"})
##print ("%s sentences in training data" % len(training_data))


words = []
classes = []
documents = []
ignore_words = ['?']
# loop through each sentence in our training data
for pattern in training_data:
    # tokenize each word in the sentence
    w = nltk.word_tokenize(pattern['sentence'])
    print(w)
    # add to our words list
    words.extend(w)
##    # add to documents in our corpus
##    documents.append((w, pattern['class']))
##    # add to our classes list
##    if pattern['class'] not in classes:
##        classes.append(pattern['class'])

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words] 
print(words)
words = list(set(words))
print(words)
# remove duplicates
classes = list(set(classes))
