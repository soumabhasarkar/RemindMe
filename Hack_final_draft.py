import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
from flask import Flask
import numpy as np
import time
from textblob import TextBlob
import language_check
from flask import request

app = Flask(__name__)

@app.route('/post', methods=['GET', 'POST'])
def text_category():
    text = request.get_data()    
    result = main()
    return result

if __name__=="__main__":
    app.run()

tool = language_check.LanguageTool('en-US')

def main(text):
    zen = TextBlob(text)
    sentences = zen.sentences
    reminder_list = model2(sentences)
    relevence_dict, sen, mom_data, grammer_error = model(sentences)
    category_dict = model1(sentences)
    output = {}
    output['sen'] = sen
    output['mom_data'] = mom_data
    output['rel_data'] = relevence_dict
    output['category_dict'] = category_dict
    output['error'] = grammer_error
    output['reminder_list'] = reminder_list
    out = json.dumps(output)
    # for k,v  in data.items():
    #     if k =='non-business':
    #         print (v)       
    # pass
    # print(out)
    return out 

def model(sentences):
    mom_data = []
    grammer_error = []
    sen = {"pos":[],"neg":[]} 
    relevence_dict = {}
    category_dict = {"business":[],"nonbusiness":[]}
    list_reminder = []
    tmp_bus = [] 
    tmp_non = []    
    for sentence in sentences:        
        sentence = str(sentence)
        result = classify(sentence)
        if result:
            if result[0][0]=="mom" and float(result[0][1])*100 > 95:
                # mom_data.append(sentence)
                # mom_data.append(result)
                dt = [sentence, result[0][0],result[0][1]]
                mom_data.append(dt)
                pass            
            pass
        pass
        sentiment_analysis = TextBlob(sentence)
        if sentiment_analysis.sentiment.polarity > 0:
            sen["pos"].append(sentence)
        elif sentiment_analysis.sentiment.polarity < 0:
            sen["neg"].append(sentence)
        
        if result:
            if result[0][0] in ["greeting","goodbye","mom"] and float(result[0][1])*100 > 90:
                tmp_bus.append([sentence,result[0][0],result[0][1]])                              
                pass
            else:
                tmp_non.append([sentence,result[0][0],result[0][1]])                               
            pass

        # if result:
        #     if result[0][0] == "reminder":
        #         list_reminder.append(sentence)
        #         pass
        #     pass
        matches = tool.check(sentence)
        if matches:
            i=0
            while i<len(matches):
                grammer_error.append(matches[i].context)
                i+=1
                pass            
        pass
    relevence_dict["business"] = tmp_bus
    relevence_dict['nonbusiness'] = tmp_non    
    return relevence_dict, sen, mom_data, grammer_error

stemmer = LancasterStemmer()

training_data = []
training_data.append({"class":"greeting", "sentence":"how are you?"})
training_data.append({"class":"greeting", "sentence":"how is your day?"})
training_data.append({"class":"greeting", "sentence":"Hi, Vilas"})
training_data.append({"class":"greeting", "sentence":"how is it going today?"})
training_data.append({"class":"greeting", "sentence":"I am doing good"})

training_data.append({"class":"goodbye", "sentence":"have a nice day"})
training_data.append({"class":"goodbye", "sentence":"see you later"})
training_data.append({"class":"goodbye", "sentence":"have a nice day"})
training_data.append({"class":"goodbye", "sentence":"talk to you soon"})
training_data.append({"class":"goodbye", "sentence":"thanks for the update"})

training_data.append({"class":"mom", "sentence":"request you to review the request and provide your feedback"})
training_data.append({"class":"mom", "sentence":"has finally been fixed"})
training_data.append({"class":"mom", "sentence":"I worked on this"})
training_data.append({"class":"mom", "sentence":"Work In Progress"})
training_data.append({"class":"mom", "sentence":"Can we have a discussion"})
training_data.append({"class":"mom", "sentence":"Request access for all the developers"})
training_data.append({"class":"mom", "sentence":"need to validate once it is resolved in Production"})
training_data.append({"class":"mom", "sentence":"Can you please look into this and see that we close these tasks"})
training_data.append({"class":"mom", "sentence":"Request you to provide necessary access rights"})
training_data.append({"class":"mom", "sentence":"create new meeting invite for the daily sync up call using AT&T"})
training_data.append({"class":"mom", "sentence":"I will schedule the meeting but someone has to initiate the call"})
training_data.append({"class":"mom", "sentence":"I’m hoping to get it scheduled for next week"})
training_data.append({"class":"mom", "sentence":"Please email your feedback to me by eod tomorrow "})
training_data.append({"class":"mom", "sentence":"Will sit with the team and close tickets that have been done"})



words = []
classes = []
documents = []
ignore_words = ['?']
# loop through each sentence in our training data
for pattern in training_data:
    # tokenize each word in the sentence
    w = nltk.word_tokenize(pattern['sentence'])
    # add to our words list
    words.extend(w)
    # add to documents in our corpus
    documents.append((w, pattern['class']))
    # add to our classes list
    if pattern['class'] not in classes:
        classes.append(pattern['class'])

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
##words = list(set(words))

# remove duplicates
##classes = list(set(classes))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)

# create our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    training.append(bag)
    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    output.append(output_row)

# sample training/output
i = 3
w = documents[i][0]
print ([stemmer.stem(word.lower()) for word in w])
print (training[i])
print (output[i])

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)
 
def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

def think(sentence, show_details=False):
    x = bow(sentence.lower(), words, show_details)
    if show_details:
        print ("sentence:", sentence, "\n bow:", x)
    # input layer is our bag of words
    l0 = x
    # matrix multiplication of input and hidden layer
    l1 = sigmoid(np.dot(l0, synapse_0))
    # output layer
    l2 = sigmoid(np.dot(l1, synapse_1))
    return l2

	
def train(X, y, hidden_neurons=10, alpha=1, epochs=50000, dropout=False, dropout_percent=0.5):

    print ("Training with %s neurons, alpha:%s, dropout:%s %s" % (hidden_neurons, str(alpha), dropout, dropout_percent if dropout else '') )
    print ("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X),len(X[0]),1, len(classes)) )
    np.random.seed(1)

    last_mean_error = 1
    # randomly initialize our weights with mean 0
    synapse_0 = 2*np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_1 = 2*np.random.random((hidden_neurons, len(classes))) - 1

    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    prev_synapse_1_weight_update = np.zeros_like(synapse_1)

    synapse_0_direction_count = np.zeros_like(synapse_0)
    synapse_1_direction_count = np.zeros_like(synapse_1)
        
    for j in iter(range(epochs+1)):

        # Feed forward through layers 0, 1, and 2
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))
                
        if(dropout):
            layer_1 *= np.random.binomial([np.ones((len(X),hidden_neurons))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))

        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        # how much did we miss the target value?
        layer_2_error = y - layer_2

        if (j% 10000) == 0 and j > 5000:
            # if this 10k iteration's error is greater than the last iteration, break out
            if np.mean(np.abs(layer_2_error)) < last_mean_error:
                print ("delta after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error))) )
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                print ("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error )
                break
                
        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        layer_1_error = layer_2_delta.dot(synapse_1.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
        
        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))
        
        if(j > 0):
            synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))
            synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0) + 0))        
        
        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update
        
        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update

    now = datetime.datetime.now()

    # persist synapses
    synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
               'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'words': words,
               'classes': classes
              }
    synapse_file = "synapses.json"

    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True)
    print ("saved synapses to:", synapse_file)
	
	
	
X = np.array(training)
y = np.array(output)

start_time = time.time()

# train(X, y, hidden_neurons=20, alpha=0.1, epochs=100000, dropout=False, dropout_percent=0.2)

elapsed_time = time.time() - start_time
print ("processing time:", elapsed_time, "seconds")



# probability threshold
ERROR_THRESHOLD = 0.2
# load our calculated synapse values
synapse_file = 'synapses.json' 
with open(synapse_file) as data_file: 
    synapse = json.load(data_file) 
    synapse_0 = np.asarray(synapse['synapse0']) 
    synapse_1 = np.asarray(synapse['synapse1'])

def classify(sentence, show_details=False):
    results = think(sentence, show_details)
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD ] 
    results.sort(key=lambda x: x[1], reverse=True) 
    return_results =[[classes[r[0]],r[1]] for r in results]
    # print ("%s \n classification: %s" % (sentence, return_results))
    # if return_results:
    #     print(return_results[0][0])
    #     pass    
    return return_results


########################################################################################################################################

training_data1 = []
training_data1.append({"class":"db", "sentence":"Implementation will be same underlying DB Schema upgrade script"})
training_data1.append({"class":"db", "sentence":"Pull the right DBA snapshot and create DB for any enviornment"})
training_data1.append({"class":"db", "sentence":"DB deployment scripts"})
training_data1.append({"class":"db", "sentence":"Database Upgrade on both environment"})
training_data1.append({"class":"db", "sentence":"DB infrabuild-up"})
training_data1.append({"class":"db", "sentence":"MySQL RDS Backup strategy"})
training_data1.append({"class":"db", "sentence":"Oracle RDS backup strategy"})
training_data1.append({"class":"db", "sentence":"DB Migration checklist completed and approved"})
training_data1.append({"class":"db", "sentence":"exact fetch returns more than requested number of rows"})


training_data1.append({"class":"bug", "sentence":"This is a blocker to branching and merging tests"})
training_data1.append({"class":"bug", "sentence":"a solution does not exist and this is a blocker"})
training_data1.append({"class":"bug", "sentence":"deployment failed and now enrollment is not working"})
training_data1.append({"class":"bug", "sentence":"The underlying provider failed on Open.; Cannot find Oracle Home"})
training_data1.append({"class":"bug", "sentence":"invalid username/password; logon denied"})
training_data1.append({"class":"bug", "sentence":"An error occurred while updating the entries. See the inner exception for details"})
training_data1.append({"class":"bug", "sentence":"I worked on bug fixing"})


training_data1.append({"class":"review", "sentence":"Please review before executing burp suite"})
training_data1.append({"class":"review", "sentence":"how long the tests will run"})
#training_data1.append({"class":"review", "sentence":"I recall there were some pre-tasks need to completed before we run burp suite"})
training_data1.append({"class":"review", "sentence":"Do we have any update"})
training_data1.append({"class":"review", "sentence":"Tests will be performed first on Dev2 Environment"})
training_data1.append({"class":"review", "sentence":"Performance test to be captured in the project plan"})
training_data1.append({"class":"review", "sentence":"include items that are not in the project plan"})
# training_data1.append({"class":"review", "sentence":"Review Jira tickets and reconcile against project plan to find items project plan items that are in progress or complete"})
training_data1.append({"class":"review", "sentence":"We did not have time to review duration"})
training_data1.append({"class":"review", "sentence":"Review open jira tickets and close ones that are complete"})
training_data1.append({"class":"review", "sentence":"Review DOC in Confluence"})
training_data1.append({"class":"review", "sentence":"Please take the next day to review the tasks"})
training_data1.append({"class":"review", "sentence":"We will meet on Thu morning"})
training_data1.append({"class":"review", "sentence":"Content for the merge to be reviewed by Jeff "})
training_data1.append({"class":"review", "sentence":"Jira task cleanup activity by tomorrow"})
training_data1.append({"class":"review", "sentence":"Prasant will complete it Today EOD"})
training_data1.append({"class":"review", "sentence":"QA to confirm to manually deploy antivirus"})
training_data1.append({"class":"review", "sentence":"Reviewed by network, update document shared by Srinath"})
training_data1.append({"class":"review", "sentence":"Review and Sign off by Affinion "})
training_data1.append({"class":"review", "sentence":"Update the scripts based on above for all the servers to automatically deploy the certificates"})
training_data1.append({"class":"review", "sentence":"Script created and validated by Kranthi"})
training_data1.append({"class":"review", "sentence":"Detailed Migration plan created and reviewed"})
training_data1.append({"class":"review", "sentence":"Document and review with security team the Keyroll testing approach"})
training_data1.append({"class":"review", "sentence":"Review changes from 17.3 regression branch to master"})
training_data1.append({"class":"review", "sentence":"Review and signoff of the automated deployment scripts"})



words1 = []
classes1 = []
documents1 = []
ignore_words1 = ['?']
# loop through each sentence in our training1 data
for pattern in training_data1:
    # tokenize each word in the sentence
    w = nltk.word_tokenize(pattern['sentence'])
    # add to our words1 list
    words1.extend(w)
    # add to documents1 in our corpus
    documents1.append((w, pattern['class']))
    # add to our classes1 list
    if pattern['class'] not in classes1:
        classes1.append(pattern['class'])

# stem and lower each word and remove duplicates
words1 = [stemmer.stem(w.lower()) for w in words1 if w not in ignore_words1]

# create our training1 data
training1 = []
output1 = []
# create an empty array for our output1
output_empty1 = [0] * len(classes1)

# training1 set, bag of words1 for each sentence
for doc in documents1:
    # initialize our bag of words1
    bag = []
    # list of tokenized words1 for the pattern
    pattern_words1 = doc[0]
    # stem each word
    pattern_words1 = [stemmer.stem(word.lower()) for word in pattern_words1]
    # create our bag of words1 array
    for w in words1:
        bag.append(1) if w in pattern_words1 else bag.append(0)

    training1.append(bag)
    # output1 is a '0' for each tag and '1' for current tag
    output1_row = list(output_empty1)
    output1_row[classes1.index(doc[1])] = 1
    output1.append(output1_row)

def think_category(sentence, show_details=False):
    x = bow(sentence.lower(), words1, show_details)
    if show_details:
        print ("sentence:", sentence, "\n bow:", x)
    # input layer is our bag of words1
    l0 = x
    # matrix multiplication of input and hidden layer
    l1 = sigmoid(np.dot(l0, synapse_01))
    # output1 layer
    l2 = sigmoid(np.dot(l1, synapse_11))
    return l2

    
def train_category(X, y, hidden_neurons=10, alpha=1, epochs=50000, dropout=False, dropout_percent=0.5):

    print ("training1 with %s neurons, alpha:%s, dropout:%s %s" % (hidden_neurons, str(alpha), dropout, dropout_percent if dropout else '') )
    print ("Input matrix: %sx%s    output1 matrix: %sx%s" % (len(X),len(X[0]),1, len(classes1)) )
    np.random.seed(1)

    last_mean_error = 1
    # randomly initialize our weights with mean 0
    synapse_01 = 2*np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_11 = 2*np.random.random((hidden_neurons, len(classes1))) - 1

    prev_synapse_01_weight_update = np.zeros_like(synapse_01)
    prev_synapse_11_weight_update = np.zeros_like(synapse_11)

    synapse_01_direction_count = np.zeros_like(synapse_01)
    synapse_11_direction_count = np.zeros_like(synapse_11)
        
    for j in iter(range(epochs+1)):

        # Feed forward through layers 0, 1, and 2
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0, synapse_01))
                
        if(dropout):
            layer_1 *= np.random.binomial([np.ones((len(X),hidden_neurons))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))

        layer_2 = sigmoid(np.dot(layer_1, synapse_11))

        # how much did we miss the target value?
        layer_2_error = y - layer_2

        if (j% 10000) == 0 and j > 5000:
            # if this 10k iteration's error is greater than the last iteration, break out
            if np.mean(np.abs(layer_2_error)) < last_mean_error:
                print ("delta after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error))) )
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                print ("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error )
                break
                
        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        layer_1_error = layer_2_delta.dot(synapse_11.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
        
        synapse_11_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_01_weight_update = (layer_0.T.dot(layer_1_delta))
        
        if(j > 0):
            synapse_01_direction_count += np.abs(((synapse_01_weight_update > 0)+0) - ((prev_synapse_01_weight_update > 0) + 0))
            synapse_11_direction_count += np.abs(((synapse_11_weight_update > 0)+0) - ((prev_synapse_11_weight_update > 0) + 0))        
        
        synapse_11 += alpha * synapse_11_weight_update
        synapse_01 += alpha * synapse_01_weight_update
        
        prev_synapse_01_weight_update = synapse_01_weight_update
        prev_synapse_11_weight_update = synapse_11_weight_update

    now = datetime.datetime.now()

    # persist synapses
    synapse = {'synapse01': synapse_01.tolist(), 'synapse11': synapse_11.tolist(),
               'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'words1': words1,
               'classes1': classes1
              }
    synapse_file = "synapses_category.json"

    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True)
    print ("saved synapses to:", synapse_file)
    
    
    
X = np.array(training1)
y = np.array(output1)

start_time = time.time()

train_category(X, y, hidden_neurons=20, alpha=0.1, epochs=100000, dropout=False, dropout_percent=0.2)

elapsed_time = time.time() - start_time
print ("processing time:", elapsed_time, "seconds")



# probability threshold
ERROR_THRESHOLD = 0.2
# load our calculated synapse values
synapse_file = 'synapses_category.json' 
with open(synapse_file) as data_file: 
    synapse = json.load(data_file) 
    synapse_01 = np.asarray(synapse['synapse01']) 
    synapse_11 = np.asarray(synapse['synapse11'])

def classify_category(sentence, show_details=False):
    results = think_category(sentence, show_details)
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD ] 
    results.sort(key=lambda x: x[1], reverse=True) 
    return_results =[[classes1[r[0]],r[1]] for r in results]
    # print ("%s \n classification: %s" % (sentence, return_results))
    # if return_results:
    #     print(return_results[0][0])
    #     pass    
    return return_results

def model1(sentences):
    category_dict = {"db":[],"bug":[],"review":[]}
    for sentence in sentences:        
        sentence = str(sentence)
        result = classify_category(sentence)
        if result:
            if result[0][0] == "db" and float(result[0][1])*100 > 99:
                category_dict["db"].append([sentence,result[0][0],result[0][1]])                                              
            elif result[0][0] == "bug" and float(result[0][1])*100 > 99:
                category_dict["bug"].append([sentence,result[0][0],result[0][1]]) 
            elif result[0][0] == "review" and float(result[0][1])*100 > 99:
                category_dict["review"].append([sentence,result[0][0],result[0][1]]) 
            pass
        pass
    return category_dict  


###########################################################################################################################################

training_data2 = []
training_data2.append({"class":"reminder", "sentence":"Can we have a call at 6PM today"})
training_data2.append({"class":"reminder", "sentence":"we are meeting at 4pm today"})
training_data2.append({"class":"reminder", "sentence":"we will discuss in today's sync up call"})
training_data2.append({"class":"reminder", "sentence":"Please join spu call today by 7.30pm"})
training_data2.append({"class":"reminder", "sentence":"please share the meeting link"})
training_data2.append({"class":"reminder", "sentence":"Can we schedule a meeting with"})
training_data2.append({"class":"reminder", "sentence":"Make sure we will merge code to master tomorrow"})
training_data2.append({"class":"reminder", "sentence":"meeting invite for the daily sync up call"})
training_data2.append({"class":"reminder", "sentence":"Performance daily sync-up at Weekly from 4pm to 4:30pm"})
training_data2.append({"class":"reminder", "sentence":"Kickoff meeting at Tue Oct 4, 2016"})
training_data2.append({"class":"reminder", "sentence":"Business as usual on 14-Sep-16"})
training_data2.append({"class":"reminder", "sentence":"Holiday Declared for tomorrow dated 13-Sep-16"})

training_data2.append({"class":"notremind", "sentence":"how are you?"})
training_data2.append({"class":"notremind", "sentence":"how is your day?"})
training_data2.append({"class":"notremind", "sentence":"good day"})
training_data2.append({"class":"notremind", "sentence":"how is it going today?"})



words2 = []
classes2 = []
documents2 = []
ignore_words2 = ['?']
# loop through each sentence in our training2 data
for pattern in training_data2:
    # tokenize each word in the sentence
    w = nltk.word_tokenize(pattern['sentence'])
    # add to our words2 list
    words2.extend(w)
    # add to documents2 in our corpus
    documents2.append((w, pattern['class']))
    # add to our classes2 list
    if pattern['class'] not in classes2:
        classes2.append(pattern['class'])

# stem and lower each word and remove duplicates
words2 = [stemmer.stem(w.lower()) for w in words2 if w not in ignore_words2]

# create our training2 data
training2 = []
output2 = []
# create an empty array for our output2
output_empty2 = [0] * len(classes2)

# training2 set, bag of words2 for each sentence
for doc in documents2:
    # initialize our bag of words2
    bag = []
    # list of tokenized words2 for the pattern
    pattern_words2 = doc[0]
    # stem each word
    pattern_words2 = [stemmer.stem(word.lower()) for word in pattern_words2]
    # create our bag of words2 array
    for w in words2:
        bag.append(1) if w in pattern_words2 else bag.append(0)

    training2.append(bag)
    # output2 is a '0' for each tag and '1' for current tag
    output2_row = list(output_empty2)
    output2_row[classes2.index(doc[1])] = 1
    output2.append(output2_row)

def think_reminder(sentence, show_details=False):
    x = bow(sentence.lower(), words2, show_details)
    if show_details:
        print ("sentence:", sentence, "\n bow:", x)
    # input layer is our bag of words2
    l0 = x
    # matrix multiplication of input and hidden layer
    l1 = sigmoid(np.dot(l0, synapse_02))
    # output2 layer
    l2 = sigmoid(np.dot(l1, synapse_12))
    return l2

    
def train_reminder(X, y, hidden_neurons=10, alpha=1, epochs=50000, dropout=False, dropout_percent=0.5):

    print ("training2 with %s neurons, alpha:%s, dropout:%s %s" % (hidden_neurons, str(alpha), dropout, dropout_percent if dropout else '') )
    print ("Input matrix: %sx%s    output2 matrix: %sx%s" % (len(X),len(X[0]),1, len(classes2)) )
    np.random.seed(1)

    last_mean_error = 1
    # randomly initialize our weights with mean 0
    synapse_02 = 2*np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_12 = 2*np.random.random((hidden_neurons, len(classes2))) - 1

    prev_synapse_02_weight_update = np.zeros_like(synapse_02)
    prev_synapse_12_weight_update = np.zeros_like(synapse_12)

    synapse_02_direction_count = np.zeros_like(synapse_02)
    synapse_12_direction_count = np.zeros_like(synapse_12)
        
    for j in iter(range(epochs+1)):

        # Feed forward through layers 0, 1, and 2
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0, synapse_02))
                
        if(dropout):
            layer_1 *= np.random.binomial([np.ones((len(X),hidden_neurons))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))

        layer_2 = sigmoid(np.dot(layer_1, synapse_12))

        # how much did we miss the target value?
        layer_2_error = y - layer_2

        if (j% 10000) == 0 and j > 5000:
            # if this 10k iteration's error is greater than the last iteration, break out
            if np.mean(np.abs(layer_2_error)) < last_mean_error:
                print ("delta after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error))) )
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                print ("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error )
                break
                
        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        layer_1_error = layer_2_delta.dot(synapse_12.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
        
        synapse_12_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_02_weight_update = (layer_0.T.dot(layer_1_delta))
        
        if(j > 0):
            synapse_02_direction_count += np.abs(((synapse_02_weight_update > 0)+0) - ((prev_synapse_02_weight_update > 0) + 0))
            synapse_12_direction_count += np.abs(((synapse_12_weight_update > 0)+0) - ((prev_synapse_12_weight_update > 0) + 0))        
        
        synapse_12 += alpha * synapse_12_weight_update
        synapse_02 += alpha * synapse_02_weight_update
        
        prev_synapse_02_weight_update = synapse_02_weight_update
        prev_synapse_12_weight_update = synapse_12_weight_update

    now = datetime.datetime.now()

    # persist synapses
    synapse = {'synapse01': synapse_02.tolist(), 'synapse11': synapse_12.tolist(),
               'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'words2': words2,
               'classes2': classes2
              }
    synapse_file = "synapses_category.json"

    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True)
    print ("saved synapses to:", synapse_file)
    
    
    
X = np.array(training2)
y = np.array(output2)

start_time = time.time()

train_reminder(X, y, hidden_neurons=20, alpha=0.1, epochs=100000, dropout=False, dropout_percent=0.2)

elapsed_time = time.time() - start_time
print ("processing time:", elapsed_time, "seconds")



# probability threshold
ERROR_THRESHOLD = 0.2
# load our calculated synapse values
synapse_file = 'synapses_category.json' 
with open(synapse_file) as data_file: 
    synapse = json.load(data_file) 
    synapse_02 = np.asarray(synapse['synapse01']) 
    synapse_12 = np.asarray(synapse['synapse11'])

def classify_reminder(sentence, show_details=False):
    results = think_reminder(sentence, show_details)
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD ] 
    results.sort(key=lambda x: x[1], reverse=True) 
    return_results =[[classes2[r[0]],r[1]] for r in results]
    # print ("%s \n classification: %s" % (sentence, return_results))
    # if return_results:
    #     print(return_results[0][0])
    #     pass    
    return return_results

def model2(sentences):
    reminder_list = []
    for sentence in sentences:        
        sentence = str(sentence)
        result = classify_reminder(sentence)
        if result:
            if result[0][0] == "reminder" and float(result[0][1])*100 > 99:
                reminder_list.append([sentence,result[0][0],result[0][1]])                                              
        pass
    return reminder_list     

##classify("sudo make me a sandwich")
##classify("how are you today?")
##classify("talk to you tomorrow")
##classify("who are you?")
##classify("make me some lunch")
##classify("how was your lunch today?")
##print()
# classify("I'm good how are you", show_details=True)

if __name__ == '__main__':
    text = "A:Hi Vishal. This is Vilas here. B:Hi Vilas. How are you doing. A:I am doing good. Thank you. B:Good. Can we start the meeting. A:Sure. \
    B:Okay. Let's go over today's progress. A:Sure. Today I worked on bug HX-1345. I have fixed the issue and the Pull Request has been generated. \
    Vishal request you to review the request and provide your feedback. B:Ok. I will work on that. A:Also, the major issue which I was working HX-1489 has finally been fixed. \
    B:Awesome!. That was a really big issue. Glad you could resolve it. I would like to know more on how you fixed this Issue. \
    Can we have a call at 6PM today to go through the solution. A:Defintely. I will keep my Calender free for 6PM. B:Okay. I worked on Issue HX-3636. The Issue has been resolved. \
    I will check in the code soon. A:Vishal. I would like to Inform you that I will be on Leave for the next two days. B:Oh Ok!. May I know the reason for the leave. \
    I guess I was not informed earlier. A:Yes. I am travelling to my home town. B:Ok. Thanks for the update. Lets meet tomorrow same time. A:Request you to also follow up on the access Issue we have been facing"
    # text = "how was your lunch today?"            
    main(text)
