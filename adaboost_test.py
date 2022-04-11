import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split

from adaboost import Adaboost

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def extractFeatures():
    f1 = open("lab2data.txt", "r",encoding='UTF-8')
    combinedlist =[]
    # list =[["None"]*7]*10
    # print(list)
    i = 0
    for line1 in f1:
        list = [None]*7
        l = re.split("[|,.\n \s]",line1)
        wordlen = []
        flag = 0
        flag0=0
        flag1=0
        flag2=0
        flag3=0
        flag4=0
        flag5=0
        for word in l:
            if flag == 0:
                if l[0] == "en":
                    list[6] = "Yes"
                    flag = 1
                elif l[0] == "nl":
                    list[6] = "No"
                    flag = 1
            else:
                #articles
                if flag0==0:
                    if word == "een" or word == "en" or word == "het" or word == "de":
                        flag0 = 1
                        list[0] = "No"
                    else:
                        list[0] = "Yes"
                #pronouns
                if flag1 == 0:
                    if word == "ik" or word =="u"or word == "jij" or word == "je" or word == "jullie" or word == "jou" or word == "gij" or word =="hij" or word=="zij" or word =="wij" or word =="ons":
                        # print(word)
                        flag1 =1
                        list[1] = "No"
                    else:
                        list[1] = "Yes"

                #count of i's j's k's
                if flag2 == 0:
                    if word.count("i")>3 or word.count("j")>3 or word.count("k")>3:
                        # print(word)
                        list[2] = "No"
                    else:
                        list[2] = "Yes"

                #check number of duplicates together
                if flag3 == 0:
                    if checkDuplicates(word) == True:
                        # print(word)
                        flag3 = 1
                        list[3] = "No"
                    else:
                        list[3] ="Yes"

                #check english part of speech
                if flag4 == 0:
                    if word =="he" or word =="she" or word == "the" or word == "him" or word == "her" or word == "they":
                        flag4 = 0
                        list[4] = "Yes"
                    else:
                        list[4] = "No"

                #length of the words
                    if flag5 == 0:
                        wordlen.append(word)
                        if calavg(wordlen):
                            list[5] ="Yes"
                        else:
                            list[5] = "No"
                            flag5 = 1
        # print(list)
        combinedlist.append(list)
        i = i+1
        # print(combinedlist)

    numpyarray = np.array(combinedlist)
    nlendataset = pd.DataFrame(numpyarray)
    # print(nlendataset)

    training_data = train_test_split(nlendataset)[0]
    #print(training_data)
    testing_data = train_test_split(nlendataset)[1]
    #print(testing_data)
    # return training_data
    return nlendataset


def calavg(wordlen):
    length =0
    for word in wordlen:
        length = length +len(word)
    avg = length/len(wordlen)
    if(avg>5):
        return False
    else:
        return True


def checkDuplicates(word):
    count = 1
    if len(word) > 1:
        for i in range(1, len(word)):
            if word[i - 1] == word[i]:
                count += 1
        if count > 1:
            return True
        else:
            return False


data = extractFeatures()
X = data.iloc[:,0:6]
y = data.iloc[:,6]


for i in range(len(y)):
    if y[i]=='Yes':
        y[i] = 1
    else:
        y[i] = -1

# print(X)
# print(y)

# y[y == 0] = -1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=4)
# print(X_train)

# Adaboost classification with 5 weak classifiers
clf = Adaboost(n_clf=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("y_pred")
print(y_pred)

acc = accuracy(y_test, y_pred)
print ("Accuracy:", acc*100)



