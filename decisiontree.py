import numpy as np
import pandas as pd
import re
import pickle
import sys

def entropy(input_col):
    """
        Calculates Entropy which measures the impurity.
        :param input_col: attribute
        :return entropy : entropy of that attribute
    """
    elements, uniqueCount = np.unique(str(output_col).split(),return_counts=True)
    entropy = 0

    for i in range(len(elements)):
        probablity = uniqueCount[i] / np.sum(uniqueCount)
        entropy = entropy + (-probablity) * np.log2(probablity)
        return entropy

def InformationGain(data,attribute,output_col):
    """
    Calculates the information gain for each attribute
    :param data: training data
    :param attribute: feature for which information gain is calculated
    :param output_col: output column
    :return: Information Gain
    """
    totalEntropy = entropy(data[output_col])
    elementsAttr, uniqueCountAttr = np.unique(data[attribute], return_counts=True)

    Weighted_Entropy =0
    for i in range(len(elementsAttr)):
        Weighted_Entropy = Weighted_Entropy + (uniqueCountAttr[i]/np.sum(uniqueCountAttr))*entropy(data.where(data[attribute]==elementsAttr[i])[output_col])

    InfoGain = totalEntropy-Weighted_Entropy
    return InfoGain


def decision_tree_learning(subset_data, original_data, features, target_attribute, previous_node=None):
    """
    Generates decision tree from training dataset
    :param subset_data: subset of the training dataset
    :param original_data: training dataset
    :param features: feature columns
    :param target_attribute: output columns
    :param previous_node: previous node in the decision tree
    :return: desion tree
    """

    if len(np.unique(data[target_attribute]))<=1:
        return np.unique(data[target_attribute])[0]
    elif len(data) == 0:
        return np.unique(givendata[target_attribute])[np.argmax(np.unique(givendata[target_attribute], return_counts=True)[1])]
    elif len(features) == 0:
        return previous_node
    else:
        previous_node = np.unique(data[target_attribute])[np.argmax(np.unique(data[target_attribute], return_counts=True)[1])]

        attr_items = [InformationGain(data, feature, target_attribute) for feature in features]

        bestfeatureindex = np.argmax(attr_items)
        bestfeature = features[bestfeatureindex]
        tree = {bestfeature: {}}

        newfeatures=[]
        for feat in features:
            if feat !=bestfeature:
                newfeatures.append(feat)

        features = newfeatures

        for value in np.unique(data[bestfeature]):
            value = value
            sub_data = data.where(data[bestfeature] == value).dropna()
            subtree = decision_tree_learning(sub_data, data, features, target_attribute, previous_node)
            tree[bestfeature][value] = subtree
        return (tree)

def predict(dictdata,tree):
    for key in list(dictdata.keys()):
        if key in list(tree.keys()):
            output = tree[key][dictdata[key]]

            if isinstance(output, dict):
                return predict(dictdata,output)
            else:
                return output

def extractFeaturesTrain(trainfile):
    """
    Generating features based on the training dataset
    :param trainfile: training dataset
    :return: returns decision tree 
    """
    f1 = open(trainfile, "r",encoding='UTF-8')
    combinedlist =[]
    for line1 in f1:
        list = [None]*7
        l = re.split("[|,.\n \s]",line1)
        all_words = []
        flag = flag0 = flag1 = flag2 = flag3 = flag4 = flag5 = 0
        for word in l:
            if flag == 0:
                if l[0] == "en":
                    list[6] = "en"
                    flag = 1
                elif l[0] == "nl":
                    list[6] = "nl"
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
                        all_words.append(word)
                        if check_average_word_length(all_words):
                            list[5] ="Yes"
                        else:
                            list[5] = "No"
                            flag5 = 1
        combinedlist.append(list)

    numpyarray = np.array(combinedlist)
    nlendataset = pd.DataFrame(numpyarray)
    training_data = nlendataset
    tree = decision_tree_learning(training_data,training_data,[i for i in range(training_data.shape[1]-1)],training_data.shape[1]-1)
    return tree


def extractFeaturesTest(testfile):
    """
    Generate features based on the test sentences
    :param testfile: testfile
    :return: dataset with features
    """
    f1 = open(testfile, "r",encoding='UTF-8')
    combinedlist =[]
    for line1 in f1:
        list = [None]*7
        l = re.split("[|,.\n \s]",line1)
        all_words = []
        flag0 = flag1 = flag2 = flag3 = flag4 = flag5 = 0
        for word in l:
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
                    all_words.append(word)
                    if check_average_word_length(all_words):
                        list[5] ="Yes"
                    else:
                        list[5] = "No"
                        flag5 = 1
        combinedlist.append(list)
    numpyarray = np.array(combinedlist)
    dataset_with_features = pd.DataFrame(numpyarray)

    return dataset_with_features

def check_average_word_length(all_words):
    """
    Check if average word length in a sentence is greater than 5 or not
    :param all_words: all the words in a sentence
    :return: True, if the average word length is more than 5 else False
    """
    length =0
    for word in all_words:
        length = length +len(word)
    avg = length/len(all_words)
    if(avg>5):
        return False
    else:
        return True


def checkDuplicates(word):
    """
    check whether adjacent letters are duplicate
    :param word: word in question
    :return: True if conatins duplicates
    """
    count = 1
    if len(word) > 1:
        for i in range(1, len(word)):
            if word[i - 1] == word[i]:
                count += 1
        if count > 1:
            return True
        else:
            return False


if __name__ == '__main__':

    flag = False

    if sys.argv[1] == "train" and len(sys.argv)==5:
        examples = sys.argv[2]
        hypoOut = sys.argv[3]
        learningType = sys.argv[4]
        tree = extractFeaturesTrain(examples)

        with open(hypoOut, 'wb') as file:
            pickle.dump(tree, file)
        flag =True
    elif sys.argv[1] == "predict" and len(sys.argv)==4:
        hypothesis = sys.argv[2]
        file = sys.argv[3]
        testdata = extractFeaturesTest(file)
        datatodict = testdata.to_dict(orient="records")

        with open(hypothesis, 'rb') as file:
            tree1 = pickle.load(file)
        predicted = pd.DataFrame(columns=["predicted"])
        for i in range(len(testdata)):
            predicted.loc[i, "predicted"] = predict(datatodict[i], tree1)
        print(predicted)

