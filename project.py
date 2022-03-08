import numpy as np
import matplotlib.pyplot as plt
import csv
import os, psutil
import time
from numpy.random.mtrand import rand

#-----CLASSES BEGIN-------
#Node - a node in a binary tree. Contains: data=data, key=the key for this node (column in features), isString=True if the key column is a string type
#colHeaders=the column headers for this node's features
class Node:
    def __init__(self, data, key, name, isString, numSamples, colHeaders):
        self.left = None
        self.right = None
        self.data = data
        self.key = key
        self.name = name
        self.numSamples = numSamples
        self.isString = isString
        self.colHeaders = colHeaders

    #prints the tree, indenting at each level, indicates the node.data and node.key
    def printTree(self, colHeaders, level=0, leftRight = ""):
        key = " prediction ="
        if(self.key is not None):
            key = colHeaders[self.key]+" ("+repr(self.numSamples)+")"
        ret = "  "*level+leftRight+key+" "+repr(self.data)+"\n"
        if(self.left is not None):
            ret += self.left.printTree(colHeaders, level+1, "left ")
        if(self.right is not None):
            ret += self.right.printTree(colHeaders, level+1, "right ")
        return ret
#-----CLASSES END---------

#-----FUNCTIONS BEGIN-----
#Reads in the data from the formatted training and testing sets. Deletes any duplicate data in testing.
def getData():
    #Training Data
    file = open('train.csv', 'r')
    reader = csv.reader(file, delimiter=',')
    items = [row for row in reader]
    trainFeatures = np.asarray(items)
    trainFeatures = np.delete(trainFeatures, 0, axis=1) #remove the ids
    columnHeaders = trainFeatures[0] #get column headerss
    columnHeaders = np.delete(columnHeaders, -1) #remove the price header
    trainFeatures = np.delete(trainFeatures, 0, axis=0) #remove column headers
    trainTargets = trainFeatures[:, -1].astype(int)
    trainFeatures = np.delete(trainFeatures, -1, axis=1)
    trainFeatures = np.delete(trainFeatures, 8, axis=1)#remove the 8th column because it's initial information gain is terrible in training
    file.close()

    trainFeatures = stringsToInts(trainFeatures)
    #printInitialInformationGain(trainFeatures, trainTargets, columnHeaders)

    #Testing Data
    file = open('test.csv', 'r')
    reader = csv.reader(file, delimiter=',')
    items = [row for row in reader]
    testFeatures = np.asarray(items)
    file.close()

    testFeatures = stringsToInts(testFeatures)
    testFeatures = np.delete(testFeatures, 0, axis=0) #remove column headers
    testFeatures = np.delete(testFeatures, 0, axis=1) #remove the ids
    testFeatures = np.delete(testFeatures, 8, axis=1)#remove the 8th column because it's initial information gain is terrible in training
    
    #split training and trainingTesting to check for accuracy
    randomizedTrainFeatures, randomizedTrainTargets = unison_shuffled_copies(trainFeatures, trainTargets)
    k = np.floor(len(randomizedTrainFeatures)/4).astype(int)
    splitTrainTestingFeatures = randomizedTrainFeatures[:k]
    splitTrainTestingTargets = randomizedTrainTargets[:k]
    splitTrainFeatures = randomizedTrainFeatures[k:]
    splitTrainTargets = randomizedTrainTargets[k:]

    return splitTrainFeatures, splitTrainTargets, splitTrainTestingFeatures, splitTrainTestingTargets, testFeatures, columnHeaders
    
#convert string numbers to ints in an array
#data=the data array to convert
def stringsToInts(data):
    newData = np.zeros([len(data),len(data[0])], dtype='O')
    for i in range(0,len(data)):
        for j in range(0,len(data[i])):
            if(data[i][j].isnumeric()):
                newData[i][j] = data[i][j].astype(int)
            else:
                newData[i][j] = data[i][j]

    return newData

#get the entropy of x feature
def entropy(x, epsilon=1e-8):
    _, counts = np.unique(x, return_counts=True)
    p = counts / len(x)
    numClasses = np.count_nonzero(p)

    #all zero count, return 0
    if(numClasses <= 1):
        return 0
        
    ent = 0
    #between 0-1 for k classes
    for i in p:
        ent -= i * np.log2(i+epsilon)#/np.log(numClasses) #/ by np.log(numClasses) to create log base numClasses

    return ent

#determines the information gain from a specific split
#x=feature column, t=target values
def informationGain(x, t):
    totalEntropy = entropy(t)
    #Calculate the values and the corresponding counts for the split attribute 
    x, t = deleteNA(x, t)
    vals,counts= np.unique(x,return_counts=True)

    #no vals after deleting NA
    if(len(vals) == 0):
        return None, None
    
    #Calculate the weighted entropy and the best value
    weightedEntropy = 0
    bestVal = vals[0]
    bestEntropy = 0
    for i in range(len(vals)):
        currEntropy = entropy(t[x == vals[i]])
        weightedEntropy += counts[i]/np.sum(counts)*currEntropy
        #current entropy better, indicate
        if(currEntropy > bestEntropy):
            bestEntropy = currEntropy
            bestVal = vals[i]
        
    #Calculate the information gain
    info = totalEntropy - weightedEntropy#info = 1 - weightedEntropy
    return info, bestVal

#calculates the best split column from a set of data (highest information gain == best split)
#data=data array to consider, t=target values
def bestSplit(data, t):
    bestInfo, bestSplitVal = informationGain(data[:,0],t)

    #no values found, assumed all NA in current split
    if(bestInfo == None):
        return None, None, None

    bestSplitCol = 0
    for i in range(1,len(data[0])):
        currInfo, currBestVal = informationGain(data[:,i],t)
        #found better information gain on this split
        if(currInfo != None and currInfo > bestInfo):
            bestInfo = currInfo
            bestSplitCol = i
            bestSplitVal = currBestVal

    return bestInfo, bestSplitCol, bestSplitVal

#split the data by the given split params
def split(data, targets, splitCol, splitVal):
    left = None
    right = None
    #split left on equivelant or less, split right on not equal or greater value
    #(left is for lesser values, right is for greater ones)
    if(isString(data[:,splitCol])):
        leftCond = data[:,splitCol] == splitVal
        rightCond = data[:,splitCol] != splitVal
    else:
        comparisonData = data[:,splitCol]
        for i in range(0,len(comparisonData)):
            if(comparisonData[i] == "NA"):
                comparisonData[i] = 999999
        leftCond = comparisonData <= splitVal
        rightCond = comparisonData > splitVal
    
    left = data[leftCond]
    leftTargets = targets[leftCond]
    right = data[rightCond]
    rightTargets = targets[rightCond]

    return left, leftTargets, right, rightTargets

#makes a prediction given the current targets.
def predict(targets):
    #String will get the commonly occuring value, real numbers will give the mean
    if(isString(targets)):
        vals, counts = np.unique(targets, return_counts=True)
        return vals[np.argmax(counts)]
    else:
        return np.mean(targets)

#makes a prediction given the created tree and selected sample to predict
def makePrediction(node, sample, originalColHeaders, parent=None):
    #reached predicted target or this node is None, return this (or parent) data
    if(node is None):
        return parent.data
    if(node.left is None and node.right is None):
        return node.data

    #get index of node.name in originalColHeaders to find the actual data index
    key = np.where(originalColHeaders == node.name)
    
    #check for equivelant on string or <= on real number. recurse through each node
    if(node.isString):
        if(sample[key] == node.data):
            return makePrediction(node.left, sample, originalColHeaders, node)
        else:
            return makePrediction(node.right, sample, originalColHeaders, node)
    else:
        if(sample[key] == "NA"):
            return makePrediction(node.right, sample, originalColHeaders, node)
        elif(sample[key] <= node.data):
            return makePrediction(node.left, sample, originalColHeaders, node)
        else:
            return makePrediction(node.right, sample, originalColHeaders, node)

#calculates the predictions of an array of features given the built tree
def predictTargets(tree, features, originalColHeaders):
    predictions = np.zeros(len(features))
    for i in range(0, len(features)):
        predictions[i] = makePrediction(tree, features[i], originalColHeaders)
    return predictions

#returns the predictions given a Random Forest, testing samples (features), and the column headers
def getForestPredictions(forest, features, originalColHeaders):
    predictions = np.zeros((len(features), len(forest)))
    i = 0
    #make the predictions
    for tree in forest:
        try:
            predictions[:,i] = predictTargets(tree, features, originalColHeaders)
        except:
            print("EXCEPTION OCCURED Printing tree with exception")
            print(tree.printTree(tree.colHeaders))
            return 0
        i += 1

    #collect the most frequent predictions and use those
    finalPredictions = np.zeros(len(features))
    for i in range(0, len(predictions)):
        #take the average for all of the tree predictions
        finalPredictions[i] = np.average(predictions[i], axis = 0)

    return finalPredictions

#calculates the percentage accuracy of predictions given actual targets
def accuracy(predictions, targets):
    sum = 0
    for i in range(0, len(predictions)):
        sum += np.abs(predictions[i] - targets[i])/targets[i]
    sum = sum/len(predictions)
    return 100-sum*100

#build the decision tree through recursion
#data=the current samples, #targets=the current targets of the samples, #colHeaders=the column headers for each index of a sample
#minSamplesRestriction= the minimum samples to restrict, #minInformationGain=the minimum amount of information gain to consider
#currDepth=The current depth in this recursion, #nullPrediction=the prediction when there is no data in the split
def buildTree(data, targets, colHeaders, maxDepth = None, minSamplesRestriction = None, minInformationGain = 1e-12, currDepth = 0, nullPredication = None):
    if(printDuringModelBuild):
        print()
        print("Building tree current depth = "+str(currDepth)+", current num samples = "+str(len(data))+", current num features = "+str(len(colHeaders)))
    
    # Check for depth conditions
    maxDepthReached = False
    if(maxDepth != None):
        if(currDepth >= maxDepth):
            maxDepthReached = True
    else:
        maxDepthReached = True

    # Check for sample conditions
    minSamplesExists = False
    if(minSamplesRestriction == None):
        minSamplesExists = True
    elif len(data) > minSamplesRestriction:
            minSamplesExists = True

    # Check for for conditions
    if((not maxDepthReached) and minSamplesExists):
        bestInfo, bestSplitCol, bestSplitVal = bestSplit(data, targets)
        bestSplitNameStr = ""
        if(bestInfo is not None):
            bestSplitNameStr = ", bestSplitName = "+str(colHeaders[bestSplitCol])
        if(printDuringModelBuild):
            print("bestInfo = "+str(bestInfo)+", bestSplitCol = "+str(bestSplitCol)+bestSplitNameStr+", bestSplitVal = "+str(bestSplitVal))

        # If information gain is found and the minimum info gain condition is met
        if bestInfo is not None and bestInfo >= minInformationGain:

            currDepth += 1

            left, leftTargets, right, rightTargets = split(data, targets, bestSplitCol, bestSplitVal)
            bestSplitName = colHeaders[bestSplitCol]

            # Instantiate sub-tree
            subtree = Node(bestSplitVal, bestSplitCol, bestSplitName, isString(data[:,bestSplitCol]), len(data), colHeaders)

            # Find answers (recursion)
            leftNode = buildTree(left, leftTargets, colHeaders, maxDepth, minSamplesRestriction, minInformationGain, currDepth)

            rightNode = Node(predict(targets), None, None, None, 0, None)
            if(len(right) != 0):
                rightNode = buildTree(right, rightTargets, colHeaders, maxDepth, minSamplesRestriction, minInformationGain, currDepth)

            subtree.left = leftNode
            subtree.right = rightNode
        # no split or minInformationGain reached, make this a leaf node
        else:
            pred = Node(predict(targets), None, None, None, 0, None)
            return pred
    # make this a leaf node or a None node if the data length is 0
    else:
        if(maxDepthReached and printDuringModelBuild):
            print("Max depth reached, force leaf node.")
        elif(not minSamplesExists and printDuringModelBuild):
            print("Minimum samples condition not satisfied, force leaf node.")

        pred = nullPredication
        #length of samples is 0, should never happen
        if(len(data) > 0):
            pred = Node(predict(targets), None, None, None, 0, None)
            
        return pred

    return subtree

#creates a random forest by creating numTrees number of decision trees
def randomForest(features, targets, colHeaders, numBatches = 3, numTrees = 1000, maxDepth = None, minSamplesRestriction = None, minInformationGain = 1e-12, currDepth = 0, nullPredication = None):
    print("Creating forest with numBatches = "+str(numBatches)+", numTrees = "+str(numTrees))
    #randomly shuffle the data
    features, targets = unison_shuffled_copies(features, targets)
    k = np.floor(len(features[0])/numBatches).astype(int)
    forest = []

    #build the forest
    for i in range(0,numTrees):
        if(printDuringModelBuild or printDuringModelBuildMinimal):
            print("Current Tree = "+str(i))
        randomColumns = np.random.randint(0, len(features[0]), size=k)
        currFeatures = features[:, randomColumns]
        currColHeaders = colHeaders[randomColumns]
        currTree = buildTree(currFeatures, targets, currColHeaders, maxDepth, minSamplesRestriction, minInformationGain, currDepth, nullPredication)
        forest.append(currTree)
    
    return forest

#--Plotting/information functions begin--
#combines like values into labels and the number of occurences of that label
def getLabels(data):
    labels = []
    labelNums = []
    for i in range(0, len(data)):
        #in labels, increment
        if(data[i] in labels):
            j = labels.index(data[i])
            labelNums[j] += 1
        #not in labels, add to labels
        else:
            labels.append(data[i])
            labelNums.append(1)
    return labels, labelNums

#plots a bar chart against the number of occurences
def plotBarChart(data, dataCol, yLabel):
    fig, ax = plt.subplots()
    labels, labelNums = getLabels(data[:,dataCol])
    x = np.arange(len(labels))
    ax.set_ylabel(yLabel)
    ax.set_title('Correlation between '+yLabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.bar(labels,labelNums)
    fig.tight_layout()
    plt.show()

#plots any integers against the home price
def plotIntegers(data, targets, dataCol, yLabel):
    x = targets
    #convert to ints, remove from this consideration if NA
    y = data[:,dataCol]
    y, x = deleteNA(y, x)
    y = y.astype(int)

    plt.scatter(x, y, color="red", marker=".")
    bins = np.linspace(y.min(), y.max(), 20)
    plt.grid(True, axis='y')
    plt.yticks(bins)
    plt.margins(y=0.02)
    plt.tight_layout()
    plt.xlabel("Home Price")
    plt.ylabel(yLabel)
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.show()

#prints the initial information gain for all features
def printInitialInformationGain(features, targets, colHeaders):
    bestInfoPrint = 0
    bestInfo = 0
    for i in range(0, 78):
        currBestInfo, currBestInfoVal = informationGain(features[:,i], targets)
        pr = str(colHeaders[i])+" ("+str(i)+") = "+str(currBestInfo)+ ". Best Info Value="+str(currBestInfoVal)
        print(pr)
        if(currBestInfo > bestInfo):
            bestInfoPrint = pr
            bestInfo = currBestInfo

    print("Best Initial Information Gain: "+bestInfoPrint)
#--Plotting/information functions end--

#--Helper functions begin--
#check for NA values in arg1 and removes the indices from arg1 and arg2
def deleteNA(arg1, arg2=None):
    indices = []
    for i in range(0,len(arg1)):
        if(arg1[i] == "NA"):
            indices.append(i)
    
    arg1 = np.delete(arg1, indices, axis=0)
    if(arg2 is not None):
        arg2 = np.delete(arg2, indices, axis=0)

    return arg1, arg2

#determines if an array of same-type values is a string
def isString(x):
    #default is not string
    type = False
    for i in x:
        if(i == "NA"):
            continue
        if(isinstance(i, int)):
            type = False
        else:
            type = i.dtype.type is np.str_
        break
    return type

#taken from https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
#shuffles two arrays in unison
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
#--Helper functions end--
#-----FUNCTIONS END-----

#------Begin main-------
#collect data
trainFeatures, trainTargets, trainTestingFeatures, trainTestingTargets, testFeatures, colHeaders = getData()

#user wants to plot
if(input("Press the Enter key to begin training, 'p' to plot data.") != ""):
    type = input("'bar' plot or 'int' plot?")
    col = input("index of column to plot:")
    col = int(col)
    if(type == 'int'):
        plotIntegers(trainFeatures, trainTargets, col, colHeaders[col])
    elif(type == 'bar'):
        plotBarChart(trainFeatures, col, colHeaders[col])
else:
    epsilon = 1e-8

    #---Begin Training---
    N = len(trainFeatures)
    M = len(trainFeatures[0])

    numTrees = 500
    maxDepth = 25
    numBatches = 4
    minSamplesRestriction = 35
    minInformationGain = 1e-8
    printDuringModelBuild = False
    printDuringModelBuildMinimal = False
    timeStart = time.process_time()
    temp = input("Include print statements during tree building and predictions? y for yes, m for minimal, enter for no.")
    if(temp == "y"):
        printDuringModelBuild = True
    elif(temp == "m"):
        printDuringModelBuildMinimal = True

    print()
    print("Begin Training with N="+str(N)+", M="+str(M))
    print("Begin building forest: numTrees = "+str(numTrees)+", maxDepth = "+str(maxDepth)+", minSamplesRestriction = "+str(minSamplesRestriction)+" minInformationGain = "+str(minInformationGain))
    forest = randomForest(trainFeatures, trainTargets, colHeaders, numBatches=numBatches, numTrees=numTrees, maxDepth=maxDepth, minSamplesRestriction=minSamplesRestriction, minInformationGain=minInformationGain)
    timeEnd = time.process_time()
    print("Forest built in "+str((timeEnd-timeStart)/60)+" minutes, memory used = "+str(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)+"MB")
    #----End Training----

    print()
    #human readable tree - uncomment to print the forest
    #print(forest[0].printTree(forest[0].colHeaders))
    #for tree in forest:
    #    print(tree.colHeaders)
    #    print(tree.printTree(tree.colHeaders))
    #print(forest[0].printTree(colHeaders))

    #---Begin Testing---

    predictions = getForestPredictions(forest, trainTestingFeatures, colHeaders)
    print("Testing Accuracy = "+str(accuracy(predictions, trainTestingTargets)))
    #print(forest[0].printTree(forest[0].colHeaders))

    testPredictions = getForestPredictions(forest, testFeatures, colHeaders)
    #create csv with the testing predictions for kaggle submission
    with open('test-predictions.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "SalePrice"])
        x = 1461
        for i in range(0, len(testPredictions)):
            writer.writerow([x, testPredictions[i]])
            x += 1

    #----End Testing----

    #add to logs:
    with open("logs.txt", "a") as myfile:
        myfile.write("N="+str(N)+", M="+str(M)+", numTrees = "+str(numTrees)+" maxDepth = "+str(maxDepth)+", numBatches = "+str(numBatches)+", minSamplesRestriction = "+str(minSamplesRestriction)+" minInformationGain = "+str(minInformationGain))
        myfile.write("\nForest built in "+str((timeEnd-timeStart)/60)+" minutes, memory used = "+str(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)+"MB\n")
        myfile.write("Testing Accuracy (N/4 samples) = "+str(accuracy(predictions, trainTestingTargets))+"\n\n")
    
    #add forest structure to file for debugging:
    with open("tree-logs.txt", "w") as myfile:
        i = 0
        for tree in forest:
            myfile.write("Tree number: "+str(i)+" with headers:\n")
            myfile.write(str(tree.colHeaders)+"\n")
            myfile.write(tree.printTree(tree.colHeaders))
            myfile.write("\n")
            i += 1