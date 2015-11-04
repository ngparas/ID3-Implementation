#!/usr/bin/python

# Handle imports
import sys
import csv
import random
import math

##### Initialization and Input Checks #####

# Get System Arguments
inputFile = str(sys.argv[1])
trainingSetSize = int(sys.argv[2])
numberOfTrials = int(sys.argv[3])
verbose = str(sys.argv[4])


# Read the data file
with open(inputFile) as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t')
    dataSet = list()
    for row in reader:
        for key in reader.fieldnames:
            if (row[key] == 'true' or row[key] == 'true '):
                row[key] = True
            else:
                row[key] = False
        dataSet.append(row)

print 'Input File ' +inputFile
print 'Training Set Size ' + str(trainingSetSize)
print 'Testing Set Size ' + str(len(dataSet) - trainingSetSize)
print 'Number of Trials ' + str(numberOfTrials)
print 'Verbose? ' + verbose

#Define Predictors for subsequent modeling
predictors = reader.fieldnames
predictors.remove('CLASS')

# Check input values, exit if failure
if (trainingSetSize == 0 or trainingSetSize >= len(dataSet)):
    print 'TrainingSetSize out of range, exiting'
    sys.exit(1)

if (numberOfTrials < 1):
    print 'NumberOfTrials must be a positive integer, exiting'
    sys.exit(1)

##### Define Functions and Class #####

# Define function to select random Indices
def trainingPartition(dataSize, numObs, seed):
    random.seed(seed)
    return random.sample(xrange(dataSize), numObs)

# Define function to select indices NOT in training set
def testingPartition(dataSize, trainingIndices):
    allInd = list(xrange(dataSize))
    for i in trainingIndices:
        allInd.remove(i)
    return allInd

# Retrieve data given a list of indices     
def getData(dataSet, indices):
    return [dataSet[i] for i in indices]

# Calculate the entropy (H) given a probability
def getEntropy(prob):
    if (prob != 0 and prob != 1):
        return (-prob * math.log(prob,2)) + (-(1-prob) * math.log((1-prob),2)) 
    else:
        # If all classes the same, return 0        
        return 0.0

# Calculate the Probability of the positive class
def getPosProb(data):
    if (len(data) == 0):
        return 0
    ct = float(0)
    for obs in data:
        if (obs['CLASS']):
            ct = ct + 1
    return (ct / len(data))

# Determine if all elements in a subset of data have the same class
def checkSame(data):
    comp = data[0]['CLASS']
    for obs in data:
        if (obs['CLASS'] != comp):
            return False
    return True

# Determine the most common class
def getMode(data, label):
    trueCount = 0
    falseCount = 0
    for obs in examples:
        if (obs['CLASS']):
            trueCount = trueCount + 1
        else:
            falseCount = falseCount + 1
    if (trueCount >= falseCount):
        return True
    else:
        return False

# split data on a single attribute
def splitData(data, attribute):
    lc = []
    rc = []
    for obs in data:
        if (obs[attribute]):
            lc.append(obs)
        else:
            rc.append(obs)
    return lc, rc

# ID3 Decision Tree Learning Algorithm (slide 13)
def DTL(examples, attributes, default):
    if (len(examples) == 0):
        # if there are no examples in the node, return the default value
        return default
    elif (checkSame(examples)):
        # if all of the examples in the node have the same class, return that class
        return examples[0]['CLASS']
    elif (len(attributes) == 0):
        # if there are no more attributes to split on, pick the most common class
        return getMode(examples, 'CLASS')        
    else:
        # make a new node
        tree = Node(examples, attributes, default)
        # pick the best attribute to split on
        tree.chooseAttribute()
        # execute the split and recurse down the subtrees
        tree.makeSplit()
        return tree

class Node:
    def __init__(self, data, attrChoices, default):
        self.attribute = None
        self.data = data
        self.attrChoices = attrChoices
        self.leftChild = None
        self.rightChild = None
        self.default = default
        self.posProb = getPosProb(self.data)
        self.entropy = getEntropy(self.posProb)

    # calculate the entropies and IG for each potential split
    def chooseAttribute(self):
        #set default split in case all result in the same IG
        self.attribute = self.attrChoices[0]
        newIG = 0.0
        numObs = float(len(self.data))
        for key in self.attrChoices:
            lc, rc = splitData(self.data, key)
            lcProb = getPosProb(lc)
            rcProb = getPosProb(rc)
            iG = self.entropy - ((len(lc)/numObs)*getEntropy(lcProb) + (len(rc)/numObs)*getEntropy(rcProb))
            if (iG > newIG):
                newIG = iG
                self.attribute = key
        
    def makeSplit(self):
        # copy possible attributes and then remove the split option
        attrPass = [None] * len(self.attrChoices)
        for i in range(len(self.attrChoices)):
            attrPass[i] = self.attrChoices[i]
        attrPass.remove(self.attribute)
        # split data on the splitting attribute
        lcPass, rcPass = splitData(self.data, self.attribute)
        #recurse, left direction is true, right is false
        self.leftChild = DTL(lcPass, attrPass, self.default)
        self.rightChild = DTL(rcPass, attrPass, self.default)
            
# Use Depth First travseral to print
def printTree(tree, default, depth):
    if (tree == default or tree == True or tree == False):
        print "  "*depth + str(tree)
    else:
        print "  "*depth + str(tree.attribute)
        printTree(tree.leftChild, default, depth + 1)
        printTree(tree.rightChild, default, depth + 1)

# Use Depth First Traversal to generate predictions
def predictTree(example, tree):
    if (tree == True or tree == False):
        return tree
    else:
        if (example[tree.attribute]):
            return predictTree(example, tree.leftChild)
        else:
            return predictTree(example, tree.rightChild)

# Append the prediction to the dictionaries of the data set (testing set)
def classifyObservations(dataSet, model, modelName):
    for i in dataSet:
        i[modelName] = predictTree(i, model)
    return dataSet

# Compare the predictions with the true classes
def percentAccuracy(dataSet, predictionName, actualName):
    numObs = float(len(dataSet))
    numCorrect = 0
    for i in dataSet:
        if (i[predictionName] == i[actualName]):
            numCorrect = numCorrect + 1
    return (numCorrect / numObs)

# Slightly more concise way to print data for verbose mode
def printData(dataSet):
    print dataSet[0].keys()
    for i in dataSet:
        print i.values()



##### Execute Trials #####

#initialize lists to store the accuracies for the final average performance
treeAccuracies = [0] * numberOfTrials
probAccuracies = [0] * numberOfTrials

for trial in range(numberOfTrials):
    #Partition the dataSet into training and testing sets
    trainingIndices = trainingPartition(len(dataSet),trainingSetSize,trial)
    trainingSet = getData(dataSet, trainingIndices)
    testingIndices = testingPartition(len(dataSet),trainingIndices)
    testingSet = getData(dataSet, testingIndices)

    #Estimate the expected prior probability
    priorProb = getPosProb(trainingSet)
    #Classify the examples by selecting the most likely class via Prior Prob
    if (priorProb >= 0.5):
        mostCommon = True
    else:
        mostCommon = False

    #Construct a decision tree (using mostCommon as the default Value)
    decisionTree = DTL(trainingSet, predictors, mostCommon)

    #Classify the examples in the testing set using the tree
    classifiedData = classifyObservations(testingSet, decisionTree, 'Tree Prediction')
    classifiedData = classifyObservations(classifiedData, mostCommon ,'Prior Prob Prediction')

    #Get the percent accuracies
    treePercAcc = percentAccuracy(classifiedData, 'Tree Prediction', 'CLASS')
    probPercAcc = percentAccuracy(classifiedData, 'Prior Prob Prediction', 'CLASS')
    
    #Store results for averages
    treeAccuracies[trial] = treePercAcc
    probAccuracies[trial] = probPercAcc

    #print the results to std out
    print "######## Trial Number: " + str(trial) + " ########\n"
    print "Binary Decision Tree Structure:\nNote: branches split left on true, right on false\n"
    printTree(decisionTree, mostCommon, 0)
    print "\n\nDecision Tree Classification Accuracy: " + str(treePercAcc * 100) + "%"
    print "Prior Probability Classification Accuracy: " + str(probPercAcc * 100) + "%"
    if (verbose == "1"):
        print "\n\n\n########"
        print "The Set of Training Examples"
        printData(trainingSet)
        print "\n\n\n########"
        print "The Set of Testing Examples With Classifications inserted as Keys"
        printData(classifiedData)

# Get the average performance across all trials
print "\n\nMean Tree Accuracy: " + str(float(sum(treeAccuracies)/len(treeAccuracies)))
print "Mean Prior Probability Accuracy: " + str(float(sum(probAccuracies)/len(probAccuracies))) + "\n\n"

