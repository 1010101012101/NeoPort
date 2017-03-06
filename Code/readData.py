import re
import random
from collections import defaultdict
from string import ascii_lowercase


def getData(trainingPoints=700,validPoints=300):
    data=[]
    data_words=[]

    for line in open("../Data/dataset.csv"):
        words=re.split("\W+",line)
        words=words[:3]
        char_words=[list(word.lower()) for word in words]

        data.append(char_words)
        data_words.append(words)
        print words
        print char_words


    wids=defaultdict(lambda: 0)
    wids["STOP"]=0
    wids["SEPARATOR"]=1
    STOP=0
    SEPARATOR=1

    for c in ascii_lowercase:
        wids[c]=len(wids)

    data=[[[wids[character] for character in elem] for elem in x] for x in data]
    dataInputs=[x[1]+[SEPARATOR,]+x[2]+[STOP,] for x in data]
    dataOutputs=[x[0]+[STOP,] for x in data]
   
    trainInputs=dataInputs[:trainingPoints]
    trainOutputs=dataOutputs[:trainingPoints]
    validInputs=dataInputs[trainingPoints:trainingPoints+validPoints]
    validOutputs=dataOutputs[trainingPoints:trainingPoints+validPoints]
    testInputs=dataInputs[trainingPoints+validPoints:]
    testOutputs=dataOutputs[trainingPoints+validPoints:]

    #print wids["STOP"]
    return trainInputs,trainOutputs,validInputs,validOutputs,testInputs,testOutputs,wids
    #return dataInputs,dataOutputs,wids

def getDataDisjoint(trainingPoints=500,validPoints=200):
    data=[]
    data_words=[]

    for line in open("../Data/dataset.csv"):
        words=re.split("\W+",line)
        words=words[:3]
        char_words=[list(word.lower()) for word in words]

        data.append(char_words)
        data_words.append(words)
        print words
        print char_words


    wids=defaultdict(lambda: 0)
    wids["STOP"]=0
    wids["SEPARATOR"]=1
    STOP=0
    SEPARATOR=1

    for c in ascii_lowercase:
        wids[c]=len(wids)

    data=[[[wids[character] for character in elem] for elem in x] for x in data]
    dataInputs=[x[1]+[SEPARATOR,]+x[2]+[STOP,] for x in data]
    dataOutputs=[x[0]+[STOP,] for x in data]
   
    print len(dataInputs)
    trainInputs=dataInputs[:trainingPoints]
    trainOutputs=dataOutputs[:trainingPoints]
    trainVocab=set()

    print len(trainInputs)
    for elem in data_words[:trainingPoints]:
        trainVocab.add(elem[1])
        trainVocab.add(elem[2])
    
    #print trainVocab

    validInputs=[]
    validOutputs=[]
    validVocab=set()

    for i,elem in enumerate(dataInputs):
        parentWord1=data_words[i][1]
        parentWord2=data_words[i][2]

        if (parentWord1 in trainVocab) or (parentWord2 in trainVocab):
            continue
        
        validInputs.append(dataInputs[i])
        validOutputs.append(dataOutputs[i])
        validVocab.add(parentWord1)
        validVocab.add(parentWord2)
        
        if len(validInputs)>=validPoints:
            break

    print len(validInputs)    
    #print validVocab

    testInputs=[]
    testOutputs=[]

    for i,elem in enumerate(dataInputs):
        parentWord1=data_words[i][1]
        parentWord2=data_words[i][2]

        if (parentWord1 in trainVocab) or (parentWord2 in trainVocab):
            continue
        
        if (parentWord1 in validVocab) or (parentWord2 in validVocab):
            continue
        
        testInputs.append(dataInputs[i])
        testOutputs.append(dataOutputs[i])
    
    print len(testInputs)
    #validInputs=dataInputs[trainingPoints:trainingPoints+validPoints]
    #validOutputs=dataOutputs[trainingPoints:trainingPoints+validPoints]
    #testInputs=dataInputs[trainingPoints+validPoints:]
    #testOutputs=dataOutputs[trainingPoints+validPoints:]

    #print wids["STOP"]
    return trainInputs,trainOutputs,validInputs,validOutputs,testInputs,testOutputs,wids
    #return dataInputs,dataOutputs,wids

def getDataKnightHoldOut(trainingPoints=1000):
    data=[]
    data_words=[]

    knightData=[]
    knightData_words=[]

    for line in open("../Data/dataset.csv"):
        words=re.split("\W+",line)
        knightFlag=words[3]
        words=words[:3]
        char_words=[list(word.lower()) for word in words]
        
        if knightFlag=="knight":
            knightData.append(char_words)
            knightData_words=(words)
            print words
            print char_words
        else:
            data.append(char_words)
            data_words.append(words)
            print words
            print char_words
    
    print "Knight Words",len(knightData)
    print "Other Words",len(data)

    wids=defaultdict(lambda: 0)
    wids["STOP"]=0
    wids["SEPARATOR"]=1
    STOP=0
    SEPARATOR=1

    for c in ascii_lowercase:
        wids[c]=len(wids)

    data=[[[wids[character] for character in elem] for elem in x] for x in data]
    dataInputs=[x[1]+[SEPARATOR,]+x[2]+[STOP,] for x in data]
    dataOutputs=[x[0]+[STOP,] for x in data]

    knightData=[[[wids[character] for character in elem] for elem in x] for x in knightData]
    knightDataInputs=[x[1]+[SEPARATOR,]+x[2]+[STOP,] for x in knightData]
    knightDataOutputs=[x[0]+[STOP,] for x in knightData]
 

    trainInputs=dataInputs[:trainingPoints]
    trainOutputs=dataOutputs[:trainingPoints]
    validInputs=dataInputs[trainingPoints:]
    validOutputs=dataOutputs[trainingPoints:]
    testInputs=knightDataInputs
    testOutputs=knightDataOutputs
    
    return trainInputs,trainOutputs,validInputs,validOutputs,testInputs,testOutputs,wids



def reverseDictionary(dic):
    reverseDic={}
    for x in dic:
        reverseDic[dic[x]]=x
    return reverseDic

if __name__=="__main__":
    trainInputs,trainOutputs,validInputs,validOutputs,testInputs,testOutputs,wids=getDataDisjoint()
