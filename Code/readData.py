import re
import random
from collections import defaultdict
from string import ascii_lowercase


def getData():
    data=[]
    data_words=[]

    for line in open("../Data/finalPorts.csv"):
        words=re.split("\W+",line)
        words=words[:3]
        char_words=[list(word.lower()) for word in words]

        data.append(char_words)
        data_words.append(words)
        print words
        print char_words

    random.shuffle(data)

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
    
    return dataInputs,dataOutputs,wids

def reverseDictionary(dic):
    reverseDic={}
    for x in dic:
        reverseDic[dic[x]]=x
    return reverseDic

if __name__=="__main__":
    getData()
