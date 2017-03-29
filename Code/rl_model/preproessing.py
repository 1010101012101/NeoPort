import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
import numpy as np
import csv
import configuration as config
from sklearn.preprocessing import LabelEncoder
import models
import environments
import pickle
from keras.callbacks import ModelCheckpoint
import sys
import random
import keras

sys.path.insert(0, sys.path[0]+'../')
print sys.path

import readData

class Preprocessing:


    def __init_(self):
        for i in range(10):
            trainInputs,trainOutputs,validInputs,validOutputs,testInputs,testOutputs,wids=getData(crossValidate=True,filterKnight=True,foldId=i)
            print len(trainInputs)
            print len(validInputs)
            print len(testInputs)
        self.sequences = readData.getData(crossValidate=True,filterKnight=False,foldId=i)
