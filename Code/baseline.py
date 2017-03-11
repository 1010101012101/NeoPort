import matplotlib.pyplot as plt
import numpy as np
import csv
import pickle
import sys

sys.path.insert(0, sys.path[0]+'/language_model/')
print sys.path
import configuration as config
import utilities
import main_lm
import readData as datasets


class BaselineModel:

	separator_idx=1 # this should be same as in readData file
	separator="SEPARATOR"

	def __init__(self, args):
		#Load Data
		data = datasets.getData()
		self.data=data		
		#Load LM
		lm_model = main_lm.RNNLanguageModelHandler( args )
		self.lm_model=lm_model

	def reverse(self, seq):
		dct=self.revers_dict
		return [dct[i] for i in seq]

	def reverseAllTuples(self, data):
		data_sep_indexes = [ w.index(self.separator_idx) for w in data ]
		data_train = [ [self.reverse(w[:idx]), self.reverse( w[idx+1:]) ] for w,idx in zip(data,data_sep_indexes) ]
		data_train = [ [w[0],w[1][:-1]] for w in data_train ] #remove STOP from end of 2nd word
		data_train = [ [''.join(w[0]),''.join(w[1])] for w in data_train ] #remove STOP from end of 2nd word
		return data_train

	def reverseAll(self, data):
		data = [ ''.join(self.reverse(w[:-1])) for w in data ]
		return data

	def evaluateUsingLM(self, lm_model, data):
		trainInputs,trainOutputs,validInputs,validOutputs,testInputs,testOutputs,wids = data
		revers_dict = {i:ch for ch,i in wids.items()}
		self.revers_dict=revers_dict
		word_train = self.reverseAllTuples( trainInputs )
		print word_train[0]
		print word_train[1]
		label_train = self.reverseAll(trainOutputs)
		print label_train[0]
		print label_train[1]

	def evaluate(self,params):
		data = self.data
		lm_model = self.lm_model
		method = params['method']
		trainInputs,trainOutputs,validInputs,validOutputs,testInputs,testOutputs,wids = data
		if method=="lm":
			self.evaluateUsingLM(lm_model, data)

def main():
	baseline = BaselineModel(("load","./language_model/checkpoints/weights.134-1.86.hdf5",None))
	print "------------------------------------------------------------------------"
	params = dict(method="lm")
	baseline.evaluate( params )


if __name__=="__main__":
	main()