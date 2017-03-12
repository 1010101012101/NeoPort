import matplotlib.pyplot as plt
import numpy as np
import csv
import pickle
import sys
import math

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

	def getReverseTuplesLabels(self, data, labels):
		return self.reverseAllTuples(data), self.reverseAll(labels)

	def evaluateUsingLM(self, lm_model, data):
		trainInputs,trainOutputs,validInputs,validOutputs,testInputs,testOutputs,wids = data
		revers_dict = {i:ch for ch,i in wids.items()}
		self.revers_dict=revers_dict
		
		word_train, label_train = self.getReverseTuplesLabels( trainInputs, trainOutputs )
		print word_train[0]
		print word_train[1]
		print label_train[0]
		print label_train[1]
		word_val, label_val = self.getReverseTuplesLabels( validInputs, validOutputs )
		word_test, label_test = self.getReverseTuplesLabels( testInputs, testOutputs )

		fw = open("baseline_results.txt","w")
		fw.write("w1"+"\t"+"w2"+"\t"+"label"+"\t"+"prediction"+"\n")
		for i,w1w2 in enumerate(word_test):
			if i%10==0:
				print i
			w1,w2=w1w2
			label = label_test[i]
			candidates = utilities.generateCandidates(w1,w2)
			candidates=[c for c in candidates]
			all_scores = map(lm_model.getSequenceScore, candidates)
			best_score, best_score_idx = np.max(all_scores), np.argmax(all_scores)
			if i<5:
				print w1," ",w2
				print best_score," ",best_score_idx," ", candidates[best_score_idx]
				print ""
			fw.write(w1+"\t"+w2+"\t"+label+"\t"+candidates[best_score_idx])
			fw.write("\n")
		fw.close()


	def getFeats(self, w1, w2, wnew, lm_score):
		feats = []
		feats.append(len(w1))
		feats.append(len(w2))
		feats.append(len(wnew))
		feats.append(lm_score)
		feats.append(lm_score-math.log(len(wnew)))
		return feats  #{i+1:s for i,s in enumerate(feats)}

	def dumpRerankDataToFile(self, args):
		rerank_train_feats, rerank_val_feats, rerank_train_labels,rerank_val_labels = args
		'''
		3 qid:1 1:1 2:1 3:0 4:0.2 5:0
		2 qid:1 1:0 2:0 3:1 4:0.1 5:1
		1 qid:1 1:0 2:1 3:0 4:0.4 5:0
		1 qid:1 1:0 2:0 3:1 4:0.3 5:0
		'''
		fw = open("train.dat","w")
		qid=1
		for feats,ranks in zip(rerank_train_feats, rerank_train_labels):
			for cur_feats, cur_rank in zip(feats, ranks):
				s=[]
				s.append(str(cur_rank))
				s.append('qid:'+str(qid))
				for f,val in enumerate(cur_feats):
					s.append(str(f+1)+":"+str(val))
				s=' '.join(s)
				fw.write(s)
				fw.write('\n')
			qid+=1
		fw.close()


	def evaluateUsingDiscriminative(self, lm_model, data):
		trainInputs,trainOutputs,validInputs,validOutputs,testInputs,testOutputs,wids = data
		revers_dict = {i:ch for ch,i in wids.items()}
		self.revers_dict=revers_dict
		
		word_train, label_train = self.getReverseTuplesLabels( trainInputs, trainOutputs )
		print word_train[0]
		print word_train[1]
		print label_train[0]
		print label_train[1]
		word_val, label_val = self.getReverseTuplesLabels( validInputs, validOutputs )
		word_test, label_test = self.getReverseTuplesLabels( testInputs, testOutputs )

		rerank_train_feats = []
		rerank_val_feats = []
		rerank_train_labels=[]
		rerank_val_labels=[]

		for i,w1w2 in enumerate(word_train):
			if i%10==0:
				print i
			w1,w2=w1w2
			label = label_test[i]
			candidates = utilities.generateCandidates(w1,w2)
			candidates=[c for c in candidates]
			all_scores = map(lm_model.getSequenceScore, candidates)
			all_feats=[]
			edit_distances = []
			for j,candidate in enumerate(candidates):
				all_feats.append( self.getFeats(w1,w2,candidate,all_scores[j]) )
				edit_distances.append(utilities.getEditDistance(label, candidate))
			ranks = utilities.scoresToRanks(edit_distances, rev=True)
			rerank_train_feats.append( all_feats )
			rerank_train_labels.append( ranks )
			if i>10:
				break
		for i,w1w2 in enumerate(word_val):
			if i%10==0:
				print i
			w1,w2=w1w2
			label = label_test[i]
			candidates = utilities.generateCandidates(w1,w2)
			candidates=[c for c in candidates]
			all_scores = map(lm_model.getSequenceScore, candidates)
			all_feats=[]
			edit_distances = []
			for j,candidate in enumerate(candidates):
				all_feats.append( self.getFeats(w1,w2,candidate,all_scores[j]) )
				edit_distances.append(utilities.getEditDistance(label, candidate))
			ranks = utilities.scoresToRanks(edit_distances, rev=True)
			rerank_val_feats.append( all_feats )
			rerank_val_labels.append( ranks )
			if i>10:
				break
		self.dumpRerankDataToFile( (rerank_train_feats, rerank_val_feats, rerank_train_labels,rerank_val_labels) )
			


	def evaluate(self,params):
		data = self.data
		lm_model = self.lm_model
		method = params['method']
		trainInputs,trainOutputs,validInputs,validOutputs,testInputs,testOutputs,wids = data
		if method=="lm":
			self.evaluateUsingLM(lm_model, data)
		elif method=="rerank":
			self.evaluateUsingDiscriminative(lm_model, data)

def main():
	baseline = BaselineModel(("load","./language_model/checkpoints/weights.134-1.86.hdf5",None))
	print "------------------------------------------------------------------------"
	params = dict(method="rerank") #lm
	baseline.evaluate( params )


if __name__=="__main__":
	main()