import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Merge
from keras.utils import np_utils
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import Input, Embedding, LSTM, Dense, merge, SimpleRNN
import keras.backend as K
import keras

class SampleEnvironment:

	def __init__(self):
		self.state_space_size = 4
		self.actions_space_size=5
		self.actions = [0,1,2,3,4]
		self.n=10
		sequences = np.random.randint(0,self.state_space_size,(self.n,7))
		sequences = [ keras.utils.np_utils.to_categorical(sequence, num_classes=self.state_space_size) for sequence in sequences]
		self.sequences = sequences
		self.cur_sequence_idx=0
		self.cur_sequence=None
		self.position_in_cur_sequence=None
		self.cur_sequence_user_behavior=None # will be used to ascertain final reward
		#print sequences

	def getStateSpaceSize(self):
		return self.state_space_size

	def reset(self):
		self.cur_sequence_idx+=1
		if self.cur_sequence_idx >= self.n:
			self.cur_sequence_idx=0
		self.cur_sequence = self.sequences[self.cur_sequence_idx]
		self.position_in_cur_sequence = 0
		ret = self.cur_sequence[ self.position_in_cur_sequence ]
		self.cur_sequence_user_behavior=[]
		return ret

	def updateUserBehaviorAndGetObservationReward(self, action):
		reward=0
		if action<self.state_space_size:
			self.cur_sequence_user_behavior.append(action)
			#print "self.position_in_cur_sequence = ",self.position_in_cur_sequence
			if self.cur_sequence[ self.position_in_cur_sequence ] [action] >0.999:
				reward=0.2
			else:
				reward=-0.2
		else:
			reward = -0.1
			self.cur_sequence_user_behavior.append(None)
		#Update observation
		self.position_in_cur_sequence +=1
		next_obs = None
		if self.position_in_cur_sequence == len(self.cur_sequence):
			next_obs = None # over
		else:
			next_obs = self.cur_sequence[ self.position_in_cur_sequence ]
		return next_obs, reward

	def getBonusReward(self):
		r=0.0
		for i,s in enumerate(self.cur_sequence):
			a = self.cur_sequence_user_behavior[i]
			if a is None:
				continue
			r+=( s[a] * 1.0 )
		return r

	def performStep(self, action  ):
		# return obs, reward, done
		next_obs, reward = self.updateUserBehaviorAndGetObservationReward(action)
		done=False
		if next_obs==None:
			done=True
			reward+=self.getBonusReward()
		return next_obs, reward, done

	
