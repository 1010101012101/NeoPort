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

class SimpleAgent:
	def __init__(self):
		self.policy_model = None
		self.value_model = None

	def getPolicyValueModels(self, params, weight=None  ):
		
		actions_space_size = params['actions_space_size']
		state_size = params['state_size']
		value_model_hidden_rep_size = params['value_model_hidden_rep_size']
		state = Input(shape=(state_size,), dtype='float32', name="state")

		#################
		advantages = Input(shape=(1,), dtype='float32', name="advantages")
		good_actions = Input(shape=(actions_space_size,), dtype='float32', name="good_actions")
		actions = Dense(actions_space_size, activation='softmax')(state)
		probs_good = merge( [actions, good_actions], mode='mul', output_shape=(actions_space_size,) ) # element wise multiply with sum
		#probs_good_log = K.log(probs_good)

		def lambda_layer_function(x):
			probs_good_log = K.log(0.01 + x)
			return probs_good_log * advantages
		probs_good_log_advantages = keras.layers.core.Lambda(lambda_layer_function) (probs_good)
		#_tmp = keras.layers.core.Lambda(lambda_layer_function) (probs_good)
		#probs_good_log_advantages = Dense(1) (probs_good_log_advantages_tmp)

		def policy_loss_function(probs_good_log_advantages_true, probs_good_log_advantages_pred):
			return -K.sum(probs_good_log_advantages_true)

		policy_update_model = Model(inputs=[state, advantages, good_actions], outputs=[probs_good_log_advantages])
		policy_update_model.compile(optimizer='adam', loss='mean_squared_error' )

		policy_predict_model = Model(inputs=[state], outputs=[actions])
		
		print policy_update_model.summary()
		print "---------------------------"
		print policy_predict_model.summary()

		#################
		hidden_rep = Dense(value_model_hidden_rep_size, activation='sigmoid', use_bias=True, name='value_hidden_layer')(state)
		value = Dense(1, name='value', activation='linear', use_bias=True) (hidden_rep)
		value_model = Model(inputs=[state], outputs=[value])
		value_model.compile(optimizer='adam', loss='mean_squared_error' )
		print "---------------------------"
		print value_model.summary()

		###############
		self.policy_update_model = policy_update_model
		self.policy_predict_model = policy_predict_model
		self.value_model = value_model

	
