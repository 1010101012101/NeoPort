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


def sampleFromDistribution(vals):
    #print "vals = ",vals
    p = random.random()
    s=0.0
    for i,v in enumerate(vals):
        s+=v
        if s>=p:
            return i
    return len(vals)-1

import matplotlib.pyplot as plt

def main():

    environment = environments.SampleEnvironment()
    obs = environment.reset()
    lim = 100
    for t in range(lim): 
        print obs
        a = 0 #np.random.randint(0, environment.actions_space_size)
        obs,reward, done = environment.performStep( a )
        print a, reward
        print "--------"
        if done:
            print "over"
            break

    agent = models.Agent()            
    params = dict(actions_space_size=environment.actions_space_size, 
        state_size=environment.state_space_size,
        value_model_hidden_rep_size=10
        )
    agent.getPolicyValueModels(params)

    all_rewards = []
    cor_ans = []
    cor_prob0 = []
    cor_prob1 = []
    for episode in range(400):
        print "======================================= "
        print "episode : ",episode
        obs = environment.reset()
        lim = 100
        r=0
        cur_episode_states = []
        cur_episode_actions = []
        cur_episode_predicted_values = []
        cur_episode_transitions = []
        for t in range(lim): 
            #print obs
            #a = 0 #np.random.randint(0, environment.actions_space_size)
            obs = np.array(obs).reshape(1, environment.state_space_size)
            a = agent.policy_predict_model.predict( [obs] )[0]
            #a = np.argmax(a)
            a = sampleFromDistribution(a)
            prev_obs = obs[:]
            obs, reward, done = environment.performStep( a )
            cur_episode_transitions.append( [prev_obs.reshape(environment.state_space_size), a, reward] )

            #print a, reward
            r+=reward
            #print "--------"
            if done:
                print "over"
                break
        #print "Reward = ",r
        all_rewards.append(r)

        update_vals = []
        advantages = []
        for i,transition in enumerate(cur_episode_transitions):
            prev_obs, a, reward = transition
            cur_episode_states.append(prev_obs)
            pred_val = agent.value_model.predict( prev_obs.reshape(1,environment.state_space_size) )[0]
            cur_episode_predicted_values.append( pred_val )
            cur_episode_actions.append(a)
            fut_reward = 0.0
            fut_transition_count = len(cur_episode_transitions) - i
            dec=1.0
            for j in range(fut_transition_count):
                nxt = i+j # first value is current reward 
                fut_reward+= ( dec * cur_episode_transitions[nxt][2] )
                dec*=0.97
            advantages.append( fut_reward - pred_val )
            update_vals.append(fut_reward)

        cur_episode_actions = keras.utils.np_utils.to_categorical(cur_episode_actions, 
            num_classes=environment.actions_space_size)
        #print cur_episode_actions
        cur_episode_actions = np.array(cur_episode_actions)
        cur_episode_states = np.array(cur_episode_states)
        advantages = np.array(advantages)
        #print cur_episode_states.shape
        update_vals = np.array(update_vals)
        #print update_vals.shape
        agent.value_model.fit( [cur_episode_states], [update_vals], epochs=1, batch_size=1 )

        #print cur_episode_states
        #print agent.policy_update_model.predict( [cur_episode_states, advantages, cur_episode_actions])
        agent.policy_update_model.fit( [cur_episode_states, advantages, cur_episode_actions] , np.random.rand(cur_episode_states.shape[0],5), epochs=1, batch_size=1)

        print " --->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ", np.mean( np.array(all_rewards) )
        tmp=0
        cor_prob0.append( agent.policy_predict_model.predict( np.array([[1.0, 0.0, 0.0, 0.0]]) )[0][0] )
        cor_prob1.append( agent.policy_predict_model.predict( np.array([[0.0, 1.0, 0.0, 0.0]]) )[0][1] )
        #print agent.policy_predict_model.predict( np.array([[1.0, 0.0, 0.0, 0.0]]) )[0]
        #print agent.policy_predict_model.predict( np.array([[0.0, 1.0, 0.0, 0.0]]) )[0]
        a_tmp = np.argmax( agent.policy_predict_model.predict( np.array([[0.0, 1.0, 0.0, 0.0]]) )[0] )
        #print "a_tmp = ",a_tmp
        if a_tmp==1:
            tmp+=1
        a_tmp = np.argmax( agent.policy_predict_model.predict( np.array([[1.0, 0.0, 0.0, 0.0]]) )[0] )
        if a_tmp==0:
            tmp+=1        
        a_tmp = np.argmax( agent.policy_predict_model.predict( np.array([[0.0, 0.0, 1.0, 0.0]]) )[0] )
        if a_tmp==2:
            tmp+=1
        a_tmp = np.argmax( agent.policy_predict_model.predict( np.array([[0.0, 0.0, 0.0, 1.0]]) )[0] )
        if a_tmp==3:
            tmp+=1            
        cor_ans.append(tmp)

    print cor_ans
    plt.plot(cor_ans)
    plt.show()
    plt.plot(cor_prob0)
    plt.show()
    plt.plot(cor_prob1)
    plt.show()    

if __name__ == "__main__":
    main()
