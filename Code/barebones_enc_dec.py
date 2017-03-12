import sys
import math
import random
import dynet as dy
import readData
from collections import defaultdict
import argparse
import numpy as np
import datetime
import nltk
import copy
import loadEmbeddingsFile
import utilities

def attend(encoder_outputs,state_factor_matrix):
    miniBatchLength=state_factor_matrix.npvalue().shape[1]
    encoderOutputLength=state_factor_matrix.npvalue().shape[0]
    hiddenSize=encoder_outputs[0].npvalue().shape[0]

    factor_Products=[state_factor_matrix[l] for l in range(encoderOutputLength)]
    factor_Products=dy.esum([dy.cmult(encoder_outputs[l],dy.concatenate([state_factor_matrix[l]]*hiddenSize)) for l in range(encoderOutputLength)])
    
    return factor_Products

def attend_vector(encoder_outputs,state_factor_vector):
    encoderOutputLength=state_factor_vector.npvalue().shape[0]
    hiddenSize=encoder_outputs[0].npvalue().shape[0]
    
    factor_Products=[dy.cmult(dy.concatenate([state_factor_vector[l]]*hiddenSize),encoder_outputs[l]) for l in range(encoderOutputLength)]
   
    factor_Products=dy.esum(factor_Products)
    return factor_Products

def topk(vector,k):
    topklist=[]
    while len(topklist)<k:
        top=np.argmax(vector)
        topklist.append(top)
        vector[top]=-np.inf

    return topklist

def beamDecode(model,encoder,revcoder,decoder,encoder_params,decoder_params,sentence_de,downstream=False,k=10):
    dy.renew_cg()
    encoder_lookup=encoder_params["lookup"]
    decoder_lookup=decoder_params["lookup"]
    R=dy.parameter(decoder_params["R"])
    bias=dy.parameter(decoder_params["bias"])

    sentence_de_forward=sentence_de
    sentence_de_reverse=sentence_de[::-1]

    s=encoder.initial_state()
    inputs=[dy.lookup(encoder_lookup,de) for de in sentence_de_forward]
    states=s.add_inputs(inputs)
    encoder_outputs=[s.output() for s in states]

    s_reverse=revcoder.initial_state()
    inputs=[dy.lookup(encoder_lookup,de) for de in sentence_de_reverse]
    states_reverse=s_reverse.add_inputs(inputs)
    revcoder_outputs=[s.output() for s in states_reverse]

    final_coding_output=encoder_outputs[-1]+revcoder_outputs[-1]
    final_state=states[-1].s()
    final_state_reverse=states_reverse[-1].s()
    final_coding_state=((final_state_reverse[0]+final_state[0]),(final_state_reverse[1]+final_state[1]))
    final_combined_outputs=[revcoder_output+encoder_output for revcoder_output,encoder_output in zip(revcoder_outputs[::-1],encoder_outputs)]

    s_init=decoder.initial_state().set_s(final_state_reverse)
    o_init=s_init.output() 
    alpha_init=dy.softmax(dy.concatenate([dy.dot_product(o_init,final_combined_output) for final_combined_output in final_combined_outputs]))
    c_init=attend_vector(final_combined_outputs,alpha_init)

    
    s_0=s_init
    o_0=o_init
    alpha_0=alpha_init
    c_0=c_init
    
    finishedSequences=[]
    currentSequences=[(s_0,c_0,o_0,[],0.0),]

    #print "Beam Search Start"
    while len(finishedSequences)<2*k:
        candidates=[]
        for currentSequence in currentSequences:
            scores=None
            if downstream:
                scores=dy.affine_transform([bias,R,dy.concatenate([currentSequence[2],currentSequence[1]])])
            else:
                scores=dy.affine_transform([bias,R,currentSequence[2]])
            topkTokens=topk(scores.npvalue(),k)
            for topkToken in topkTokens:
                loss=(dy.pickneglogsoftmax(scores,topkToken)).value()
                candidate_i_t=dy.concatenate([dy.lookup(decoder_lookup,topkToken),currentSequence[1]])
                candidate_s_t=currentSequence[0].add_input(candidate_i_t)
                candidate_o_t=candidate_s_t.output()
                candidate_alpha_t=dy.softmax(dy.concatenate([dy.dot_product(candidate_o_t,final_combined_output) for final_combined_output in final_combined_outputs]))
                candidate_c_t=attend_vector(final_combined_outputs,candidate_alpha_t)
                candidate_loss=currentSequence[4]+loss
                candidate_sequence=copy.deepcopy(currentSequence[3])
                candidate_sequence.append(topkToken)
                candidate=(candidate_s_t,candidate_c_t,candidate_o_t,candidate_sequence,candidate_loss)
                if topkToken==STOP or len(candidate_sequence)>len(sentence_de)+10:
                    if len(candidate_sequence)>3 or len(candidate_sequence)>=len(sentence_de):
                        finishedSequences.append(candidate)
                else:
                    candidates.append(candidate)
        #Sort candidates by loss, lesser loss is better
        candidates.sort(key = lambda x: x[4])
        currentSequences=candidates[:k]

    #print "Beam Search End"

    finishedSequences.sort(key = lambda x:x[4])
    sentence_en=finishedSequences[0][3]      

    return loss,sentence_en





class Config:
    READ_OPTION="NORMAL"
    downstream=True
    sharing=False
    GRU=False

    def __init__(self,READ_OPTION="NORMAL",downstream=True,sharing=False,GRU=False,preTrain=False,initFromFile=False,initFileName=None,foldId=None):
        self.READ_OPTION=READ_OPTION
        self.downstream=downstream
        self.sharing=sharing
        self.GRU=GRU
        self.preTrain=preTrain
        self.initFromFile=initFromFile
        self.initFileName=initFileName
        self.foldId=foldId

class HyperParams:
    EMB_SIZE=None
    LAYER_DEPTH=None
    HIDDEN_SIZE=None
    NUM_EPOCHS=None
    STOP=None
    SEPARATOR=None

    def __init__(self,EMB_SIZE=50,LAYER_DEPTH=1,HIDDEN_SIZE=100,NUM_EPOCHS=10,STOP=0,SEPARATOR=1,dMethod="REVERSE"):
        #Hyperparameter Definition
        self.EMB_SIZE=EMB_SIZE
        self.LAYER_DEPTH=LAYER_DEPTH
        self.HIDDEN_SIZE=HIDDEN_SIZE
        self.NUM_EPOCHS=NUM_EPOCHS
        self.STOP=STOP
        self.SEPARATOR=SEPARATOR
        self.dMethod=dMethod

def equalSequence(x,y):
    if len(x)!=len(y):
        return False

    for i,alpha in enumerate(x):
        if y[i]!=alpha:
            return False

    return True

class Model:

    def beamDecode(self,sentence_de,k=1):
        model=self.model
        encoder=self.encoder
        revcoder=self.revcoder
        decoder=self.decoder
        encoder_params=self.encoder_params
        decoder_params=self.decoder_params
        downstream=self.config.downstream
        GRU=self.config.GRU
        hyperParams=self.hyperParams

        part1=sentence_de[:sentence_de.index(hyperParams.SEPARATOR)]+[hyperParams.STOP,]
        part2=sentence_de[sentence_de.index(hyperParams.SEPARATOR)+1:]
        part1Length=len(part1)
        part2Length=len(part2)

        dy.renew_cg() 
        encoder_lookup=encoder_params["lookup"]
        decoder_lookup=decoder_params["lookup"]
        R=dy.parameter(decoder_params["R"])
        bias=dy.parameter(decoder_params["bias"])

        sentence_de_forward=sentence_de
        sentence_de_reverse=sentence_de[::-1]

        s=encoder.initial_state()
        inputs=[dy.lookup(encoder_lookup,de) for de in sentence_de_forward]
        states=s.add_inputs(inputs)
        encoder_outputs=[s.output() for s in states]

        s_reverse=revcoder.initial_state()
        inputs=[dy.lookup(encoder_lookup,de) for de in sentence_de_reverse]
        states_reverse=s_reverse.add_inputs(inputs)
        revcoder_outputs=[s.output() for s in states_reverse]

        final_coding_output=encoder_outputs[-1]+revcoder_outputs[-1]
        final_state=states[-1].s()
        final_state_reverse=states_reverse[-1].s()

        if GRU:
            final_coding_state=final_state_reverse+final_state
        else:
            final_coding_state=((final_state_reverse[0]+final_state[0]),(final_state_reverse[1]+final_state[1]))
        final_combined_outputs=[revcoder_output+encoder_output for revcoder_output,encoder_output in zip(revcoder_outputs[::-1],encoder_outputs)]

        initDict={}
        initDict["FORWARD"]=final_state
        initDict["REVERSE"]=final_state_reverse
        initDict["BI"]=final_coding_state

        s_init=decoder.initial_state().set_s(initDict[hyperParams.dMethod])
        #s_init=decoder.initial_state().set_s(final_coding_state)
        o_init=s_init.output() 
        alpha_init=dy.softmax(dy.concatenate([dy.dot_product(o_init,final_combined_output) for final_combined_output in final_combined_outputs]))
        c_init=attend_vector(final_combined_outputs,alpha_init)
        
        s_0=s_init
        o_0=o_init
        alpha_0=alpha_init
        c_0=c_init
       
        finishedSequences=[]
        currentSequences=[(s_0,c_0,o_0,[],0.0),]

        #print "Beam Search Start"
        while len(finishedSequences)<k:
            candidates=[]
            for currentSequence in currentSequences:
                scores=None
                if downstream:
                    scores=dy.affine_transform([bias,R,dy.concatenate([currentSequence[2],currentSequence[1]])])
                else:
                    scores=dy.affine_transform([bias,R,currentSequence[2]])
                topkTokens=topk(scores.npvalue(),k)
                for topkToken in topkTokens:
                    loss=(dy.pickneglogsoftmax(scores,topkToken)).value()
                    candidate_i_t=dy.concatenate([dy.lookup(decoder_lookup,topkToken),currentSequence[1]])
                    candidate_s_t=currentSequence[0].add_input(candidate_i_t)
                    candidate_o_t=candidate_s_t.output()
                    candidate_alpha_t=dy.softmax(dy.concatenate([dy.dot_product(candidate_o_t,final_combined_output) for final_combined_output in final_combined_outputs]))
                    candidate_c_t=attend_vector(final_combined_outputs,candidate_alpha_t)
                    candidate_loss=currentSequence[4]+loss
                    candidate_sequence=copy.deepcopy(currentSequence[3])
                    candidate_sequence.append(topkToken)
                    candidate=(candidate_s_t,candidate_c_t,candidate_o_t,candidate_sequence,candidate_loss)
                    if topkToken==hyperParams.STOP or len(candidate_sequence)>len(sentence_de)+20:
                        if len(candidate_sequence)>2:
                            finishedSequences.append(candidate)
                    else:
                        candidates.append(candidate)
            #Sort candidates by loss, lesser loss is better
            candidates.sort(key = lambda x: x[4])
            currentSequences=candidates[:k]

        #print "Beam Search End"

        prunedFinishedSequences=[]
        
        #Remove output sequences identical to parents. If that leaves nothing, fall back onto the original
        for finishedSequence in finishedSequences:
            if equalSequence(finishedSequence[3],part1) or equalSequence(finishedSequence[3],part2):
                continue
            else:
                prunedFinishedSequences.append(finishedSequence)

        if len(prunedFinishedSequences)!=0:
            finishedSequences=prunedFinishedSequences

        finishedSequences.sort(key = lambda x:x[4])
        sentence_en=finishedSequences[0][3]      
        loss_en=finishedSequences[0][4]

        return loss_en,sentence_en

    def greedyDecode(self,sentence_de):
        model=self.model
        encoder=self.encoder
        revcoder=self.revcoder
        decoder=self.decoder
        encoder_params=self.encoder_params
        decoder_params=self.decoder_params
        downstream=self.config.downstream
        GRU=self.config.GRU
        hyperParams=self.hyperParams

        dy.renew_cg() 
        encoder_lookup=encoder_params["lookup"]
        decoder_lookup=decoder_params["lookup"]
        R=dy.parameter(decoder_params["R"])
        bias=dy.parameter(decoder_params["bias"])

        sentence_de_forward=sentence_de
        sentence_de_reverse=sentence_de[::-1]

        s=encoder.initial_state()
        inputs=[dy.lookup(encoder_lookup,de) for de in sentence_de_forward]
        states=s.add_inputs(inputs)
        encoder_outputs=[s.output() for s in states]

        s_reverse=revcoder.initial_state()
        inputs=[dy.lookup(encoder_lookup,de) for de in sentence_de_reverse]
        states_reverse=s_reverse.add_inputs(inputs)
        revcoder_outputs=[s.output() for s in states_reverse]

        final_coding_output=encoder_outputs[-1]+revcoder_outputs[-1]
        final_state=states[-1].s()
        final_state_reverse=states_reverse[-1].s()

        if GRU:
            final_coding_state=final_state_reverse+final_state
        else:
            final_coding_state=((final_state_reverse[0]+final_state[0]),(final_state_reverse[1]+final_state[1]))
        final_combined_outputs=[revcoder_output+encoder_output for revcoder_output,encoder_output in zip(revcoder_outputs[::-1],encoder_outputs)]

        initDict={}
        initDict["FORWARD"]=final_state
        initDict["REVERSE"]=final_state_reverse
        initDict["BI"]=final_coding_state

        s_init=decoder.initial_state().set_s(initDict[hyperParams.dMethod])
        #s_init=decoder.initial_state().set_s(final_coding_state)
        o_init=s_init.output() 
        alpha_init=dy.softmax(dy.concatenate([dy.dot_product(o_init,final_combined_output) for final_combined_output in final_combined_outputs]))
        c_init=attend_vector(final_combined_outputs,alpha_init)
        
        s_0=s_init
        o_0=o_init
        alpha_0=alpha_init
        c_0=c_init
        

        losses=[]
        currentToken=None
        englishSequence=[]

        while currentToken!=hyperParams.STOP and len(englishSequence)<len(sentence_de)+10:
            #Calculate loss and append to the losses array
            scores=None
            if downstream:
                scores=R*dy.concatenate([o_0,c_0])+bias
            else:
                scores=R*o_0+bias
            currentToken=np.argmax(scores.npvalue())
            loss=dy.pickneglogsoftmax(scores,currentToken)
            losses.append(loss)
            englishSequence.append(currentToken)

            #Take in input
            i_t=dy.concatenate([dy.lookup(decoder_lookup,currentToken),c_0])
            s_t=s_0.add_input(i_t)
            o_t=s_t.output()
            alpha_t=dy.softmax(dy.concatenate([dy.dot_product(o_t,final_combined_output) for final_combined_output in final_combined_outputs]))
            c_t=attend_vector(final_combined_outputs,alpha_t)
            
            #Prepare for the next iteration
            s_0=s_t
            o_0=o_t
            c_0=c_t
            alpha_0=alpha_t

        total_loss=dy.esum(losses)
        return total_loss,englishSequence



    def greedyDecode(self,sentence_de):
        model=self.model
        encoder=self.encoder
        revcoder=self.revcoder
        decoder=self.decoder
        encoder_params=self.encoder_params
        decoder_params=self.decoder_params
        downstream=self.config.downstream
        GRU=self.config.GRU
        hyperParams=self.hyperParams

        dy.renew_cg() 
        encoder_lookup=encoder_params["lookup"]
        decoder_lookup=decoder_params["lookup"]
        R=dy.parameter(decoder_params["R"])
        bias=dy.parameter(decoder_params["bias"])

        sentence_de_forward=sentence_de
        sentence_de_reverse=sentence_de[::-1]

        s=encoder.initial_state()
        inputs=[dy.lookup(encoder_lookup,de) for de in sentence_de_forward]
        states=s.add_inputs(inputs)
        encoder_outputs=[s.output() for s in states]

        s_reverse=revcoder.initial_state()
        inputs=[dy.lookup(encoder_lookup,de) for de in sentence_de_reverse]
        states_reverse=s_reverse.add_inputs(inputs)
        revcoder_outputs=[s.output() for s in states_reverse]

        final_coding_output=encoder_outputs[-1]+revcoder_outputs[-1]
        final_state=states[-1].s()
        final_state_reverse=states_reverse[-1].s()

        if GRU:
            final_coding_state=final_state_reverse+final_state
        else:
            final_coding_state=((final_state_reverse[0]+final_state[0]),(final_state_reverse[1]+final_state[1]))
        final_combined_outputs=[revcoder_output+encoder_output for revcoder_output,encoder_output in zip(revcoder_outputs[::-1],encoder_outputs)]

        initDict={}
        initDict["FORWARD"]=final_state
        initDict["REVERSE"]=final_state_reverse
        initDict["BI"]=final_coding_state

        s_init=decoder.initial_state().set_s(initDict[hyperParams.dMethod])
        #s_init=decoder.initial_state().set_s(final_coding_state)
        o_init=s_init.output() 
        alpha_init=dy.softmax(dy.concatenate([dy.dot_product(o_init,final_combined_output) for final_combined_output in final_combined_outputs]))
        c_init=attend_vector(final_combined_outputs,alpha_init)
        
        s_0=s_init
        o_0=o_init
        alpha_0=alpha_init
        c_0=c_init
        

        losses=[]
        currentToken=None
        englishSequence=[]

        while currentToken!=hyperParams.STOP and len(englishSequence)<len(sentence_de)+10:
            #Calculate loss and append to the losses array
            scores=None
            if downstream:
                scores=R*dy.concatenate([o_0,c_0])+bias
            else:
                scores=R*o_0+bias
            currentToken=np.argmax(scores.npvalue())
            loss=dy.pickneglogsoftmax(scores,currentToken)
            losses.append(loss)
            englishSequence.append(currentToken)

            #Take in input
            i_t=dy.concatenate([dy.lookup(decoder_lookup,currentToken),c_0])
            s_t=s_0.add_input(i_t)
            o_t=s_t.output()
            alpha_t=dy.softmax(dy.concatenate([dy.dot_product(o_t,final_combined_output) for final_combined_output in final_combined_outputs]))
            c_t=attend_vector(final_combined_outputs,alpha_t)
            
            #Prepare for the next iteration
            s_0=s_t
            o_0=o_t
            c_0=c_t
            alpha_0=alpha_t

        total_loss=dy.esum(losses)
        return total_loss,englishSequence


    def do_one_example_pretrain(self,sentence_de):
        model=self.model
        encoder=self.encoder
        revcoder=self.revcoder
        encoder_params=self.encoder_params
        downstream=self.config.downstream
        GRU=self.config.GRU

        dy.renew_cg()
        encoder_lookup=encoder_params["lookup"]
        R_DE=dy.parameter(encoder_params["R_DE"])
        bias_DE=dy.parameter(encoder_params["bias_DE"])

        sentence_de_forward=sentence_de
        sentence_de_reverse=sentence_de[::-1]

        s=encoder.initial_state()
        s=s.add_input(dy.lookup(encoder_lookup,self.hyperParams.SEPARATOR))
        
        inputs=[dy.lookup(encoder_lookup,de) for de in sentence_de_forward]
        states=[s,]+s.add_inputs(inputs)
        states=states[:-1]
        forward_scores=[]
        for s in states:
            o_0=s.output()
            forward_score=(R_DE)*(o_0)+(bias_DE)
            forward_scores.append(forward_score)

        forward_losses=[dy.pickneglogsoftmax(forward_scores[i],de) for i,de in enumerate(sentence_de_forward)]


        s_reverse=revcoder.initial_state()
        s_reverse=s_reverse.add_input(dy.lookup(encoder_lookup,self.hyperParams.SEPARATOR))
        inputs=[dy.lookup(encoder_lookup,de) for de in sentence_de_reverse]
        states_reverse=[s_reverse,]+s_reverse.add_inputs(inputs)
        states=states_reverse[:-1]
        backward_scores=[]
        for s in states_reverse:
            o_0=s.output()
            backward_score=(R_DE)*o_0+(bias_DE)
            backward_scores.append(backward_score)

        backward_losses=[dy.pickneglogsoftmax(backward_scores[i],de) for i,de in enumerate(sentence_de_reverse)]

        forward_loss=dy.esum(forward_losses)
        backward_loss=dy.esum(backward_losses)
        losses=[forward_loss,]+[backward_loss,]
        random.shuffle(losses)
        
        return losses

    def genDecode(self,sentence_de,useBaseline=False):
        part1=sentence_de[:sentence_de.index(self.hyperParams.SEPARATOR)]
        part2=sentence_de[sentence_de.index(self.hyperParams.SEPARATOR)+1:-1]
        part1=[self.reverse_wids[c] for c in part1]
        part2=[self.reverse_wids[c] for c in part2]
        part1=''.join(part1)
        part2=''.join(part2)
        parentSet=set()
        parentSet.add(part1)
        parentSet.add(part2)
        candidates=list(utilities.generateCandidates(part1,part2)-parentSet)
        if useBaseline==False:
            candidates=[[self.wids[c] for c in candidate]+[self.hyperParams.STOP,] for candidate in candidates]
            prunedCandidates=[]
            for candidate in candidates:
                if len(candidate)>4:
                    prunedCandidates.append(candidate)
            candidates=prunedCandidates
            losses=[]
            for candidate in candidates:
                loss,words=self.do_one_example(sentence_de,candidate)
                loss=np.sum(loss.npvalue())
                losses.append(loss)
            candidateLosses=zip(candidates,losses)
            candidateLosses.sort(key= lambda x:x[1])

            print [self.reverse_wids[c] for c in candidateLosses[0][0]]
            #exit()
            return candidateLosses[0][1],candidateLosses[0][0] 
        else:
            prunedCandidates=[]
            for candidate in candidates:
                if len(candidate)>4:
                    prunedCandidates.append(candidate)
            candidates=prunedCandidates
            losses=[]
            for candidate in candidates:
                loss=self.lm_model.getSequenceScore(candidate)
                losses.append(loss)
            candidateLosses=zip(candidates,losses)
            candidateLosses.sort(key=lambda x:x[1])

            print candidateLosses[0][0]
            bestCandidate=[self.wids[c] for c in candidateLosses[0][0]]+[self.hyperParams.STOP,]
            bestCandidateLoss=candidateLosses[0][1]
            return bestCandidateLoss,bestCandidate

    def do_one_example(self,sentence_de,sentence_en):
        model=self.model
        encoder=self.encoder
        revcoder=self.revcoder
        decoder=self.decoder
        encoder_params=self.encoder_params
        decoder_params=self.decoder_params
        downstream=self.config.downstream
        GRU=self.config.GRU
        hyperParams=self.hyperParams

        dy.renew_cg()
        total_words=len(sentence_en)
        encoder_lookup=encoder_params["lookup"]
        decoder_lookup=decoder_params["lookup"]
        R=dy.parameter(decoder_params["R"])
        bias=dy.parameter(decoder_params["bias"])

        sentence_de_forward=sentence_de
        sentence_de_reverse=sentence_de[::-1]

        s=encoder.initial_state()
        inputs=[dy.lookup(encoder_lookup,de) for de in sentence_de_forward]
        states=s.add_inputs(inputs)
        encoder_outputs=[s.output() for s in states]

        s_reverse=revcoder.initial_state()
        inputs=[dy.lookup(encoder_lookup,de) for de in sentence_de_reverse]
        states_reverse=s_reverse.add_inputs(inputs)
        revcoder_outputs=[s.output() for s in states_reverse]

        final_coding_output=encoder_outputs[-1]+revcoder_outputs[-1]
        final_state=states[-1].s()
        final_state_reverse=states_reverse[-1].s()

        if GRU:
            final_coding_state=final_state_reverse+final_state
        else:
            final_coding_state=((final_state_reverse[0]+final_state[0]),(final_state_reverse[1]+final_state[1]))
        final_combined_outputs=[revcoder_output+encoder_output for revcoder_output,encoder_output in zip(revcoder_outputs[::-1],encoder_outputs)]

        initDict={}
        initDict["FORWARD"]=final_state
        initDict["REVERSE"]=final_state_reverse
        initDict["BI"]=final_coding_state

        s_init=decoder.initial_state().set_s(initDict[hyperParams.dMethod])
        #s_init=decoder.initial_state().set_s(final_coding_state)
        o_init=s_init.output() 
        alpha_init=dy.softmax(dy.concatenate([dy.dot_product(o_init,final_combined_output) for final_combined_output in final_combined_outputs]))
        c_init=attend_vector(final_combined_outputs,alpha_init)

        
        s_0=s_init
        o_0=o_init
        alpha_0=alpha_init
        c_0=c_init
        

        losses=[]
        
        for en in sentence_en:
            #Calculate loss and append to the losses array
            scores=None
            if downstream:
                scores=R*dy.concatenate([o_0,c_0])+bias
            else:
                scores=R*o_0+bias
            loss=dy.pickneglogsoftmax(scores,en)
            losses.append(loss)

            #Take in input
            i_t=dy.concatenate([dy.lookup(decoder_lookup,en),c_0])
            s_t=s_0.add_input(i_t)
            o_t=s_t.output()
            alpha_t=dy.softmax(dy.concatenate([dy.dot_product(o_t,final_combined_output) for final_combined_output in final_combined_outputs]))
            c_t=attend_vector(final_combined_outputs,alpha_t)
            
            #Prepare for the next iteration
            s_0=s_t
            o_0=o_t
            c_0=c_t
            alpha_0=alpha_t

        total_loss=dy.esum(losses)
        return total_loss,total_words

    def save_model(self):
        print "Saving Model"
        self.model.save(self.modelFile,[self.encoder,self.revcoder,self.decoder,self.encoder_params["lookup"],self.decoder_params["lookup"],self.decoder_params["R"],self.decoder_params["bias"]])
        print "Model Saved"

    def load_model(self):
        print "Loading Model"
        (self.encoder,self.revcoder,self.decoder,self.encoder_params["lookup"],self.decoder_params["lookup"],self.decoder_params["R"],self.decoder_params["bias"])=self.model.load(self.modelFile)
        print "Model Loaded"

    def __init__(self,config,hyperParams,modelFile="Buffer/model"):
        self.config=config
        self.hyperParams=hyperParams
        self.modelFile=modelFile
        self.bestPerplexity=float("inf")

        if self.config.READ_OPTION=="KNIGHTCROSSVALIDATE":
            train_sentences_de,train_sentences_en,valid_sentences_de,valid_sentences_en,test_sentences_de,test_sentences_en,wids=readData.getData(filterKnight=True,crossValidate=True,foldId=config.foldId)
        elif self.config.READ_OPTION=="NORMAL":
            train_sentences_de,train_sentences_en,valid_sentences_de,valid_sentences_en,test_sentences_de,test_sentences_en,wids=readData.getData(trainingPoints=700,validPoints=400)
        elif self.config.READ_OPTION=="KNIGHTONLY":
            train_sentences_de,train_sentences_en,valid_sentences_de,valid_sentences_en,test_sentences_de,test_sentences_en,wids=readData.getData(trainingPoints=320,validPoints=40,filterKnight=True) 
        elif self.config.READ_OPTION=="NORMALDISJOINT":
            train_sentences_de,train_sentences_en,valid_sentences_de,valid_sentences_en,test_sentences_de,test_sentences_en,wids=readData.getDataDisjoint(trainingPoints=500,validPoints=200)
        elif self.config.READ_OPTION=="KNIGHTHOLDOUT":
            train_sentences_de,train_sentences_en,valid_sentences_de,valid_sentences_en,test_sentences_de,test_sentences_en,wids=readData.getDataKnightHoldOut(trainingPoints=1000)
        elif self.config.READ_OPTION=="PHONETICINPUT":
            train_sentences_de,train_sentences_en,valid_sentences_de,valid_sentences_en,test_sentences_de,test_sentences_en,wids,wids_phonetic=readData.getDataPhoneticInput(trainingPoints=700,validPoints=400)       

        self.train_sentences_de=train_sentences_de
        self.train_sentences_en=train_sentences_en
        self.valid_sentences_de=valid_sentences_de
        self.valid_sentences_en=valid_sentences_en
        self.test_sentences_de=test_sentences_de
        self.test_sentences_en=test_sentences_en
        
        
        self.wids=wids
        self.reverse_wids=readData.reverseDictionary(self.wids)
        if self.config.READ_OPTION=="PHONETICINPUT" or self.config.READ_OPTION=="PHONOLEXINPUT":
            self.wids_phonetic=wids_phonetic
            self.reverse_wids_phonetic=readData.reverseDictionary(self.wids_phonetic)

        print len(self.train_sentences_de)
        print len(self.train_sentences_en)

        print len(self.valid_sentences_de)
        print len(self.valid_sentences_en)

        if self.config.READ_OPTION=="PHONETICINPUT":
            self.VOCAB_SIZE_DE=len(self.wids_phonetic)
            self.VOCAB_SIZE_EN=len(self.wids)
        else:
            self.VOCAB_SIZE_DE=len(wids)
            self.VOCAB_SIZE_EN=self.VOCAB_SIZE_DE

        self.train_sentences=zip(self.train_sentences_de,self.train_sentences_en)
        self.valid_sentences=zip(self.valid_sentences_de,self.valid_sentences_en)
        self.test_sentences=zip(self.test_sentences_de,self.test_sentences_en)


        #Specify model
        self.model=dy.Model()

        config=self.config
        hyperParams=self.hyperParams
        model=self.model

        if self.config.GRU:
            self.encoder=dy.GRUBuilder(hyperParams.LAYER_DEPTH,hyperParams.EMB_SIZE,hyperParams.HIDDEN_SIZE,model)
            self.revcoder=dy.GRUBuilder(hyperParams.LAYER_DEPTH,hyperParams.EMB_SIZE,hyperParams.HIDDEN_SIZE,model)
            self.decoder=dy.GRUBuilder(hyperParams.LAYER_DEPTH,hyperParams.EMB_SIZE+hyperParams.HIDDEN_SIZE,hyperParams.HIDDEN_SIZE,model)
        else:
            self.encoder=dy.LSTMBuilder(hyperParams.LAYER_DEPTH,hyperParams.EMB_SIZE,hyperParams.HIDDEN_SIZE,model)
            self.revcoder=dy.LSTMBuilder(hyperParams.LAYER_DEPTH,hyperParams.EMB_SIZE,hyperParams.HIDDEN_SIZE,model)
            self.decoder=dy.LSTMBuilder(hyperParams.LAYER_DEPTH,hyperParams.EMB_SIZE+hyperParams.HIDDEN_SIZE,hyperParams.HIDDEN_SIZE,model)

        if self.config.initFromFile:
            if self.config.READ_OPTION=="PHONETICINPUT":
                print "Cannot initialize phonetic embeddings from character embeddings"
                exit()
            preEmbeddings=loadEmbeddingsFile.loadEmbeddings(wids,self.config.initFileName,hyperParams.EMB_SIZE)
            
    
        self.encoder_params={}
        self.encoder_params["lookup"]=model.add_lookup_parameters((self.VOCAB_SIZE_DE,hyperParams.EMB_SIZE))
        
        if self.config.initFromFile:
            self.encoder_params["lookup"].init_from_array(preEmbeddings)

        self.decoder_params={}
        if config.sharing:
            self.decoder_params["lookup"]=self.encoder_params["lookup"]
        else:
            self.decoder_params["lookup"]=model.add_lookup_parameters((self.VOCAB_SIZE_EN,hyperParams.EMB_SIZE))

        if self.config.initFromFile:
            self.decoder_params["lookup"].init_from_array(preEmbeddings)

        if config.downstream:
            self.decoder_params["R"]=model.add_parameters((self.VOCAB_SIZE_EN,2*hyperParams.HIDDEN_SIZE))
        else:
            self.decoder_params["R"]=model.add_parameters((self.VOCAB_SIZE_EN,hyperParams.HIDDEN_SIZE))

        self.decoder_params["bias"]=model.add_parameters((self.VOCAB_SIZE_EN))

        if config.preTrain:
            self.encoder_params["R_DE"]=model.add_parameters((self.VOCAB_SIZE_DE,hyperParams.HIDDEN_SIZE))
            self.encoder_params["bias_DE"]=model.add_parameters((self.VOCAB_SIZE_DE))

        import example
        self.lm_model=example.lm_model

    def pretrain(self):
        trainer=dy.SimpleSGDTrainer(self.model)
        totalSentences=0
        for epochId in xrange(hyperParams.NUM_EPOCHS):    
            random.shuffle(self.train_sentences)
            for sentenceId,sentence in enumerate(self.train_sentences):
                totalSentences+=1
                sentence_de=sentence[0]
                losses=self.do_one_example_pretrain(sentence_de)
                for loss in losses:
                    loss.value()
                    loss.backward()
                    trainer.update()
            trainer.update_epoch(1.0)


    def train(self,interEpochPrinting=False):
        if self.config.preTrain:
            print "Pretraining"
            self.pretrain()

        trainer=dy.SimpleSGDTrainer(self.model)
        totalSentences=0
        for epochId in xrange(hyperParams.NUM_EPOCHS):    
            random.shuffle(self.train_sentences)
            for sentenceId,sentence in enumerate(self.train_sentences):
                totalSentences+=1
                sentence_de=sentence[0]
                sentence_en=sentence[1]
                loss,words=self.do_one_example(sentence_de,sentence_en)
                loss.value()
                loss.backward()
                trainer.update()
                if interEpochPrinting:
                    if totalSentences%100==0:
                        #random.shuffle(valid_sentences)
                        perplexity=0.0
                        totalLoss=0.0
                        totalWords=0.0
                        for valid_sentence in self.valid_sentences:
                            valid_sentence_de=valid_sentence[0]
                            valid_sentence_en=valid_sentence[1]
                            validLoss,words=self.do_one_example(valid_sentence_de,valid_sentence_en)
                            totalLoss+=float(validLoss.value())
                            totalWords+=words
                        print totalLoss
                        print totalWords
                        perplexity=math.exp(totalLoss/totalWords)
                        print "Validation perplexity after epoch:",epochId,"sentenceId:",sentenceId,"Perplexity:",perplexity,"Time:",datetime.datetime.now()             
            trainer.update_epoch(1.0)
            perplexity=0.0
            totalLoss=0.0
            totalWords=0.0
            for valid_sentence in self.valid_sentences:
                valid_sentence_de=valid_sentence[0]
                valid_sentence_en=valid_sentence[1]
                validLoss,words=self.do_one_example(valid_sentence_de,valid_sentence_en)
                totalLoss+=float(validLoss.value())
                totalWords+=words
            perplexity=math.exp(totalLoss/totalWords)
            if epochId>=5 and perplexity<self.bestPerplexity:
                self.save_model()
                self.bestPerplexity=perplexity
                print "Best Perplexity:",self.bestPerplexity

        #self.save_model()
        #self.load_model()

    def stripStops(self,sentence):
        if sentence[-1]==self.hyperParams.STOP:
            return sentence[:-1]
        else:
            return sentence

    def testOut(self,valid_sentences,verbose=True,originalFileName="originalWords.txt",outputFileName="outputWords.txt",decodeMethod="greedy",beam_decode_k=1):
        reverse_wids=self.reverse_wids

        if self.config.READ_OPTION=="PHONETICINPUT" or self.config.READ_OPTION=="PHONOLEXINPUT":
            reverse_wids_phonetic=self.reverse_wids_phonetic

        originalWordFile=open(originalFileName,"w")
        outputWordFile=open(outputFileName,"w")
        import editdistance
        exactMatches=0
        editDistance=0.0
        

        for validSentenceId,validSentence in enumerate(valid_sentences):
            valid_sentence_de=validSentence[0]
            valid_sentence_en=validSentence[1]
            if decodeMethod=="greedy":
                validLoss,valid_sentence_en_hat=self.greedyDecode(valid_sentence_de)
            elif decodeMethod=="beam":
                validLoss,valid_sentence_en_hat=self.beamDecode(valid_sentence_de,k=beam_decode_k)
            elif decodeMethod=="gen":
                validLoss,valid_sentence_en_hat=self.genDecode(valid_sentence_de)
            elif decodeMethod=="genBase":
                validLoss,valid_sentence_en_hat=self.genDecode(valid_sentence_de,useBaseline=True)
 
            #valid_sentence_en_stripped=self.stripStops(valid_sentence_en)
            #valid_sentence_en_hat_stripped=self.stripStops(valid_sentence_en_hat)

            originalWord="".join([reverse_wids[c] for c in valid_sentence_en[:-1]])
            outputWord="".join([reverse_wids[c] for c in valid_sentence_en_hat[:-1]])
    
            if originalWord==outputWord:
                exactMatches+=1

            editDistance+=editdistance.eval(originalWord,outputWord)
            
            if verbose:
                if self.config.READ_OPTION=="PHONETICINPUT" or self.config.READ_OPTION=="PHONOLEXINPUT":
                    print "Input Word Pair:,","".join([reverse_wids_phonetic[c] for c in valid_sentence_de])
                else:
                    print "Input Word Pair:,","".join([reverse_wids[c] for c in valid_sentence_de])

                print "Original Word:,",originalWord    
                print "Output Word:,",outputWord

        originalWordFile.write(originalWord+"\n")
        outputWordFile.write(outputWord+"\n")

        totalWords=len(valid_sentences)
        
        
        
        print "Total Words",totalWords
        print "Exact Matches",exactMatches
        print "Average Edit Distance",(editDistance+0.0)/(totalWords+0.0)


        originalWordFile.close()
        outputWordFile.close()
        
        return exactMatches,(editDistance+0.0)/(totalWords+0.0)
        

if __name__=="__main__":

    if sys.argv[1]=="PLAIN":
        random.seed(491)

        READ_OPTION="KNIGHTONLY"
        preTrain=False
        downstream=True
        sharing=False
        GRU=False
        initFromFile=True
        initFileName="../Pretrained/output_embeddings_134iter_lowestValLoss.txt"

        config=Config(READ_OPTION=READ_OPTION,downstream=downstream,sharing=sharing,GRU=GRU,preTrain=preTrain,initFromFile=initFromFile,initFileName=initFileName)
        hyperParams=HyperParams(NUM_EPOCHS=10)

        predictor=Model(config,hyperParams)
        predictor.train(interEpochPrinting=False)

        print "Greedy Decode"
        print "Validation Performance"
        predictor.testOut(predictor.valid_sentences,verbose=False,originalFileName="originalWords.txt",outputFileName="outputWords.txt")
        print "Test Performance"
        predictor.testOut(predictor.test_sentences,verbose=False,originalFileName="originalWords.txt",outputFileName="outputWords.txt")

        
        #print "Beam Decode"
        #predictor.testOut(predictor.valid_sentences,verbose=False,originalFileName="originalWords.txt",outputFileName="outputWords.txt",decodeMethod="beam",beam_decode_k=5)
        """
        #predictor.testOut(predictor.test_sentences,verbose=False,originalFileName="originalWords.txt",outputFileName="outputWords.txt",decodeMethod="beam",beam_decode_k=2)
        """

        predictor.load_model()

        print "Greedy Decode"
        print "Validation Performance"
        predictor.testOut(predictor.valid_sentences,verbose=False,originalFileName="originalWords.txt",outputFileName="outputWords.txt")
        #print "Test Performance"
        predictor.testOut(predictor.test_sentences,verbose=False,originalFileName="originalWords.txt",outputFileName="outputWords.txt")

        """
        print "Beam Decode"
        predictor.testOut(predictor.valid_sentences,verbose=False,originalFileName="originalWords.txt",outputFileName="outputWords.txt",decodeMethod="beam",beam_decode_k=5)
        #predictor.testOut(predictor.test_sentences,verbose=False,originalFileName="originalWords.txt",outputFileName="outputWords.txt",decodeMethod="beam",beam_decode_k=2)
        """

    elif sys.argv[1]=="CROSSVALTRAIN":
        random.seed(491)

        READ_OPTION="KNIGHTCROSSVALIDATE"
        preTrain=False
        downstream=True
        sharing=False
        GRU=False
        initFromFile=True
        initFileName="../Pretrained/output_embeddings_134iter_lowestValLoss.txt"
        dMethod="REVERSE"
        decodeMethod="genBase"

        averageValidMatches=0.0
        averageTestMatches=0.0
        averageValidDistance=0.0
        averageTestDistance=0.0

        averageValidMatchesBest=0.0
        averageTestMatchesBest=0.0
        averageValidDistanceBest=0.0
        averageTestDistanceBest=0.0



        for foldId in range(10):
            config=Config(READ_OPTION=READ_OPTION,downstream=downstream,sharing=sharing,GRU=GRU,preTrain=preTrain,initFromFile=initFromFile,initFileName=initFileName,foldId=foldId)
            hyperParams=HyperParams(NUM_EPOCHS=10,dMethod=dMethod)

            predictor=Model(config,hyperParams)
            predictor.train(interEpochPrinting=False)

            print "Greedy Decode"
            print "Validation Performance"
            validMatches,validDistance=predictor.testOut(predictor.valid_sentences,verbose=False,originalFileName="originalWords.txt",outputFileName="outputWords.txt",decodeMethod=decodeMethod)
            print "Test Performance"
            testMatches,testDistance=predictor.testOut(predictor.test_sentences,verbose=False,originalFileName="originalWords.txt",outputFileName="outputWords.txt",decodeMethod=decodeMethod)


            averageValidMatches+=validMatches
            averageValidDistance+=validDistance
            averageTestMatches+=testMatches
            averageTestDistance+=testDistance
            
            predictor.load_model()

            print "Validation Performance Best"
            validMatches,validDistance=predictor.testOut(predictor.valid_sentences,verbose=False,originalFileName="originalWords.txt",outputFileName="outputWords.txt",decodeMethod=decodeMethod)
            print "Test Performance Best"
            testMatches,testDistance=predictor.testOut(predictor.test_sentences,verbose=False,originalFileName="originalWords.txt",outputFileName="outputWords.txt",decodeMethod=decodeMethod)

            averageValidMatchesBest+=validMatches
            averageValidDistanceBest+=validDistance
            averageTestMatchesBest+=testMatches
            averageTestDistanceBest+=testDistance
            


        print "Average Valid Matches",averageValidMatches/10
        print "Average Test Matches",averageTestMatches/10
        print "Average Valid Distance",averageValidDistance/10
        print "Average Test Distance",averageTestDistance/10

        print "Average Valid Matches Best",averageValidMatchesBest/10
        print "Average Test Matches Best",averageTestMatchesBest/10
        print "Average Valid Distance Best",averageValidDistanceBest/10
        print "Average Test Distance Best",averageTestDistanceBest/10
