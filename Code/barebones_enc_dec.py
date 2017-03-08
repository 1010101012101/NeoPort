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

    def __init__(self,READ_OPTION="NORMAL",downstream=True,sharing=False,GRU=False,preTrain=False):
        self.READ_OPTION=READ_OPTION
        self.downstream=downstream
        self.sharing=sharing
        self.GRU=GRU
        self.preTrain=preTrain

class HyperParams:
    EMB_SIZE=None
    LAYER_DEPTH=None
    HIDDEN_SIZE=None
    NUM_EPOCHS=None
    STOP=None
    SEPARATOR=None

    def __init__(self,EMB_SIZE=50,LAYER_DEPTH=1,HIDDEN_SIZE=100,NUM_EPOCHS=10,STOP=0,SEPARATOR=1):
        #Hyperparameter Definition
        self.EMB_SIZE=EMB_SIZE
        self.LAYER_DEPTH=LAYER_DEPTH
        self.HIDDEN_SIZE=HIDDEN_SIZE
        self.NUM_EPOCHS=NUM_EPOCHS
        self.STOP=STOP
        self.SEPARATOR=SEPARATOR

class Model:
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

        s_init=decoder.initial_state().set_s(final_state_reverse)
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

    def do_one_example(self,sentence_de,sentence_en):
        model=self.model
        encoder=self.encoder
        revcoder=self.revcoder
        decoder=self.decoder
        encoder_params=self.encoder_params
        decoder_params=self.decoder_params
        downstream=self.config.downstream
        GRU=self.config.GRU

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

        s_init=decoder.initial_state().set_s(final_state_reverse)
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



    def __init__(self,config,hyperParams):
        self.config=config
        self.hyperParams=hyperParams

        if self.config.READ_OPTION=="NORMAL":
            train_sentences_de,train_sentences_en,valid_sentences_de,valid_sentences_en,test_sentences_de,test_sentences_en,wids=readData.getData(trainingPoints=700,validPoints=400)
        elif self.config.READ_OPTION=="NORMALDISJOINT":
            train_sentences_de,train_sentences_en,valid_sentences_de,valid_sentences_en,test_sentences_de,test_sentences_en,wids=readData.getDataDisjoint(trainingPoints=500,validPoints=200)
        elif self.config.READ_OPTION=="KNIGHTHOLDOUT":
            train_sentences_de,train_sentences_en,valid_sentences_de,valid_sentences_en,test_sentences_de,test_sentences_en,wids=readData.getDataKnightHoldOut(trainingPoints=1000)
    
        self.train_sentences_de=train_sentences_de
        self.train_sentences_en=train_sentences_en
        self.valid_sentences_de=valid_sentences_de
        self.valid_sentences_en=valid_sentences_en
        self.test_sentences_de=test_sentences_de
        self.test_sentences_en=test_sentences_en
        self.wids=wids
        self.reverse_wids=readData.reverseDictionary(self.wids)

        print len(self.train_sentences_de)
        print len(self.train_sentences_en)

        print len(self.valid_sentences_de)
        print len(self.valid_sentences_en)

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

        self.encoder_params={}
        self.encoder_params["lookup"]=model.add_lookup_parameters((self.VOCAB_SIZE_DE,hyperParams.EMB_SIZE))

        self.decoder_params={}
        if config.sharing:
            self.decoder_params["lookup"]=self.encoder_params["lookup"]
        else:
            self.decoder_params["lookup"]=model.add_lookup_parameters((self.VOCAB_SIZE_EN,hyperParams.EMB_SIZE))

        if config.downstream:
            self.decoder_params["R"]=model.add_parameters((self.VOCAB_SIZE_EN,2*hyperParams.HIDDEN_SIZE))
        else:
            self.decoder_params["R"]=model.add_parameters((self.VOCAB_SIZE_EN,hyperParams.HIDDEN_SIZE))

        self.decoder_params["bias"]=model.add_parameters((self.VOCAB_SIZE_EN))

        if config.preTrain:
            self.encoder_params["R_DE"]=model.add_parameters((self.VOCAB_SIZE_DE,hyperParams.HIDDEN_SIZE))
            self.encoder_params["bias_DE"]=model.add_parameters((self.VOCAB_SIZE_DE))



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


    def train(self):
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


    def testOut(self,valid_sentences,verbose=True,originalFileName="originalWords.txt",outputFileName="outputWords.txt"):
        reverse_wids=self.reverse_wids

        originalWordFile=open(originalFileName,"w")
        outputWordFile=open(outputFileName,"w")
        import editdistance
        exactMatches=0
        editDistance=0.0
        

        for validSentenceId,validSentence in enumerate(valid_sentences):
            valid_sentence_de=validSentence[0]
            valid_sentence_en=validSentence[1]
            validLoss,valid_sentence_en_hat=self.greedyDecode(valid_sentence_de)

            originalWord="".join([reverse_wids[c] for c in valid_sentence_en[:-1]])
            outputWord="".join([reverse_wids[c] for c in valid_sentence_en_hat[:-1]])
    
            if originalWord==outputWord:
                exactMatches+=1

            editDistance+=editdistance.eval(originalWord,outputWord)
            
            if verbose:
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

if __name__=="__main__":
    READ_OPTION="NORMAL"
    preTrain=True
    downstream=True
    sharing=False
    GRU=False

    config=Config(READ_OPTION=READ_OPTION,downstream=downstream,sharing=sharing,GRU=GRU,preTrain=preTrain)
    hyperParams=HyperParams()

    predictor=Model(config,hyperParams)
    predictor.train()

    predictor.testOut(predictor.valid_sentences,verbose=True,originalFileName="originalWords.txt",outputFileName="outputWords.txt")
    predictor.testOut(predictor.test_sentences,verbose=False,originalFileName="originalWords.txt",outputFileName="outputWords.txt")

