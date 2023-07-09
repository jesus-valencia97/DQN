# import the necessary libraries
import numpy as np
import random
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop, Adam
from collections import deque 
from tensorflow import gather_nd
from tensorflow.keras.losses import mean_squared_error, BinaryCrossentropy
from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.callbacks import Callback

from keras.callbacks import CSVLogger

from sklearn.metrics import mean_squared_error as skmse

import keras
import keras.backend as K

def recall_m(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    Positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = TP / (Positives+K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = TP / (Pred_Positives+K.epsilon())
    return precision


def f1(y_true, y_pred):
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))


METRICS = [
    #   keras.metrics.TruePositives(name='tp'),
    #   keras.metrics.FalsePositives(name='fp'),
    #   keras.metrics.TrueNegatives(name='tn'),
    #   keras.metrics.FalseNegatives(name='fn'), 
    #   keras.metrics.Precision(name='precision'),
    #   keras.metrics.Recall(name='recall'),
    recall_m,
    precision_m,
    f1
]


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        
    def on_train_end(self, logs=None):
        printt(f'\t\t Start: {self.losses[0]} \t End: {self.losses[-1]}')
       
            


def printt(expr):
    with open("train_log.txt","a") as f:
        print(expr, file = f)
#         printt(expr)

class DeepQLearning:
    
    def __init__(self,gamma,epsilon,numberEpisodes):
        
        self.gamma=gamma
        self.epsilon=epsilon
        self.numberEpisodes=numberEpisodes
        self.stateDimension=X_train.shape[1]
        self.actionDimension=len(actions)
        self.replayBufferSize=120
        self.batchReplayBufferSize=30
        self.updateTargetNetworkPeriod=30
        self.counterUpdateTargetNetwork=0
        self.sumRewardsEpisode=[]
        self.replayBuffer=deque(maxlen=self.replayBufferSize)
        self.mainNetwork=self.createNetwork()
        self.targetNetwork=self.createNetwork()
        self.targetNetwork.set_weights(self.mainNetwork.get_weights())
        self.actionsAppend=[]

    
    def my_loss_fn(self,y_true, y_pred):

        s1,s2=y_true.shape
        indices=np.zeros(shape=(s1,2))
        indices[:,0]=np.arange(s1)
        indices[:,1]=self.actionsAppend

        printt(indices)
        tf.printt(y_true,y_pred)
        tf.printt(gather_nd(y_true,indices=indices.astype(int)), gather_nd(y_pred,indices=indices.astype(int)))
        printt('-')
        loss = mean_squared_error(gather_nd(y_true,indices=indices.astype(int)), gather_nd(y_pred,indices=indices.astype(int)))
        tf.printt(loss)
        return loss    

    def cust_mse(self, y_true, y_pred):

        loss = mean_squared_error(y_true,y_pred) * self.actionDimension

        # tf.printt(loss)
        
        return loss
        
    def cust_binary(self, y_true, y_pred):
        
        bce = BinaryCrossentropy()
        loss = mean_squared_error(y_true,y_pred)
        printt(loss.shape)
        tf.printt(loss)
        
        # printt(loss)


        return loss


    def createNetwork(self):
        # DQN
        model=Sequential()
        model.add(Dense(500,input_dim=self.stateDimension,activation='relu'))
        model.add(Dense(500,input_dim=self.stateDimension,activation='relu'))
        model.add(Dense(self.actionDimension,activation='linear'))
        # model.compile(optimizer =  keras.optimizers.Adam(learning_rate=0.00025), loss = self.my_loss_fn)
        opt = Adam(lr=0.00025)
        model.compile(optimizer = opt, loss = self.cust_mse)
        return model

    def buildNetwork(self):

        model=Sequential()
        model.add(Dense(500,input_dim=self.stateDimension,activation='relu'))
        model.add(Dense(64,input_dim=self.stateDimension,activation='tanh'))
        # model.add(Dropout(0.1))
        model.add(Dense(20,activation='relu'))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(optimizer = "adam", loss = self.cust_binary, metrics=METRICS)
        return model


    def trainingEpisodes(self):

        for indexEpisode in range(self.numberEpisodes):

            X_train_ = X_train.copy()
            y_train_ = y_train.copy()
            rng_idx = pd.Series(y_train).sample(len(y_train)).index
            X_train_ = X_train_[rng_idx]
            y_train_ = y_train_[rng_idx]
            M_idx, m_idx = pd.DataFrame(y_train_).groupby(0).apply(lambda x: x.index.values)

            rewardsEpisode=[]
            statesEpisode = []
            printt("Simulating episode {}".format(indexEpisode))
            
            i=0
            currentState = X_train_[i,:]
            # currentState = np.array([0.5])
            terminated = False
            self.fistTrain = 0

            if indexEpisode>25:
                self.epsilon=0.95*self.epsilon

            while not terminated:
                action = self.selectAction(currentState,indexEpisode)
                # printt(action)
                (reward, nextState, terminated, i) = step(action, i, M_idx, m_idx, X_train_, y_train_)   
                rewardsEpisode.append(reward)
                statesEpisode.append(currentState)
                self.replayBuffer.append((currentState,action,reward,nextState,terminated))
                # printt(len(self.replayBuffer))
                self.trainNetwork()
                currentState=nextState

            printt(f'\t First 1 index: {m_idx[:10]}')
            printt(f'\t Reached sample: {i}')
            printt(f'\t Reward distribution: {np.unique(rewardsEpisode, return_counts = True)}')
            printt("\t Sum of rewards {}".format(np.sum(rewardsEpisode)))

    def selectAction(self,state,index):
        import numpy as np
        
        if index<1:
            return np.random.choice(self.actionDimension) 
            
        randomNumber=np.random.random()
        
        # if index>25:
        #     self.epsilon=0.99*self.epsilon
        
        if randomNumber < self.epsilon:
            # printt('random')
            # printt(self.epsilon)
            return np.random.choice(self.actionDimension)           
        
        else:
            # printt('Explotation...')
            Qvalues=self.mainNetwork.predict(state.reshape(1,self.stateDimension), verbose=0)
            # return np.random.choice(np.where(Qvalues[0,:]==np.max(Qvalues[0,:]))[0])
            return np.argmax(Qvalues)
  
    
    def trainNetwork(self):


        if (len(self.replayBuffer)>self.batchReplayBufferSize):

            # printt('\t train...')
            self.fistTrain += 1
            
            if self.fistTrain == 1:
                printt('\t First train of main network...')

            randomSampleBatch=random.sample(self.replayBuffer, self.batchReplayBufferSize)
            currentStateBatch=np.zeros(shape=(self.batchReplayBufferSize,self.stateDimension))
            nextStateBatch=np.zeros(shape=(self.batchReplayBufferSize,self.stateDimension))            

            for index,tupleS in enumerate(randomSampleBatch):
                currentStateBatch[index,:]=tupleS[0]
                nextStateBatch[index,:]=tupleS[3]

            QnextStateTargetNetwork=self.targetNetwork.predict(nextStateBatch, verbose=0)
            QcurrentStateMainNetwork=self.mainNetwork.predict(currentStateBatch, verbose = 0)

            inputNetwork=currentStateBatch
            outputNetwork=np.zeros(shape=(self.batchReplayBufferSize,self.actionDimension))

            for index,(currentState,action,reward,nextState,terminated) in enumerate(randomSampleBatch):
                
                if terminated:
                    y=reward                    
                else:
                    y=reward+self.gamma*np.max(QnextStateTargetNetwork[index])

                outputNetwork[index]=QcurrentStateMainNetwork[index]
                outputNetwork[index,action]=y


            # csv_logger = CSVLogger('train_log.txt', append=True, separator='\t')
            loss_logger = LossHistory()

            self.mainNetwork.fit(inputNetwork,outputNetwork,batch_size = self.batchReplayBufferSize, verbose=0,epochs=20, callbacks=[loss_logger])     
            self.counterUpdateTargetNetwork+=1  

            if (self.counterUpdateTargetNetwork>(self.updateTargetNetworkPeriod-1)):
                self.targetNetwork.set_weights(self.mainNetwork.get_weights())        
                printt("\t\t [UPDATED] Target network")
                printt(f"\t\t Current epsilon: {self.epsilon}")
                self.counterUpdateTargetNetwork=0
