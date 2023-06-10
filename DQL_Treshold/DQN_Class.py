# import the necessary libraries
import numpy as np
import random
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from collections import deque 
from tensorflow import gather_nd
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import RMSprop

class DeepQLearning:
    
    def __init__(self,gamma,epsilon,numberEpisodes):
        
        self.gamma=gamma
        self.epsilon=epsilon
        self.numberEpisodes=numberEpisodes
        self.stateDimension=(2,2)
        self.actionDimension=len(actions)
        self.replayBufferSize=300
        self.batchReplayBufferSize=100
        self.updateTargetNetworkPeriod=50
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
        loss = mean_squared_error(gather_nd(y_true,indices=indices.astype(int)), gather_nd(y_pred,indices=indices.astype(int)))
        return loss    

    def createNetwork(self):
        model=Sequential()
        model.add(Input(shape=self.stateDimension))
        model.add(Dense(128,activation='relu'))
        model.add(Dense(56,activation='relu'))
        model.add(Flatten())
        model.add(Dense(self.actionDimension,activation='linear'))
        model.compile(optimizer = 'adam', loss = self.my_loss_fn)
        return model

    def trainingEpisodes(self):

        for indexEpisode in range(self.numberEpisodes):
            rewardsEpisode=[]
            statesEpisode = []
            print("Simulating episode {}".format(indexEpisode))
            
            s0 = [np.random.uniform(1,0),np.random.uniform(1,0)]
            s0 = [round_closest(s) for s in s0]
            r0 = [rew(s) for s in s0]
            currentState = np.array([s0,r0])
            # currentState = np.array([[s0[0],r0[0]],[s0[1],r0[1]]])
            terminated = False
            self.fistTrain = 0

            if indexEpisode>25:
                self.epsilon=0.98*self.epsilon

            while not terminated:
                action = self.selectAction(currentState,indexEpisode)
                (reward, nextState, terminated) = step(action, currentState)   
                rewardsEpisode.append(reward)
                statesEpisode.append(currentState[0][0])
                self.replayBuffer.append((currentState,action,reward,nextState,terminated))
                self.trainNetwork()
                currentState=nextState
           
            print("\t Max of rewards {}".format(np.sum(rewardsEpisode)))
            print("\t Min state {}".format(np.min(statesEpisode)))             
            print("\t Max state {}".format(np.max(statesEpisode)))    

    def selectAction(self,state,index):
        import numpy as np
        
        if index<1:
            return np.random.choice(self.actionDimension) 
            
        randomNumber=np.random.random()
        
        # if index>25:
        #     self.epsilon=0.99*self.epsilon
        
        if randomNumber < self.epsilon:
            # print('random')
            # print(self.epsilon)
            return np.random.choice(self.actionDimension)           
        
        else:
            # print('Explotation...')
            Qvalues=self.mainNetwork.predict(state.reshape(1,*self.stateDimension), verbose=0)
            # return np.random.choice(np.where(Qvalues[0,:]==np.max(Qvalues[0,:]))[0])
            return np.argmax(Qvalues)
  
    
    def trainNetwork(self):


        if (len(self.replayBuffer)>self.batchReplayBufferSize):
            self.fistTrain += 1
            
            if self.fistTrain == 1:
                print('\t Fist train of main network...')

            randomSampleBatch=random.sample(self.replayBuffer, self.batchReplayBufferSize)
            currentStateBatch=np.zeros(shape=(self.batchReplayBufferSize,*self.stateDimension))
            nextStateBatch=np.zeros(shape=(self.batchReplayBufferSize,*self.stateDimension))            

            for index,tupleS in enumerate(randomSampleBatch):
                currentStateBatch[index,:]=tupleS[0]
                nextStateBatch[index,:]=tupleS[3]

            QnextStateTargetNetwork=self.targetNetwork.predict(nextStateBatch, verbose=0)
            QcurrentStateMainNetwork=self.mainNetwork.predict(currentStateBatch, verbose = 0)
            inputNetwork=currentStateBatch
            outputNetwork=np.zeros(shape=(self.batchReplayBufferSize,self.actionDimension))
            self.actionsAppend=[]  

            for index,(currentState,action,reward,nextState,terminated) in enumerate(randomSampleBatch):
                
                if terminated:
                    y=reward                    
                else:
                    y=reward+self.gamma*np.max(QnextStateTargetNetwork[index])

                self.actionsAppend.append(action)
                outputNetwork[index]=QcurrentStateMainNetwork[index]
                outputNetwork[index,action]=y
            
            self.mainNetwork.fit(inputNetwork,outputNetwork,batch_size = self.batchReplayBufferSize, verbose=0,epochs=100)     
            self.counterUpdateTargetNetwork+=1  

            if (self.counterUpdateTargetNetwork>(self.updateTargetNetworkPeriod-1)):
                self.targetNetwork.set_weights(self.mainNetwork.get_weights())        
                print("\t\t [UPDATED] Target network")
                print(f"\t\t Current epsilon: {self.epsilon}")
                self.counterUpdateTargetNetwork=0
