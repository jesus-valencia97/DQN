# import the necessary libraries
import numpy as np
import random
from tensorflow.keras.layers import Dense
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
        self.stateDimension=1
        self.actionDimension=9
        self.replayBufferSize=300
        self.batchReplayBufferSize=100
        self.updateTargetNetworkPeriod=100
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
        model.add(Dense(128,input_dim=self.stateDimension,activation='relu'))
        model.add(Dense(56,activation='relu'))
        model.add(Dense(self.actionDimension,activation='linear'))
        model.compile(optimizer =  RMSprop(), loss = self.my_loss_fn, metrics = ['accuracy'])
        return model

    def trainingEpisodes(self):

        for indexEpisode in range(self.numberEpisodes):
            rewardsEpisode=[]
            statesEpisode = []
     
            print("Simulating episode {}".format(indexEpisode))
            # currentState = np.array([np.random.uniform(0,50)])
            currentState = np.array([1])
            terminated = False
            self.fistTrain = 0

            if indexEpisode>200:
                self.epsilon=0.997*self.epsilon

            while not terminated:
                action = self.selectAction(currentState,indexEpisode)
                (reward, nextState) = step(action, currentState)   
                rewardsEpisode.append(reward)
                statesEpisode.append(currentState)
                if nextState<=0 or nextState == currentState:
                    terminated = True
                self.replayBuffer.append((currentState,action,reward,nextState,terminated))
                self.trainNetwork()
                currentState=nextState
           
            print("\t Max of rewards {}".format(np.max(rewardsEpisode)))
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
            Qvalues=self.mainNetwork.predict(state.reshape(1,self.stateDimension))
            return np.random.choice(np.where(Qvalues[0,:]==np.max(Qvalues[0,:]))[0])
  
    
    def trainNetwork(self):


        if (len(self.replayBuffer)>self.batchReplayBufferSize):
            self.fistTrain += 1
            
            if self.fistTrain == 1:
                print('\t Fist train of main network...')

            randomSampleBatch=random.sample(self.replayBuffer, self.batchReplayBufferSize)
            currentStateBatch=np.zeros(shape=(self.batchReplayBufferSize,1))
            nextStateBatch=np.zeros(shape=(self.batchReplayBufferSize,1))            

            for index,tupleS in enumerate(randomSampleBatch):
                currentStateBatch[index,:]=tupleS[0]
                nextStateBatch[index,:]=tupleS[3]

            QnextStateTargetNetwork=self.targetNetwork.predict(nextStateBatch)
            QcurrentStateMainNetwork=self.mainNetwork.predict(currentStateBatch)
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
