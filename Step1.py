import random
import numpy as np
from Env import Env

EPISODES = 60000

#step 1
#only for get data

if __name__ == "__main__":
    env = Env()
    # get size of state and action from environment
    state_size = env.stateSize
    action_size = env.actionSize
    
    records = np.array([])
    for e in range(EPISODES):
        done = 0
        state = env.reset()

        record = np.array([])
        record = np.append(record,state) # s1
        while not done:
            actions = env.getActions(state.tolist()) #get actions can be done in this state
            index = random.randrange(len(actions)) # all random in step1
            action = actions[index]
            record = np.append(record,action) #a1
            next_state, reward, done, info = env.step(action)
            if done:
                record = np.append(record,reward) #r,only record reward in the end
                records = np.append(records,record)
            else:
                state = next_state
                record = np.append(record,next_state) #s2

    #save data like this:
    # s1         a1          s2          a2          s3          a3          r
    #0:16        16:17       17:33       33:34       34:50       50:51       51:52

    npRecords = records.reshape(-1,state_size*3+3+1)
    np.save('step1.npy',npRecords)
    
