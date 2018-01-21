import numpy as np

def random_maze(N, r):
    maze_temp = np.ones((N, N))
    for i in range(0,N):
        for j in range(0,N):
            if np.random.uniform(low=0.0, high=1.0, size=None) < 0.01 * r:
                maze_temp[i, j] = 0
    return maze_temp

def initialize_q_matrix( N, actions ):
    q_matrix=np.zeros((N*N,actions))
    q_matrix[0:N,0]=-np.inf
    q_matrix[N * (N - 1): N * N, 3] = -np.inf
    q_matrix[0, 2] = -np.inf
    for i in range(0,N):
        q_matrix[i * N - 1, 1] = -np.inf
        q_matrix[i * N , 2] = -np.inf
    q_matrix = q_matrix[0:N*N,:]
    return q_matrix

def state2coordinate( state,N ):
    a=np.fix(state/N)
    b=np.mod(state,N)
    return np.int(a),np.int(b)

def state_environment_control(maze,N):
    state_environment=np.zeros((N*N,4))
    state=maze.reshape(N*N,1)
    for i in range(0, N * N):
        if i - N < 0:
            state_environment[i, 0] = 0
        else:
            state_environment[i, 0] = state[i - N]

        if np.mod(i, N+1) == 0:
            state_environment[i, 1] = 0
        else:
            state_environment[i, 1] = state[i + 1]
        if np.mod(i, N) == 0:
            state_environment[i, 2] = 0
        else:
            state_environment[i, 2] = state[i - 1]
        if i + N > N * N - 1:
            state_environment[i, 3] = 0
        else:
            state_environment[i, 3] = state[i + N]
    return state,state_environment




iteration = 50
N = 10
r = 20
gamma = 0.75
learning_rate = 0.75
action_number = 4
action_choices = [-N, 1, -1, N]
initial_state = 0
goal_state = N * N -1
maze = random_maze(N, r)
maze[0, 0] = 1
maze[N-1,N-1]=1
state_number=[]
for i in range(0, N*N):
    state_number.append(i)
state_number=np.reshape(state_number,(N*N,1))
q_matrix = initialize_q_matrix(N, action_number)
[state,state_environment]=state_environment_control(maze,N)
print (maze)
for episode in range(0,iteration):
    print('Iteration number: ', str(episode))
    instance = []
    instance.append(initial_state)
    current_state = initial_state
    while True:
        if current_state == goal_state:
            break

        max_val = max(q_matrix[current_state,:])
        l=q_matrix[current_state,:]
        index=list(l).index(max_val)
        chosen_action = index

        next_state = current_state + action_choices[chosen_action]
        [a, b] = state2coordinate(next_state, N)
        if maze[a, b] == 0:
            reward = -10
        elif (next_state == goal_state):
            reward = 20
        else:
            reward = -1

        q_matrix[current_state, chosen_action] = q_matrix[current_state, chosen_action] + learning_rate * (reward + gamma * max(q_matrix[next_state,:])-q_matrix[current_state, chosen_action])

        if maze[a, b] == 1:
            current_state = next_state
        instance.append(current_state)

for m in instance:
    [a, b] = state2coordinate(m, N)


print(q_matrix)
print(maze)
print(instance)
#print(state)
#print(state_environment)
#print(state_number)
#input=np.concatenate((state_number,state,state_environment),axis=1)
#input=input.transpose()
input=state_number
print(input)

import random


from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam

model=Sequential()
model.add(Dense(8,input_shape=(1,),activation='relu'))
#model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='linear'))
model.compile(optimizer=Adam(lr=0.01),loss='mse')

memory=[]
max_memory=500
discount=0.80

def remember(experience):
    memory.append(experience)
    if len(memory) > max_memory:
        del memory[0]

def update_Q(Q,current_state):
    for a in range(4):
        next_state = current_state[0] + action_choices[a]
        if not (0 <= next_state <= 99):
            Q[a]=-np.inf
        elif ((np.mod(current_state[0],10)==9 and next_state-current_state[0]==1) or (np.mod(current_state[0],10)==0 and next_state-current_state[0]==-1)):
            Q[a]=-np.inf


def get_batch(model, batch_size=32):
    len_memory = len(memory)

    inputs = np.zeros((min(len_memory, batch_size), 1))
    targets = np.zeros((inputs.shape[0], 4))

    for i, idx in enumerate(np.random.randint(0, len_memory,size=inputs.shape[0])):
        state, action, reward, next_state = memory[idx]
        inputs[i:i + 1] = state

        targets[i] = model.predict(state)[0]
        Q = model.predict(next_state)[0]
        update_Q(Q,next_state)
        if state==goal_state:  # if game_over is True
            targets[i, action] = reward
        else:
            targets[i, action] = reward + discount * np.amax(Q)
    return inputs, targets

epoch=50

current_epsilon = 1.
exploration_rate = .999
epsilon_min = .1

instance=[]
instance_last=[]
wincount=0
for episode in range(epoch):
    loss=0
    print('Iteration number: ', str(episode))
    instance = []
    #instance_last=[]
    instance.append(initial_state)
    current_state = initial_state
    current_state=np.array([current_state])
    for step in range(200):
        if current_state == np.array([goal_state]):
            wincount+=1
            instance_last=instance
            break

        if np.random.rand() <= current_epsilon:
            next_hidden_state=-1
            while not ((0<=next_hidden_state<=99) and not(np.mod(current_state[0],10)==9 and next_hidden_state-current_state[0]==1) and not(np.mod(current_state[0],10)==0 and next_hidden_state-current_state[0]==-1)):
                chosen_action = np.random.randint(0, 4, size=1)[0]
                next_hidden_state=current_state[0] + action_choices[chosen_action]
            print('random taken')
        else:
            Q = model.predict(current_state)[0]
            update_Q(Q,current_state)
            chosen_action = np.argmax(Q)

        next_state = current_state[0] + action_choices[chosen_action]
        next_state=np.array([next_state])




        [a, b] = state2coordinate(next_state[0], N)
        if maze[a, b] == 0:
            step-=1

        else:
            print("wincount {} | Epsilon {:.4f} | State {} | Act {} | Next State {}".
                  format(wincount, current_epsilon, current_state, chosen_action, next_state))
            if next_state == np.array([goal_state]):
                reward = 0
            else:
                reward = -1

            remember((current_state,chosen_action,reward,next_state))
            inputs, targets = get_batch(model)
            loss += model.train_on_batch(inputs, targets)

            # Keep exploring
            if current_epsilon > epsilon_min:
                current_epsilon *= exploration_rate

            current_state = next_state
            instance.append(current_state[0])




    print(loss)
print(instance_last)
print(wincount)





'''
import tensorflow as tf

tf.reset_default_graph()

#These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[1,6],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([6,4],0,0.01))
Qout = tf.matmul(inputs1,W)
predict = tf.argmax(Qout,1)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

init = tf.global_variables_initializer()

# Set learning parameters
y = .99
e = 0.1
num_episodes = 2000
#create lists to contain total rewards and steps per episode
jList = []
rList = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        # Reset environment and get first new observation

        rAll = 0
        d = False
        j = 0
        # The Q-Network
        position=0
        while j < 99:
            j += 1
            # Choose an action by greedily (with e chance of random action) from the Q-network
            #a, allQ = sess.run([predict, Qout], feed_dict={inputs1: input[position]})
            #if np.random.rand(1) < e:
            #    a[0] = env.action_space.sample()
            # Get new state and reward from environment
            #s1, r, d, _ = env.step(a[0])
            # Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout, feed_dict={inputs1: input[position]})
            # Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            index=Q1.index(maxQ1)
            #targetQ = allQ
            targetQ= r + y * maxQ1
            # Train our network using target and predicted Q values
            _, W1 = sess.run([updateModel, W], feed_dict={inputs1: input[position], nextQ: targetQ})
            position += action_choices[index]




            rAll += r
            #s = s1
            if d == True:
                # Reduce chance of random action as we train the model.
                e = 1. / ((i / 50) + 10)
                break
        jList.append(j)
        rList.append(rAll)
    print ("Percent of succesful episodes: " + str(sum(rList) / num_episodes) + "%")'''
