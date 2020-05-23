import sys
import gym
from gym import wrappers
import time
from time import time  # just to have timestamps in the files
#import pylab
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from IPython import display
import copy
import scipy.io as sio

# Hyperparameters
num_episodes = 10000  # the length of each iteration
learning_rate = 0.1  # the step size to run the gradient ascent algorithm
gamma = 0.96  # the discount factor

# Create gym and seed numpy
env = gym.make('FrozenLake-v0')
# env = wrappers.Monitor(env, "./vid", video_callable=lambda episode_id: episode_id%100==0, force=True)
nA = env.action_space.n
nS = env.observation_space.n
np.random.seed(1)

# Init weight
w = np.random.rand(nS, nA)
x = np.zeros((nS, nA))
#x=x/(np.linalg.norm(x))
z = 0 * np.ones((nS, nA))
#z = z / (sum(sum(z)))#Q_sa = np.zeros((nS, nA, num_episodes))
#V_sa = np.zeros((nS, num_episodes))


# Keep stats for final print of graph
episode_rewards = []
success_rate = []
global_feature = np.zeros((1, nS))


def Calculate_Q(states_in, actions_in, gamma, rew):  # this is the Q function with reward sas rew defined in our paper
    virtual_Q = 0
    # virtual_reward = gamma ** kk
    for ii in range(len(states_in)):
        virtual_Q += rew[states_in[ii],actions_in[ii]] * gamma ** ii
        #virtual_Q +=  gamma ** ii # this was for the testing that lambda is correct
    return virtual_Q


# Our policy that maps state to action parameterized by w
def policy(state, w):
    z = np.dot(state, w)
    exp = np.exp(z)
    return exp / np.sum(exp)

# Vectorized softmax Jacobian
def softmax_grad(softmax):
    s = softmax.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)

# this is required for the algoroithm to work
def feature(current_S):
    local = np.zeros((1, nS))
    local[0, current_S] = 1
    return local

# this calculates the moving average of values with window
def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    smas = np.convolve(values, weights, 'valid')
    return smas

# lambda estimation for each state action pair
def Calculate_lambda_sa(w,N,gamma): # w= paramete theta, N is the number of sample to average
    lam = np.zeros((nS, nA)) # placeholder for lambda
    #lam = lam/(sum(sum(lam))) # we make sure the sum is (1/(1-gamma))
    check = np.zeros((nS, nA)) # dummy variable to avoid the repetition of the state action pairs in the trajectory
    for nn in range(N): # no of trajectories to estimate the lambda
        current_states_sub = []
        actions_sub = []
        episode_rewards_sub = []
        current_S_sub = env.reset()
        while True:
            state_sub = feature(current_S_sub) # get the state feature
            probs_sub = policy(state_sub, w)   # generate of the probabilities for different actions under current state and w
            action_sub = np.random.choice(nA, p=probs_sub[0]) # play the probabilistic action
            next_state, reward_sub, done, _ = env.step(action_sub) # # observe the next state and reward
            current_S_sub = next_state # store the next state
            current_states_sub.append(current_S_sub) # append the state
            actions_sub.append(action_sub) # append the action
            if done: # check if the episode ends
                # reward = -0.5
                # score += reward
                episode_rewards_sub.append(reward_sub) # end reward append
                break
        lam_local = np.zeros((nS, nA)) # local lambda placeholder for each episde
        for ii in range(len(current_states_sub)): # this loop select a state action pair
            if (check[current_states_sub[ii],actions_sub[ii]] == 0): # to make sure the state action pair is not repeated, if repeated, then we don't do the update becasue its already done
                for jj in range(len(current_states_sub)): # this loop runs through the trajectory
                    if (current_states_sub[jj] == current_states_sub[ii]) & (actions_sub[jj] == actions_sub[ii]): # check the current action pair to be sa
                        lam_local[current_states_sub[ii],actions_sub[ii]] += gamma**jj # update lambda for current state anda action
            check[current_states_sub[ii], actions_sub[ii]] == 1 # make this 1 to show that this particular state action pair is updated to avoid repetition
        lam_local=(1/(1-gamma))*lam_local/sum(sum(lam_local)) # this is to get the scaling correct of lambda
        lam += lam_local # take the sum over all trajectories
    return lam/N # calcualting the average values

# entropy function
def F_ent(lambda_SA,gamma):
    temp=0
    for ii in range(nS):
         for jj in range(nA):
            if (lambda_SA[ii,jj]>0): # this is to avoid the limiting case of lambda[ii,jj]==0 at which entropy is zero
                temp += (-1)*(1-gamma)*lambda_SA[ii,jj]*np.log((1-gamma)*lambda_SA[ii,jj])
    return temp

# this function is to calculate the finite difference based gradient of F
def grad_FD(w): # w is thew policy parameter as theta, delta is the differencing constant
    delta=0.0001
    N=10000
    temp=np.zeros((nS,nA)) # temporary variable
    w_1=w
    lambda_1 = Calculate_lambda_sa(w_1, N,gamma)  # for w_1--> lambda_1 --> F(lambda_1) ############# #
    lambda_2 = Calculate_lambda_sa(w_1, N,gamma) #
    ((F_ent(lambda_2,gamma) - F_ent(lambda_1,gamma)))
    per = np.random.normal(0, 1,[nS, nA])  # perturbation we add to w/theta
    per = per/ np.linalg.norm(per)
    # we need to evaluate the rate of change in F for each w_{sa}
    for ii in range(nS): # loop over states
        for jj in range(nA): # loop over actions
            print(ii)
            print(jj)
            w_2=w
            w_2[ii,jj] = w[ii,jj] + delta * (per[ii,jj])  # perturbed w/theta for the ii, jj variable
            lambda_2 = Calculate_lambda_sa(w_2, N,gamma) ## for perturbed w_2 --> lambda_2 --> F(lambda_2)
            # next we evaluates # (F(lambda_2)-F(lambda_1))/delta
            # temp[ii,jj]=(1/(delta))*(1/2)*((np.linalg.norm(lambda_1)**2-(np.linalg.norm(lambda_2))**2)) # we have F(lambda)=-(1/2)*\|lambda\|^2
            temp[ii,jj]=(1/((delta * (per[ii,jj]))))*((F_ent(lambda_2,gamma)-F_ent(lambda_1,gamma))) # we have F(lambda)=entropy
    return [temp, lambda_1] # this returns |S|X|A| gradient estimate


#################### the algorithm starts here #################
###############################################################
################################################################

# finite difference method to evaluate x_star (the true policy gradient estimate)
[x_star, lambda_sa] = grad_FD(w) # gradient evaluations using the finite difference method
x_star = x_star/np.linalg.norm(x_star) # make the gradient unit norm

num_episodes = 100000
episodes = []
scores = []
norms = []
avg_return = []
store_x = []
values_dot = []
values_norm = []
learning_rate = 0.001  # the step size to run the gradient ascent algorithm
#cc = 0.001  # the step size to run the gradient ascent algorithm

#x = np.random.rand(nS, nA)
x=-1*x_star
x = x/np.linalg.norm(x)
#z = (1/(1-gamma)) *(1/(nS*nA))* np.ones((nS, nA))
z = np.zeros((nS, nA))
#z = z / (sum(sum(z)))#Q_sa = np.zeros((nS, nA, num_episodes))
# The policy gradient estimation algorithm
for step in range(num_episodes):
    #learning_rate=cc/((step+1)**0)
    # now we have x for the current episode, we now compare with x_star using the following steps
    x_temp = np.reshape(x, (nS * nA, 1))  # reshaping from 16X4 to 64X1
    x_star_temp = np.reshape(x_star, (nS * nA, 1))  # # reshaping from 16X4 to 64X1
    value_dot = np.dot(x_temp.T, x_star_temp)  # dot  product to see if x is in the same direction to x_star
    values_dot.append(value_dot.item())  # store the dot product values
    # value_norm = np.linalg.norm(x-x_star)  # calculate norm distance between x and x_star
    value_norm = np.linalg.norm(x - x_star)  # calculate norm distance between x and x_star
    values_norm.append(value_norm.item())  # store the norm distance values
    store_x.append(np.linalg.norm(x))  # we store the norm of x
    current_S = env.reset()
    init_State = current_S
    current_states = []
    actions = []
    grads = []
    rewards = []
    z_collect = []
    reward_matrix = []
    # Keep track of game score to print
    score = 0
    while True:
        # Sample from policy and take action in environment
        state = feature(current_S)
        probs = policy(state, w)
        action = np.random.choice(nA, p=probs[0])
        next_state, reward, done, _ = env.step(action)
        # Compute gradient and save with reward in memory for our weight updates
        dsoftmax = softmax_grad(probs)[action, :]
        dlog = dsoftmax / probs[0, action]
        # grad = state.T.dot(dlog[None, :])
        grad = state.T.dot(dlog[None, :])
        # to store the state actions pairs visited during the trajectory
        current_states.append(current_S)
        actions.append(action)
        grads.append(grad)
        rewards.append(reward)
        z_collect.append(z[current_S,action])
        score += reward
        # update old state to the new state
        current_S = next_state
        if done: # check if the episode ends
            episode_rewards.append(reward)
            scores.append(score)
            episodes.append(step)
            print("episode: {}/{}, score: {}".format(step, num_episodes, score))
            # time.sleep(1)
            break
    # here we start the inner update of updating for each s_k a_k of Algorithm 1 in the ppaer
    for tt in range(len(grads)):
        # here we perform the x update of equation (13) in the paper
        Q_sa = Calculate_Q(current_states[tt:], actions[tt:], gamma, z)  # Q defined in Theorem 1 of the paper
        x += (learning_rate) * Q_sa * grads[tt]  # grad[tt] is actually the gradient of the  log of policy at current sa
        # now we project x on to the convex set \|x\|<=1
        if ((np.linalg.norm(x)) > 1):
            x = x / np.linalg.norm(x)  # we divide by x by its norm if its more than unity
        # here we perform the z update equation (14) of the paper
        indi = np.zeros((nS, nA)) # indicator matrix which is 1 at s_k,a_k
        indi[current_states[tt], actions[tt]] = 1
        # z += (-1 * learning_rate * lambda_sa) + (learning_rate *indi *(1/(1-gamma))*np.exp(-(1/(1-gamma))*((1-gamma)+z[current_states[tt], actions[tt]])))## for entropy
        z += (-1 * learning_rate * (1-gamma)*indi) + (learning_rate *indi *(1/(1-gamma))*np.exp((-1/(1-gamma))*((1-gamma)+z[current_states[tt], actions[tt]])))## for entropy
        #z += (-1 * learning_rate * (1/(1-gamma)) * indi) + (learning_rate *-1*indi *z[current_states[tt], actions[tt]])## for thr -(1/2)*\|lambda\|^2
        #z += (-1 * learning_rate/tt * lambda_sa) + (learning_rate/tt *-1*indi *z[current_states[tt], actions[tt]])## for thr -(1/2)*\|lambda\|^2
        # #z_grad = np.zeros((nS, nA))
        #z_grad[current_states[tt], actions[tt]] = -1*(rewards[tt]-z[current_states[tt], actions[tt]])
        #z += -1 * learning_rate * indi + learning_rate * (-1 * np.exp(-1 * (1 + z)))  # for the entropy function
# plot the dot pruct values
plt.figure(1)
plt.plot(episodes, values_dot, 'b')
plt.ylabel('<x,x_star>')
plt.xlabel('episode')
plt.grid(True)
# plot the norm distance values
plt.figure(2)
plt.plot(episodes, values_norm, 'b')
plt.ylabel('\|x-x_star|\|')
plt.xlabel('episode')
plt.grid(True)

# scores_ma = moving_average(scores, 10)
# plt.plot(episodes[0:len(scores_ma)], scores_ma, 'b')
#
# plt.imshow(avg_lambda[:,:,step-1], cmap='hot', interpolation='nearest')
# plt.show()

env.close()
env.env.close()

env.env.close()
