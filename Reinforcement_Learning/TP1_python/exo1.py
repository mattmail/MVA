#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 13:05:21 2018

@author: matthis
"""
import numpy as np
import matplotlib.pyplot as plt
from time import time

class MDP:
    
    def __init__(self,X,A,p,r):
        self.X = np.array(X)
        self.A = np.array(A)
        self.p = np.array(p)
        self.r = np.array(r)
        
        
    def value_iteration(self,gamma, eps, plot = False, verbose = False):
        """
        inputs:
            - gamma(float): discount factor
            - eps(float): precision criterion
            - plot(bool): plot evolution of |v* - vπ|| if True
            - verbose(bool): print number of iterations if True
            
        Outputs:
            - policy(array): The optimal policy: size of the number of states
            - v_list[-1](array): the approximate value of v*: size of the number of states
        """
        v0 = np.random.rand(len(self.X))
        v_list = [v0]
        s = gamma*np.dot(self.p, v0)
        v = np.max(self.r + s, axis = 1)
        nb_iter =0
        while(np.linalg.norm(v-v0, ord = np.inf) > eps):
            nb_iter+=1
            v0 = v
            v_list.append(v0)
            s = gamma*np.dot(self.p, v0)
            v = np.max(self.r + s, axis = 1)
        pi = np.argmax(self.r + s, axis = 1 )
        
        if(plot):
            plt.figure()
            plt.xlabel('Iterations')
            plt.ylabel('||v*-vπ||')
            plt.title('Evolution of ||v* - vπ||')
            plt.plot(np.max(np.abs(v_list - v_list[-1]), axis = 1))
            plt.show()
        if(verbose):
            print('Number of iterations: %d' % nb_iter)
        policy = self.A[pi]
        return(policy, v_list[-1])
        
    def policy_iteration(self, gamma, pi0, verbose = False):
        """
        inputs:
            - gamma(float): discount factor
            - pi0(array): initialisation of the policy: size of the number of states
            - verbose(bool): print number of iterations if True
            
        Outputs:
            - policy: The optimal policy: size of the number of states
        """
        n = self.p.shape[0]
        P = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                P[i,j] = self.p[i,pi0[i],j]
                
        d = np.eye(n) - gamma*P
        v = np.linalg.inv(d).dot(np.diag(self.r[:,pi0]))
        pi = np.argmax(self.r + gamma*np.dot(self.p, v), axis = 1 )
        nb_iter = 0
        while not np.all(pi == pi0):
            nb_iter += 1
            pi0 = pi
            P = np.zeros((n,n))
            for i in range(n):
                for j in range(n):
                    P[i,j] = self.p[i,pi0[i],j]
            d = np.eye(self.p.shape[2]) - gamma*P
            v = np.linalg.inv(d).dot(np.diag(self.r[:,pi0]))
            pi = np.argmax(self.r + gamma*np.dot(self.p, v), axis = 1 )
            
        if(verbose):
            print('Number of iterations: %d' % nb_iter)
        policy = self.A[pi]
        return(policy)
        
########
X = ['s0','s1','s2']
A = ['a0','a1','a2']        
    
p = np.array([[[0.55,0.45,0],[0.3,0.7,0],[1,0,0]],
              [[1,0,0],[0,0.4,0.6],[0,1,0]],
              [[0,1,0],[0, 0.6,0.4], [0,0,1]]])

r = np.array([[0,0,5/100],[0,0,0],[0,1,0.9]])

mdp = MDP(X,A,p,r)
t1 = time()
print('Optimal policy: ',mdp.value_iteration(0.95, 1e-2)[0])
print('value iteration time: %f' %(time()-t1))
t2 = time()
print('Optimal policy: ',mdp.policy_iteration(0.95, np.array([0,0,0])))
print('policy iteration time: %f' %(time()-t2))

