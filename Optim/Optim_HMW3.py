#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 18:26:27 2018

@author: matthis
"""
import numpy as np
import matplotlib.pyplot as plt

def centering_step(Q,p,A,b,t,v0,eps):
    def objective(x):
        return(x.T.dot(Q.dot(x))*t + t*p.T.dot(x) - np.sum(np.log(b-A.dot(x))))
        
    def gradient(x):
        grad = []
        for k in range(x.shape[0]):
            grad.append(2*t*(Q.dot(x))[k] + t*p[k] + np.sum(A[:,k]/(b-A.dot(x))))
        return(np.array(grad))
        
    def hessian(x):
        hess = []
        for i in range(x.shape[0]):
            lign = []
            for j in range(x.shape[0]):
                lign.append(2*t*Q[i,j] + np.sum(A[:,i]*A[:,j]/(b-A.dot(x))**2))
            hess.append(lign)
        return(np.array(hess))
    
    #v will contain all the succesive values of variables.
    v = []
    v.append(v0)
    grad = gradient(v[-1])
    H = hessian(v[-1])
    alpha = 0.5
    beta = 0.7
    #gradient descent
    while(grad.T.dot(np.linalg.inv(H)).dot(grad) > 2*eps):
        delta = np.linalg.inv(H).dot(grad)
        #backtracking line search
        while(objective(v[-1] - delta) >= objective(v[-1]) - alpha*grad.T.dot(delta)):
            delta*=beta
            
        v.append(v[-1] - delta)
        grad = gradient(v[-1])
        H = hessian(v[-1])
    return(v)
    
def barr_method(Q,p,A,b,v0,mu,eps):
    t = 1
    m = b.shape[0] 
    #list_v will contain the successive solution of the centering step.
    list_v = [v0]
    while(m/t > eps):
        v = centering_step(Q,p,A,b,t,list_v[-1],eps)
        list_v.append(v[-1])
        t = mu*t
    return(list_v)
    
def f(x):
    return(x.T.dot(Q.dot(x)) + p.T.dot(x))

# Create data
n = 100
d = 150
Q = 0.5*np.eye(n)
v0 = np.zeros(n)
X = np.random.rand(n,d)*10
A = np.concatenate((X.T, -X.T))
b = 10*np.ones(2*d)
y = np.random.rand(n)*20
p=-y
mu = 15
eps = 1e-5

plt.figure()
plt.yscale("log")
for mu in [2,15,50,100]:
    v_list = barr_method(Q,p,A,b,v0,mu,eps)
    best_value = min(map(f, v_list))
    gap = []
    for value in v_list:
        gap.append(f(value) - best_value)
    print("mu = %d, best value: %f" %(mu, best_value))
    x = np.arange(len(gap))
    plt.step(x,gap, label = str(mu))
    plt.legend()

plt.title('Evolution of the gap for different values of µ, n=%d, d=%d' %(n,d))  
plt.xlabel('number of iterations')
plt.ylabel('f(v)-f*')
plt.show()








