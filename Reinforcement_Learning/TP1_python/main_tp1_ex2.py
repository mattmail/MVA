from gridworld import GridWorld1
import gridrender as gui
import numpy as np
import time
import matplotlib.pyplot as plt

env = GridWorld1


def estimate_mu(env, n_iter):
    """
    inputs:
        env: Gridworld
        n_iter(int): number of iteration to estimate mu
    
    output:
        mu (array of size of the number of states): estimation of the distribution of the states
    """
    s = []
    for i in range(n_iter):
        s.append(env.reset())
    unique, counts = np.unique(s, return_counts = True)
    mu = counts/np.sum(counts)
    return(np.array(mu))

def monte_carlo(env, pol, gamma, n_iter, Tmax, mu):
    """
    inputs:
        env: Gridworld
        pol(list): a policy 
        gamma(float): discount factor
        n_iter(int): number of iterations
        T_max(int): number of maximal iteration in the while loop (represents the maximal size of a path)
        mu(array): estimation of the initial distribution of the states
        
    outputs:
        V(array): estimation of the value function of the optimal policy by Monte-Carlo estimation
        J(array): contains successive estimations of V.dot(mu)
    """
    Ns = np.zeros(env.n_states)
    rewards = np.zeros(env.n_states)
    J = []
    i = 0
    for e in range(n_iter):
        i += 1
        t = 0
        initial_state = env.reset()
        state = initial_state
        Ns[initial_state] += 1
        absorb = False
        while(absorb == False and t < Tmax):
            state, r, absorb = env.step(state, pol[state])
            rewards[initial_state] += gamma**t*r
            t += 1
        J.append(np.array(rewards/Ns).T.dot(mu))
    V = rewards / Ns
    return(V, np.array(J))

v_q4 = np.array([0.87691855, 0.92820033, 0.98817903, 0.00000000, 0.67106071, -0.99447514, 0.00000000, -0.82847001, -0.87691855,
        -0.93358351, -0.99447514])

def plot_J_difference(env,pol, n_iter, v_pi = v_q4):
    """
    Plots the evolution diference between J and J* through the iteration
    inputs:
        env: gridworld
        pol(list): a policy
        n_iter(int): number of iterations for monte-carlo estimation
        v_pi(array): optimal value of v
    """
    mu = estimate_mu(env, 2000)
    V, J = monte_carlo(env, pol, 0.95, n_iter, 50, mu)
    plt.figure()
    plt.plot(J-v_pi.dot(mu))
    plt.xlabel("Iterations")
    plt.ylabel("Jn - Jπ")
    plt.title('Evolution of Jn - Jπ')
    plt.legend()
    plt.show()
    
pol = [1, 2, 0, 0, 1, 1, 0, 0, 0, 0, 3]
plot_J_difference(env, pol, 20000)


# Work to do: Q5
################################################################################
v_opt = [0.87691855, 0.92820033, 0.98817903, 0.00000000, 0.82369294, 0.92820033, 0.00000000, 0.77818504, 0.82369294,
         0.87691855, 0.82847001]

def Q_learning(env, n_iter, eps, gamma, Tmax=50, render= False, render_policy = False):
    """
    inputs:
        env: gridworld
        n_iter(int): number of iterations
        eps(float between 0 and 1): probability of taking the optimal policy at each step
        gamma(float): discoutn factor
        Tmax(int): maximal iterations in the while loop (represents the maximal size of a path)
        render(bool): if True, vizualises all the states taken by the algorithm. Be careful, 
                      do not lauch it with a high value of n_iter, it would take time to get the final result.
        render_policy(bool): if True, plots the final policy.
        
    outputs:
        policy(array): the final policy
        V_list(array of array): the successive values of V
        rewards(array): the successive values of the cumulative reward
    """
    Q = np.zeros((env.n_states, len(env.action_names)))
    alpha = np.ones((env.n_states, len(env.action_names)))
    V_list = []
    cumulated_reward = 0
    rewards = []
    env.render = render
    fps = 10
    for i in range(n_iter):
        t = 0
        state = env.reset()
        absorb = False
        while(absorb == False and t < Tmax):
            # choose the best policy with probability 1-eps.
            if(np.random.choice([0,1], p=[eps, 1-eps]) == 1):
                action = env.state_actions[state][np.argmax(Q[state][env.state_actions[state]])]
            else:
                action = np.random.choice(env.state_actions[state])
            
            new_state, r, absorb = env.step(state, action)
            delta = r + gamma*np.max(Q[new_state]) - Q[state, action]
            cumulated_reward += r
            Q[state, action] += delta/alpha[state, action]
            alpha[state, action] += 1
            t += 1
            state = new_state
            if(render):
                time.sleep(1/fps)
        rewards.append(cumulated_reward)
        V_list.append(np.max(Q, axis=1))
    policy = np.argmax(Q, axis = 1)
    if(render_policy):
        gui.render_policy(env, policy)
    return(policy, np.array(V_list), np.array(rewards))


def plot_convergence(env, n_iter, eps, v_pi = v_opt):
    """
    plots the evolution of ||v* - vπ|| and the cumulative reward.
    input:
        env: gridworld
        n_iter(int): number of iteration in Q_learning
        eps(float between 0 and 1): probability of taking the optimal policy at each step
        v_pi(array): optimal value of v
    """
    pol, V_list, rewards = Q_learning(env, n_iter, eps, 0.95)
    plt.figure()
    plt.plot(np.max(np.abs(V_list-v_opt), axis = 1 ))
    plt.xlabel('Episodes')
    plt.ylabel('||v* - vπ||')
    plt.title('Evolution of ||v* - vπ||')
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Total rewards')
    plt.title('Reward cumulated over the episodes')
    plt.legend()
    plt.show()
    
plot_convergence(env, 10000, 0.1)

    

    
