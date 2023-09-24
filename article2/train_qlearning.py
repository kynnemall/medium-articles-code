#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 17:26:46 2023

@author: martin
"""

import os
import glob
import json
import random
import argparse
import warnings
import numba as nb
import numpy as np
from tqdm import tqdm
from game import NumbaBoard
from datetime import datetime
from time import perf_counter
from datetime import timedelta
from bayes_opt import BayesianOptimization
warnings.filterwarnings("ignore")

# import faulthandler
# faulthandler.enable()

def log(opt_score, alpha, gamma, epsilon, lengths, maxima, q_table, pbounds):
    """
    Saves a record of the run parameters, results, and the latest Q-table

    Returns
    -------
    None.

    """
    base = datetime.now().strftime("%Y%m%d_%H-%M-%S_")
    jsonfile = base + 'params_stats.json'
    pbounds_file = base + 'pbounds.json'
    qtable_fname = base + 'q-table.npy'
    moves_fname = base + 'move_counts.npy'
    rewards_fname = base + 'max_rewards.npy'
    
    log_dict = {
        'actions' : list(range(q_table.shape[1])),
        'alpha' : alpha,
        'gamma' : gamma,
        'epsilon' : epsilon,
        'epochs' : len(lengths),
        'optimization score' : opt_score,
        }
    
    # save files
    try:
        with open(jsonfile, "w") as f:
            json.dump(log_dict, f, indent=4)
        with open(pbounds_file, "w") as f:
            json.dump(pbounds, f, indent=4)
        np.save(qtable_fname, q_table)
        np.save(moves_fname, lengths)
        np.save(rewards_fname, maxima)
    except TypeError:
        print('Error logging data to JSON due to Numpy.int64')
       
def rl_loop(alpha, gamma, epsilon, n_actions=4, epochs=100000, 
            bayes_optim=False, reduction_factor=0.95, print_debug=False):
    
    env = NumbaBoard(n_actions)
    
    qtables = sorted(glob.glob('*_q-table.npy'))
    if qtables:
        q_table = np.load(qtables[-1])
    else:
        q_table = np.zeros((300, n_actions))
    
    maxima = np.zeros(epochs, dtype=np.int32)
    lengths = np.zeros(epochs, dtype=np.int32)
    
    if bayes_optim:
        gen = range(1, epochs + 1)
    else:
        gen = tqdm(range(1, epochs + 1))
    
    for i in gen:
        rewards, q_table = qlearn_iterate(env, alpha, gamma, epsilon, q_table, 
                                 print_debug)
        maxima[i-1] = int(max(rewards))
        lengths[i-1] = len(rewards)
        if i % 20000 == 0:
            alpha *= reduction_factor
            gamma *= reduction_factor
            epsilon *= reduction_factor
    
    if bayes_optim:
        return np.mean(maxima)
    else:
        return lengths, maxima, q_table

@nb.njit
def get_q_values(q_table, state):
    n = q_table.shape[1]
    res = np.zeros((n, n, n))
    for x in range(n):
        for y in range(n):
            for _ in range(n):
                res[x, y] = q_table[state[x, y]]
    return res

@nb.njit
def get_old_values(qtable, state, action):
    # function to accomplish the following:
    # old_value = q_table[state, action]
    
    n = qtable.shape[1]
    result = np.zeros((n, n), dtype=np.int32)
    res = np.zeros((n, n, n))
    
    for x in range(n):
        for y in range(n):
            idx = state[x, y]
            res[x, y] = qtable[idx]
    
    for i in range(n):
        result[:, i] = res[:, i, action]
    return result

@nb.njit
def update_q_table(qtable, state, action, new_value):
    # function to accomplish the following:
    # table[state, action] = new_value
    
    n = qtable.shape[1]
    copy = qtable.copy()
    for i in range(n):
        for j in range(n):
            copy[state[i, j], action] = new_value[i, j]
    return copy

@nb.njit
def qlearn_iterate(env, alpha, gamma, eps, q_table, print_debug=False):
    """
    

    Parameters
    ----------
    env : TYPE
        DESCRIPTION.
    alpha : TYPE
        DESCRIPTION.
    gamma : TYPE
        DESCRIPTION.
    eps : TYPE
        DESCRIPTION.
    q_table : TYPE
        DESCRIPTION.
    print_debug : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    rewards : TYPE
        DESCRIPTION.

    """
    
    state = env.reset()
    done = False
    rewards = []
    
    if print_debug:
        print('Starting game')
    while not done:
        # Exploit learned values
        qvals = get_q_values(q_table, state) # q_table[state]
        maxima = [q.max() for q in qvals]
        
        if random.uniform(0, 1) < eps:
            # Explore action space
            action = np.random.choice(env.actions)
        else:
            # use learned values
            # check if unique options greater than 
            unique = set(maxima)
            if maxima.count(max(maxima)) > 1:
                if print_debug:
                    print('Selecting from maxima')
                choices = np.array([i for i,v in enumerate(maxima) if v == max(unique)])
                if choices.size > 0:
                    action = np.random.choice(choices)
                else:
                    if print_debug:
                        print('Breaking wait loop and finishing game')
                    done = True
            else:
                action = np.argmax(np.array(maxima))
        
        if print_debug:
            print('Checking if action is playable')
        if action in env.possible_actions and env.possible_actions.size > 0:
            # evaluate the action
            next_state, reward, valid, done = env.step(action)
            rewards.append(reward)
            
            if print_debug:
                print(f'Action taken: {action}')
                print(f'Reward: {reward}')
                if done:
                    print(f'Done after {env.moves}')
                print('State')
                print(next_state)
            
            # measure quality of the action
            old_vals = get_old_values(q_table, state, action) # q_table[state, action]
            next_qvals = get_q_values(q_table, next_state)
            next_max = np.max(next_qvals) # q_table[next_state]
            
            new_value = (1 - alpha) * old_vals +\
                alpha * (reward + gamma * next_max)
            
            # update the Q-table and state
            q_table = update_q_table(
                q_table, state, action, new_value
                ) # q_table[state, action] = new_value
            
            state = next_state
            
        elif action not in env.possible_actions:
            # keep the while loop running
            idx = maxima.index(max(maxima))
            maxima[idx] = 0
            if print_debug:
                print(f'Action not in possible actions\t{action}')
                print('Reducing possible actions')
            
    return rewards, q_table

def black_box_function(alpha, gamma, epsilon):
    """
    Function to allow Bayesian optimization of Q-learning hyperparameters

    Parameters
    ----------
    alpha : float, optional
        Learning rate. The default is 0.9.
    gamma : float, optional
        Discount factor. Higher values place emphasis on long-term rewards,
        while lower values focus on short-term rewards. The default is 0.9.
    eps : float, optional
        Exploration rate. It determines chance of randomly sampling from
        the available actions. The default is 0.9.

    Returns
    -------
    mean_reward : float
        The average potential reward for the next move on the board

    """
        
    mean_reward = rl_loop(
        alpha, gamma, epsilon, epochs=1000, bayes_optim=True, print_debug=False
        )
    return mean_reward

def optimize(n_runs, n_games, a, g, e, rf, n_actions):
    """
    

    Parameters
    ----------
    n_runs : integer
        Numbers of rounds of optimization and training to run
    n_games : integer
        Number of games to be simulated

    Returns
    -------
    None.

    """
    
    # populate pbounds with parameters if value is 0
    pbounds = {'alpha': [0.5, 0.9], 
               'gamma': [0.5, 0.8], 
               'epsilon': [0.4, 0.85]}
    
    for n in range(n_runs):
        print(f'------------ Training round {n+1} of {n_runs} ------------')
        optimizer = BayesianOptimization(
            f=black_box_function, pbounds=pbounds, random_state=42,
            allow_duplicate_points=True
            )
        optimizer.maximize(init_points=10, n_iter=10)
        params = optimizer.max['params']
        a, g, e = params['alpha'], params['gamma'], params['epsilon']
        max_score = optimizer.max["target"]
        
        # run Q learning with optimized hyperparameters 
        lengths, maxima, q_table = rl_loop(
            a, g, e, epochs=n_games, reduction_factor=rf,
            n_actions=n_actions
            )
        
        # log the round
        assert q_table.max() != 0, f'Q-table has max. value of 0\n{q_table}'
        log(max_score, a, g, e, lengths, maxima, q_table, pbounds)

def test():
    env = NumbaBoard()
    state = env.reset()
    q_table = np.arange(40).reshape((10, 4))
    if np.allclose(get_q_values(q_table, state), q_table[state]):
        print('Success')
    else:
        print('Failed')
        
# env = NumbaBoard(3)
# _ = qlearn_iterate(env, 0.9, 0.9, 0.8, np.zeros((40, 3)), False)
# optimize(1, 1000, 0.8, 0.8, 0.8, 0.98, 3)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', 
                        help='Directory to store/load experimental data')
    parser.add_argument('n_runs', type=int,
                        help='How many rounds of optimization to perform')
    parser.add_argument('n_games', type=int,
                        help='How many games to run after each optimization')
    parser.add_argument('-a', type=float, default=0., nargs='?', 
                        help='Alpha parameter')
    parser.add_argument('-g', type=float,default=0., nargs='?', 
                        help='Gamma parameter')
    parser.add_argument('-e', type=float, default=0., nargs='?', 
                        help='Epsilon parameter')
    parser.add_argument('-rf', type=float, default=0.95, nargs='?', 
                        help='Reduction factor')
    parser.add_argument('-n_actions', type=int, default=4, nargs='?', 
                        help='Number of actions available')
    args = parser.parse_args()

    if not os.path.exists(args.directory):
        os.mkdir(args.directory)
    os.chdir(args.directory)

    start = perf_counter()
    optimize(args.n_runs, args.n_games, args.a, args.g, args.e, args.rf,
              args.n_actions)
    end = perf_counter()
    seconds = end - start
    time_fmt = timedelta(seconds=seconds)
    print(f'Finished {args.n_runs} of optimization and training with {args.n_games} in {time_fmt}')
