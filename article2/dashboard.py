#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 16:56:37 2023

@author: martin
"""

import os
import glob
import json
import numba as nb
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# page settings
st.set_page_config(layout="wide")

@nb.njit
def get_q_values(q_table, state):
    n = q_table.shape[1]
    res = np.zeros((n, n, n))
    for x in range(n):
        for y in range(n):
            for z in range(n):
                res[x, y] = q_table[state[x, y]]
    return res

def evaluate(q_table, board=False, print_=True, print_debug=False):

    env = NumbaBoard(n_actions=q_table.shape[1])
    if isinstance(board, bool):
        state = env.reset()
    else:
        state = board
        
    done = False
    rewards = []
    view_state = 2 ** state
    view_state[view_state == 1] = 0
    
    if print_:
        print('Starting state')
        print(view_state)
    
    while not done:

        # Exploit learned values
        qvals = get_q_values(q_table, state) # q_table[state]
        maxima = [q.max() for q in qvals]
            
        # while loop to select a doable action
        wait = True
        
        while wait:
            # check if unique options greater than 
            unique = set(maxima)
            if maxima.count(max(maxima)) > 1:
                choices = np.array([i for i,v in enumerate(maxima) if v == max(unique)])
                if choices.size > 0:
                    action = np.random.choice(choices)
                else:
                    wait = False
                    done = True
                    break
            else:
                action = np.argmax(maxima)
            
            if action in env.possible_actions and env.possible_actions.size > 0:
                # print(f'\nPossible actions: {env.possible_actions}')
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
                
                # update state
                state = next_state
                # print(f'Score\t{env.score}')
                wait = False
                
            elif action not in env.possible_actions:
                # keep the while loop running
                idx = maxima.index(max(maxima))
                maxima[idx] = 0
                # if max(maxima) == 0:
                #     done = True
                #     wait = False
            
    view_state = 2 ** state
    view_state[view_state == 1] = 0
    if print_:
        print('Final state')
        print(view_state)
        print(f'Number of moves: {env.moves}')
        print(f'Score: {env.score}')
        
    return env

# copy and paste spec and NumbaBoard Jitclass below

spec = [
        ('score', nb.int32),
        ('moves', nb.int32),
        ('n_actions', nb.uint8),
        ('possible_actions', nb.uint8[:]),
        ('arr', nb.uint8[:]),
        ('actions', nb.uint8[:]),
        ('preparing', nb.b1),
        ('action', nb.uint8),
        ('board', nb.int32[:, :]),
        ('lost', nb.b1),
        ('valid', nb.b1),
        ('changed', nb.b1),
        ('new_board', nb.int32[:, :]),
        ('rotated_board', nb.int32[:, :]),
        ('after', nb.int32[:, :]),
        ('new_col', nb.int32[:]),
        ('rot_board', nb.int32[:, :]),
        ('result', nb.int32),
        ('col', nb.int32[:]),
        ('j', nb.uint8),
        ('i', nb.uint8),
        ('val', nb.int32),
        ('options', nb.b1),
        ('done', nb.b1),
        ('k', nb.uint8),
        ('previous', nb.int32),
        ('ep_return', nb.int64),
        ('potential', nb.int64[:]),
        ('reward', nb.int64),
        ('log2_res', nb.uint8[:, :]),
        ]
        
@nb.experimental.jitclass(spec)
class NumbaBoard:

    def __init__(self, n_actions=4):
        self.score = 0
        self.moves = 0
        self.n_actions = n_actions
        self.actions = np.arange(n_actions).astype(np.uint8)
        self.possible_actions = self.actions.copy()
        # self.num_direction = {'L' : 0, 'U' : 1, 'R' : 2, 'D' : 3}

        preparing = True
        while preparing:
            self.board = np.zeros((4, 4), dtype=np.int32)
            for _ in range(2):
                self.fill_cell()
            if self.board.sum() == 4:
                preparing = False
        
    def fill_cell(self):
        i, j = (self.board == 0).nonzero()
        if i.size != 0:
            high = i.size - 1 if i.size > 1 else 1
            rnd = np.random.randint(0, high)
            self.board[i[rnd], j[rnd]] = 2 * ((np.random.random() > .9) + 1)
            
    @staticmethod
    def move_left(col):
        new_col = np.zeros(4, dtype=np.int32)
        j = 0
        previous = -1
        result = 0
        for i in range(4):
            val = col[i]
            if val != 0: # number different from zero
                if previous == -1:
                    previous = val
                else:
                    if previous == val:
                        new_col[j] = 2 * val
                        result += new_col[j]
                        j += 1
                        previous = -1
                    else:
                        new_col[j] = previous
                        j += 1
                        previous = val
        if previous != -1:
            new_col[j] = previous
        return new_col, result
        
    def move(self, k):
        rotated_board = np.rot90(self.board, k)
        score = 0
        new_board = np.zeros((4,4), dtype=np.int32)
        for i,col in enumerate(rotated_board):
            new_col, result = self.move_left(col)
            score += result
            new_board[i] = new_col
        rot_board = np.rot90(new_board, -k)
        return rot_board, score

    def check_options(self):
        """
        Check if playable moves remain

        Returns
        -------
        options : Boolean
            Whether playable moves remain.

        """
        options = False
        before = self.board.copy()
        for k in range(self.n_actions):
            after, score = self.move(k)
            
            # if before and after are different,
            # return True, meaning there are playable actions
            if not np.array_equal(before, after):
                options = True
                break
        return options

    def evaluate_action(self, k):
        """
        Evaluate how the move k affects the state of the board

        Parameters
        ----------
        k : INTEGER
            index of the move.

        Returns
        -------
        score : INTEGER
            Increase in player score.
        lost : BOOLEAN
            Whether the player has lost the game with this move.

        """
        
        lost = False
        valid = False
        after, score = self.move(k)
        if not np.array_equal(self.board, after):
            valid = True
            self.board = after
            self.score += score
            self.fill_cell()
            self.moves += 1
            
            # check if options available
            options = self.check_options()
            if not options:
                lost = True
        return score, lost, valid
                
    def evaluate_next_actions(self):
        """
        Evaluate whether there are playable moves left

        Returns
        -------
        potential : LIST of integers
            Possible scores resulting from the playable moves.

        """
        potential = []
        arr = []
        for a in self.actions:
            rot_board, score = self.move(a)
            changed = not np.array_equal(rot_board, self.board)
            if changed:
                arr.append(a)
            potential.append(score)
        self.possible_actions = np.array(arr).astype(np.uint8)
        return potential
                
    # reset and step functions required by OpenAI Gym
    def reset(self):
        self.__init__()
        self.ep_return  = 0
        log2_res = np.where(self.board != 0, 
                            np.log2(self.board), 
                            0).astype(np.uint8)
        return log2_res
    
    def step(self, action):
        # call the action and get the score and outcome of that action
        score, done, valid = self.evaluate_action(action)
        
        if valid:
            # look one step ahead at each possible option
            reward_list = self.evaluate_next_actions()
            reward = max(reward_list)
        else:
            reward = 0
        
        # Increment the episodic return
        self.ep_return += 1
        log2_res = np.where(self.board != 0, 
                            np.log2(self.board), 
                            0).astype(np.uint8)
        return log2_res, reward, valid, done

# spec and Jitclass above

@st.cache_data
def load_data(folder):
    reward_files = [s for s in sorted(glob.glob(f'{folder}/*rewards.npy'))]
    count_files = [s for s in sorted(glob.glob(f'{folder}/*counts.npy'))]
    dfs = []

    # process all JSON files into one DataFrame
    for i,r in enumerate(reward_files):
        moves = np.load(count_files[i])
        rewards = np.load(r)
        n_moves = pd.Series(moves, name='n_moves')
        max_rewards = pd.Series(rewards, name='max_rewards')
        df = pd.concat([n_moves, max_rewards], axis=1)
        df['Optimization\nRound'] = i + 1
        dfs.append(df)
    df = pd.concat(dfs)
    df['rewards_per_move'] = df['max_rewards'] / df['n_moves']

    return df

@st.cache_data
def load_params(folder):
    data = []
    param_files = [s for s in sorted(glob.glob(f'{folder}/*params_stats.json'))]
    
    for i,file in enumerate(param_files, 1):
        with open(file, 'r') as f:
            d = json.load(f)
        d['actions'] = len(d['actions'])
        d['Optimization\nRound'] = i
        data.append(d)
        
    data = pd.DataFrame(data)
    return data

@st.cache_data
def load_last_qtable(folder):
    qfile = sorted(glob.glob(f'{folder}/*q-table.npy'))[-1]
    qtable = np.load(qfile)
    return qtable

def plot_params(params):
    data = params.melt(id_vars=['Optimization\nRound', 'optimization score'], 
                       value_vars=['alpha', 'gamma', 'epsilon'],
                       var_name='Hyperparameter',
                       value_name='Value')
    fig1 = px.line(data, x='Optimization\nRound', y='Value', 
                  color='Hyperparameter'
                  )
    fig2 = px.line(params, x='Optimization\nRound', y='optimization score')
    
    return fig1, fig2
    
def plot_rewards(data):
    percs = []
    rounds = np.arange(1, data['Optimization\nRound'].max() + 1)
    for n in rounds:
        sub = data[data['Optimization\nRound'] == n]
        q75, q25 = np.percentile(sub['max_rewards'], [75 ,25])
        iqr = q75 - q25
        over_iqr = sub['max_rewards'] > (iqr * 1.5 + q75)
        perc_over = np.round(over_iqr.mean() * 100, 1)
        percs.append(perc_over)
        
    fig1 = px.bar(x=rounds, y=percs)
    fig1.update_layout(xaxis_title='Optimization Round',
                      yaxis_title='% Scores > 1.5 times IQR',
                      )
    summary = data.groupby('Optimization\nRound')['max_rewards'].describe()

    return fig1, summary

@st.cache_data
def get_table(n=0):
    board = NumbaBoard()
    table = board.board
    return table

def show_board(board):
    fig = px.imshow(table)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    
    for i in range(4):
        for j in range(4):
            fig.add_annotation(x=i, y=j,
            text=str(table[j, i]),
            showarrow=False)
            
    return fig

def print_state():
    print(st.session_state)

def simulate(board, folder, params):
    qfiles = [s for s in sorted(glob.glob(f'{folder}/*q-table.npy'))]
    moves_mean = pd.Series(np.arange(params.shape[0]), name='# Moves')
    scores_mean = pd.Series(np.arange(params.shape[0]), name='Score')
    moves_std = pd.Series(np.arange(params.shape[0]), name='moves_std')
    scores_std = pd.Series(np.arange(params.shape[0]), name='score_std')
    contains_2048 = pd.Series(np.zeros(params.shape[0]), name='2048?')
    
    for i,qfile in enumerate(qfiles):
        qtable = np.load(qfile)
        qscores = np.ones(5)
        qmoves = np.ones(5)
        q2048 = np.zeros(5)
        for n in range(5):
            env = evaluate(qtable, board, print_=False)
            if 2048 in env.board:
                q2048[n] = 1
            qscores[n] = env.score
            qmoves[n] = env.moves
        
        contains_2048[i] = q2048.mean()
        scores_mean[i] = qscores.mean()
        scores_std[i] = qscores.std()
        moves_mean[i] = qmoves.mean()
        moves_std[i] = qmoves.std()
    
    # display data
    ddf = pd.concat([moves_mean, scores_mean, moves_std, scores_std, contains_2048, params['epochs']], axis=1)
    ddf['Epochs'] = ddf['epochs'].cumsum()
    
    fig = px.scatter(ddf, x='# Moves', y='Score', size='Epochs', color='2048?', error_x='moves_std', error_y='score_std')
    return fig

folders = [''] + [f for f in os.listdir() if '.' not in f and not f.startswith('_')]
folder = st.selectbox('Select a results folder', folders)

if folder:
    tab1, tab2 = st.tabs(('Training Statistics', 'Simulations'))
    
    with tab1:
        # load and cache data
        df = load_data(folder)
        params = load_params(folder)
        qtable = load_last_qtable(folder)
        
        col1, col2 = st.columns([2, 8])
        
        # experiment metrics
        col1.metric('Playable Moves Allowed', params['actions'][0])
        col1.metric('Total Games Played', f'{df.shape[0]:.1e}')
        col1.metric('Total Moves Played', f'{df["n_moves"].sum():.1e}')
        col1.metric('Average Moves per Game', f'{df["n_moves"].mean():.0f}')
        col1.metric('Average Maximum Reward', f'{df["max_rewards"].mean():.1f}')
        col1.metric('Average Rewards per Move', f'{df["rewards_per_move"].mean():.1f}')
    
        # plots
        fig1, fig2 = plot_params(params)
        fig3, summary = plot_rewards(df)
        col2.plotly_chart(fig1)
        col2.plotly_chart(fig2)
        col2.plotly_chart(fig3)
        col2.dataframe(summary)
        
    with tab2:
        col_a, col_b = st.columns(2)
        table = get_table()
        fig = show_board(table)
        rand = np.random.randint(1, 101)
        col_a.button('Generate new starting board', on_click=print_state)
        col_a.plotly_chart(fig, theme=None)
        
        with col_b:
            with st.spinner('Simulating 5 games per Q table'):
                fig = simulate(table, folder, params)
        col_b.button('Simulate again', on_click=simulate, args=(table, folder, params))
        col_b.plotly_chart(fig, theme=None)
