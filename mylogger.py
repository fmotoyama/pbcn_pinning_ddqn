# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 14:45:03 2022

@author: f.motoyama
"""

import time, csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class MetricLogger:
    def __init__(self, config):
        self.config = config
        assert config.save_path.exists()
        self.window = 100   #移動平均の長さ
        
        # ログを残すデータ
        self.label = ['reward','loss','q']#,'action_count']
        
        self.log_path = config.save_path / 'log.csv'
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.label)
        
        # グラフとして描画するデータ
        self.reward = []
        self.loss = []
        self.q = []
        self.moving_avg_reward = []
        self.moving_avg_loss = []
        self.moving_avg_q = []
        self.moving_avg_reward_plot = config.save_path / "reward_plot.jpg"
        self.moving_avg_loss_plot = config.save_path / "loss_plot.jpg"
        self.moving_avg_q_plot = config.save_path / "q_plot.jpg"
        
        self.count = 0
        
    
        
    
    def logging(self, reward, loss, q, action_count):
        """
        記録するデータを受け取る
        """
        self.reward.append(reward)
        self.loss.append(loss)
        self.q.append(q)
        self.moving_avg_reward.append(np.mean(self.reward[-self.window:]))
        self.moving_avg_loss.append(np.mean(self.loss[-self.window:]))
        self.moving_avg_q.append(np.mean(self.q[-self.window:]))
        self.count += 1
        # ログに書き込み
        if self.count % 100 == 0:
            self.write(self.reward[-100:],self.loss[-100:],self.q[-100:],)
            self.draw()
    
    
    def write(self, rewards, losses, qs):
        # ログに書き込み
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for reward, loss, q in zip(rewards, losses, qs):
                writer.writerow((
                    reward,
                    loss,
                    q,
                    ))

    def draw(self):
        # 描画
        for label in self.label:
            plt.plot(getattr(self, f'moving_avg_{label}'))
            plt.xlabel('step')
            plt.ylabel(label)
            plt.savefig(getattr(self, f'moving_avg_{label}_plot'))
            plt.clf()
    


















