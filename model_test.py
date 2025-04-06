# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 14:07:09 2023

@author: fmotoyama
"""
import pickle, itertools
from pathlib import Path
import numpy as np
import torch

from pbcn_env import pbcn
from pbcn_env.pbcn import gym_PBCN as Env
from ddqn import DuelingNetwork, Config



"""
class Config:
    def __init__(self):
        self.observation_size = 20
        self.hidden_size = 64
        self.action_size = 2**20
"""


if __name__ == '__main__':
    load_dir = Path('checkpoints/pbcn_model_10_ddqn_79_2023-12-24-20-52-37')
    #chkpt_file = load_dir / 'pbcn_model_20_reduce_800000.chkpt'
    chkpt_file = list(load_dir.glob('**/*.chkpt'))[0]
    
    with open(load_dir / 'config.pkl', 'rb') as f:
        config = pickle.load(f)
    pbcn_model = config.env.pbcn_model
    target_x = config.env.target_x
    
    # ネットワークを取得
    net = DuelingNetwork(config).to(device='cuda').float()
    load = torch.load(chkpt_file)
    net.load_state_dict(load['online_model'])
    #net.load_state_dict(load['model'])
    
    
    # コントローラーの取得
    temp = 2**config.observation_size
    controller = np.empty(temp, dtype=np.uint32)
    """
    for x_idx, x in enumerate(itertools.product([1,0], repeat=config.observation_size)):
        obs = torch.tensor(x, device='cuda').float()
        obs = obs.unsqueeze(0)  # バッチ分の次元を追加
        action_values = net(obs).squeeze()
        action_idx = torch.argmax(action_values).item()
        controller[x_idx] = action_idx
        print(f'\r{x_idx} \\ {temp}', end='')
    print()
    """
    x_space = np.array(list(itertools.product([1,0], repeat=config.env.n)), dtype=np.bool_)
    batch_size = 256
    for idx in range(temp // batch_size):
        batch = x_space[idx*batch_size:(idx+1)*batch_size]
        batch = torch.tensor(batch, device='cuda').float()
        action_values = net(batch)      # (batch_size,action_length)
        action_idxs = torch.argmax(action_values, axis=1).tolist()
        controller[idx*batch_size:(idx+1)*batch_size] = action_idxs
        print(f'\r{(idx+1)*batch_size} \\ {temp}', end='')
    print()
    #"""
    transition_list = pbcn.pbcn_model_to_transition_list(pbcn_model, controller, config.env.m)
    # 制御できているかの判定
    transition_list_inv = pbcn.is_controlled(transition_list, target_x)
    assert transition_list_inv == pbcn.is_controlled(pbcn.pbcn_model_to_transition_list2(pbcn_model, controller, config.env.m), target_x)
    # ピニングノード
    pinning_node = np.zeros(config.env.m, dtype=np.bool_)
    pinning_count = np.zeros(config.env.m, dtype=np.int32)
    u_space = np.array(list(itertools.product([1,0], repeat=config.env.m)), dtype=np.bool_)
    for u_idx in controller:
        pinning_node[u_space[u_idx]] = True
        pinning_count[u_space[u_idx]] += 1

    """
    pbcn.save_pbcn_info(
        {
            'pbcn_model': pbcn_model,
            'target_x': target_x.tolist(),
            'controller': controller.tolist(),
            'pinning_count': pinning_count.tolist()
            },
        path=config.save_path / 'pbcn_model.txt'
        )
    #"""






















