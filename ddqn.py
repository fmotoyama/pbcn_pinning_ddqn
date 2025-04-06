# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 19:06:51 2023

@author: fmotoyama
"""

import torch
from torch import nn
import numpy as np
from pathlib import Path
import datetime, math, shutil

from cpprb import PrioritizedReplayBuffer

from mylogger import MetricLogger
from pbcn_env import drawset, def_func, pbcn
from pbcn_env.pbcn import gym_PBCN as Env


class DuelingNetwork(nn.Module):
    # Qを、状態価値関数V(s)とアドバンテージ関数A(s,a)に分ける
    # 状態sだけで決まる部分と、行動aしだいで決まる部分を分離する
    # https://ailog.site/2019/10/29/torch11/
    def __init__(self, config):
        super().__init__()
        n_in = config.observation_size
        n_mid = config.hidden_size
        n_out = config.action_size
        
        self.block = nn.Sequential(
            nn.Linear(n_in, n_mid),
            nn.ReLU(inplace=True),
            nn.Linear(n_mid, n_mid),
            nn.ReLU(inplace=True),
            nn.Linear(n_mid, n_mid),
            nn.ReLU(inplace=True)
        )
        
        self.adv = nn.Sequential(
            nn.Linear(n_mid, n_mid),
            nn.ReLU(inplace=True),
            nn.Linear(n_mid, n_out)
        )
        self.val = nn.Sequential(
            nn.Linear(n_mid, n_mid),
            nn.ReLU(inplace=True),
            nn.Linear(n_mid, 1)
        )
    
    def forward(self, x):
        h = self.block(x)
        # adv: (N,Hout), val: (N,1).expand()
        adv = self.adv(h)
        val = self.val(h).expand(adv.shape)
        #output: (N,Hout)
        output = val + adv - adv.mean(1, keepdim=True)
        
        return output


class DDQN:
    def __init__(self, config):
        self.config = config
        self.env = config.env
        
        self.device = 'cuda' if config.use_gpu else 'cpu'
        assert torch.cuda.is_available() or not config.use_gpu
        self.memory = PrioritizedReplayBuffer(
            config.memory_size,
            env_dict = {
                "obs": {"shape": config.observation_shape},
                "next_obs": {"shape": config.observation_shape},
                "action": {"shape": 1, "dtype":np.int64},
                "reward": {"shape": 1},
                "done": {"shape": 1}
                }
            )
        
        # actor
        #self.epsilon = config.epsilon
        #self.epsilon_rate = math.pow(config.epsilon_min/config.epsilon, 1/config.steps)
        self.temp = config.temp_max
        self.temp_rate = math.pow(config.temp_min/config.temp_max, 1/config.steps)
        
        # learner
        self.online_net = DuelingNetwork(config).to(device=self.device).float()
        self.target_net = DuelingNetwork(config).to(device=self.device).float()
        #self.online_net.eval() # training_modeがデフォなので
        #self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=config.lr)
        self.loss_fn = torch.nn.SmoothL1Loss()
        # logger
        self.logger = MetricLogger(config)
        self.action_count = np.zeros(config.env.m, dtype=np.uint32)
        
        # checkpointの呼び出し
        if config.load_file:
            self.load(config.load_file)
    
     
    def loop(self):
        self.curr_step = 0
        action_all_zero_idx = pbcn.state2idx(np.zeros(self.env.m, dtype=np.bool_))
        while True:
            obs = self.env.reset()
            episode_step = 0
            while True:
                self.curr_step += 1
                episode_step += 1
                # action
                action_idx = self.act2(obs, self.temp)
                #next_obs, reward, done, _ = self.env.step(action)
                next_obs, _, done, _ = self.env.step(pbcn.idx2state(action_idx,self.env.m))
                
                reward = 1 if np.all(next_obs == self.config.env.target_x) else 0
                # オリジナル報酬関数
                if action_idx != action_all_zero_idx:
                    action = pbcn.idx2state(action_idx,self.env.m)
                    self.action_count[action] += 1
                    reward -= self.cost_function1(action, self.action_count, self.config.cost_max)
                    #reward -= self.cost_function2(action, self.action_count, self.config.cost_max, self.temp)
                # memoryに保存
                self.memory.add(
                    priorities = None,
                    obs = obs,
                    next_obs = next_obs,
                    action = action_idx,
                    reward = reward,
                    done = done,
                    )
                # learn
                q, loss = self.learn()
                # Logging
                if q is not None:
                    self.logger.logging(reward, loss, q, self.action_count)
                # Update state
                obs = next_obs
                
                #self.epsilon *= self.epsilon_rate
                self.temp *= self.temp_rate
                # エピソード終了判定
                if done or self.config.max_move <= episode_step or self.config.steps <= self.curr_step:
                #if self.config.max_move <= episode_step or self.config.steps <= self.curr_step:
                    break
            
            if self.config.steps <= self.curr_step:
                break
    
    
    
    @staticmethod
    def normalize(line: np.ndarray):
        #return line/np.sum(line)
        min_ = np.min(line)
        max_ = np.max(line)
        if min_==max_:
            # 合計1に
            return np.full(line.shape, max_/line.size)
        return (line - min_) / (max_ - min_)
    
    def act(self, obs):
        # epsilon-greedy法
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.config.action_size)
            #action = self.env.random_action()
        else:
            obs = torch.tensor(obs.__array__(), device=self.device).float()
            obs = obs.unsqueeze(0)  # バッチ分の次元を追加
            action_values = self.online_net(obs).squeeze()
            action_idx = torch.argmax(action_values).item()
        return action_idx
    
    def act2(self, obs, temp=1):
        # softmax法
        # tauが大きいとランダム
        obs = torch.tensor(obs.__array__(), device=self.device).float()
        obs = obs.unsqueeze(0)  # バッチ分の次元を追加
        # 一次ベクトル化・CPUメモリに置く・データ部分と切り離す・numpy化・メモリを共有しない
        action_values = self.online_net(obs).squeeze().to('cpu').detach().numpy().copy()
        
        p = np.exp(self.normalize(action_values) / temp)
        p /= np.sum(p)
        
        action = np.random.choice(range(len(p)), p=p)
        return action
    
    
    @staticmethod
    def cost_function1(action, action_count, cost_max):
        #if ~np.any(action):
        #    return 0
        #return (1 - np.min(action_count[action]) / np.sum(action_count)) * cost_max
        c = DDQN.normalize(action_count)
        return (1 - np.min(c[action])) * cost_max
    @staticmethod
    def cost_function2(action, action_count, cost_max, temp=1):
        if ~np.any(action):
            return 0
        # softmax 0~1の罰則を算出する
        c = DDQN.normalize(action_count)    # オーバーフローを防ぐ
        c = np.exp(c / temp)
        c /= np.sum(c)
        idxs = np.where(action)[0]
        idx = idxs[np.argmin(action_count[action])]
        assert 0 < ((1 - c[idx]) * cost_max)    #!!!なぜかひっかかる
        return (1 - c[idx]) * cost_max
    
    def learn(self):
        # burn-in
        if self.curr_step < self.config.burnin:
            return None, None
        # target_networkの同期
        if self.curr_step % self.config.sync_every == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        # チェックポイントのセーブ
        if self.curr_step % self.config.save_every == 0:
            self.save()
        # online_networkの更新
        if self.curr_step % self.config.learn_every != 0:
            return None, None

        # Sample from memory
        batch = self.memory.sample(self.config.batch_size)
        obs = torch.tensor(batch['obs'], device=self.device).float()
        next_obs = torch.tensor(batch['next_obs'], device=self.device).float()
        action = torch.tensor(batch['action'], device=self.device).flatten()
        reward = torch.tensor(batch['reward'], device=self.device).flatten()
        done = torch.tensor(batch['done'], device=self.device).flatten()

        # Get TD Estimate (batch,)
        td_est = self.online_net(obs)[np.arange(0, self.config.batch_size), action]
        # Get TD Target
        with torch.no_grad():
            next_Qs = self.online_net(next_obs)
            best_action = torch.argmax(next_Qs, axis=1)
            next_Q = self.target_net(next_obs)[np.arange(0, self.config.batch_size), best_action]
            td_tgt = (reward + (1 - done.float()) * self.config.gamma * next_Q).float()
        
        # Backpropagate loss through Q_online
        weights = torch.tensor(batch['weights'], device=self.device)
        loss = self.loss_fn(td_est*weights, td_tgt*weights)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        
        # メモリの優先度の更新
        self.memory.update_priorities(batch['indexes'], torch.abs(td_tgt - td_est).tolist())

        return (td_est.mean().item(), loss)
    
    
    def save(self):
        """チェックポイントをセーブ"""
        old_files = list(self.config.save_path.glob('**/*.chkpt'))
        if old_files:
            old_files[0].unlink()
        save_file = (self.config.save_path / f"{self.config.name}_{int(self.curr_step)}.chkpt")
        torch.save(dict(
                online_model=self.online_net.state_dict(),
                target_model=self.target_net.state_dict(),
                #exploration_rate=self.epsilon
                temperature_rate=self.temp,
                action_count = self.action_count.tolist()
                ), save_file,
        )
        print(f"Network saved to {save_file}")
        
    
    def load(self, load_file):
        """セーブしたチェックポイントを呼び出す"""
        load = torch.load(load_file)
        self.online_net.load_state_dict(load['online_model'])
        self.target_net.load_state_dict(load['target_model'])
        #self.memory.load_transitions(load_memory)
        self.temp = load['temperature_rate']
        self.action_count = np.array(load['action_count'], dtype=np.uint32)
        print(f'load {load_file}', flush=True)


class Config:
    def __init__(self, name):
        self.env = Env(name)
        
        self.name = name
        self.save_path = Path("checkpoints") / (f'{name}_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        self.load_file = None
        #self.load_file = Path("checkpoints") / 'pbcn_model_10_ddqn_2023-12-04-19-26-51/pbcn_model_10_ddqn_1000000.chkpt'
        
        # Network
        self.observation_size = self.env.n
        #self.action_size = self.env.action_size
        self.action_size = 2**self.env.m
        self.observation_shape = self.env.observation_shape
        #self.action_shape = self.env.action_shape
        self.action_shape = (2**self.env.m,)
        self.hidden_size = 64
        
        # Learner
        self.steps = 1e6
        #self.steps = 5000
        #self.max_move = 100                 # 1エピソード中の最大遷移回数
        self.max_move = 1e4                 # 1エピソード中の最大遷移回数
        self.use_gpu = True
        self.batch_size = 32
        self.memory_size = 5e4
        #self.memory_size = 1000
        self.save_every = self.steps//10    # チェックポイントをセーブする間隔
        self.gamma = 0.997                  # Q学習の割引率
        self.lr = 0.00025                   # optimizerの学習率
        self.burnin = self.batch_size * 5   # トレーニングを始める前に行うステップ数
        self.sync_every = 1e3               # Q_targetにQ_onlineを同期する間隔
        self.learn_every = 1                # Q_onlineを更新する間隔
        # PER
        #self.PER_epsilon = 0.001            # 重さが0になることを防ぐ微小量
        #self.PER_alpha = 0.6                # 0~1 0のとき完全なランダムサンプリング
        # 重要度サンプリング
        #self.IS_beta = 0.4                  # 補正の強さ 固定値
        
        # Actor
        #self.epsilon = 0.8                 # epsilon-greety
        #self.epsilon_min = 0.01
        # softmax
        self.temp_max = 1
        self.temp_min = 0.02
        
        self.cost_max = 1.0
        #self.cost_max = 0.0



"""
# ピニングノードを付与
name = 'pbcn_model_10 (2)'
info = pbcn.load_pbcn_info(name)
info['pbcn_model'] = def_func.add_pinning_node(pbcn_model)
pbcn.save_pbcn_info(info)
"""

if __name__ == '__main__':
    import itertools, pickle
    
    name = 'pbcn_model_10'
    #name = 'pbcn_model_10 (2)'
    #name = 'pbcn_model_10_ddqn_79'
    #name = 'pbcn_model_3'
    info = pbcn.load_pbcn_info(name)
    pbcn_model = info['pbcn_model']
    
    config = Config(name)
    # 記録用のディレクトリを作成
    config.save_path.mkdir(parents=True)
    with open(config.save_path/'config.pkl', mode="wb") as f:
        pickle.dump(config, f)
    
    
    ddqn = DDQN(config)
    ddqn.loop()
    action_count = ddqn.action_count
    
    
    x_space = np.array(list(itertools.product([1,0], repeat=config.env.n)), dtype=np.bool_)
    u_space = np.array(list(itertools.product([1,0], repeat=config.env.m)), dtype=np.bool_)
    """
    # Q_tableを作成して保存
    Q_table = np.zeros((2**config.env.n,2**config.env.m), dtype=np.float32)
    for idx in range(2 ** config.env.n):
        obs = torch.tensor(x_space[idx].__array__(), device=ddqn.device).float()
        obs = obs.unsqueeze(0)  # バッチ分の次元を追加
        Q_table[idx] = ddqn.online_net(obs).cpu().detach()
    with open(config.save_path / 'q_table.bin', 'wb') as p:
        pickle.dump(Q_table, p)
    controller = np.argmax(Q_table, axis=1)
    """
    # controllerを直接得る
    controller = np.empty(2**config.env.n, dtype=np.uint32)
    temp = 2**config.env.n
    #"""
    for x_idx,x in zip(range(2**config.env.n),x_space):
        obs = torch.tensor(x.__array__(), device=ddqn.device).float()
        obs = obs.unsqueeze(0)  # バッチ分の次元を追加
        controller[x_idx] = np.argmax(ddqn.online_net(obs).clone().detach().cpu().numpy())
        print(f'\r{x_idx} \\ {temp}', end='')
    """
    batch_size = 256
    temp = 2**config.env.n
    for idx in range(temp // batch_size):
        batch = x_space[idx*batch_size:(idx+1)*batch_size]
        batch = torch.tensor(batch, device='cuda').float()
        action_values = ddqn.online_net(batch)      # (batch_size,action_length)
        action_idxs = torch.argmax(action_values, axis=1).tolist()
        controller[idx*batch_size:(idx+1)*batch_size] = action_idxs
        print(f'\r{(idx+1)*batch_size} \\ {temp}', end='')
    print()
    #"""
    
    
    
    # コントローラーを組み込んだ図
    #pbn_model = pbcn.embed_controller(pbcn_model, controller, minimum=True)
    transition_list = pbcn.pbcn_model_to_transition_list(pbcn_model, controller, config.env.m)
    #drawset.transition_diagram(transition_list, f'td_{name}')
    
    # 制御できているかの判定
    transition_list_inv = pbcn.is_controlled(transition_list, info['target_x'])
    if transition_list_inv:
        print(f'transition_list_inv: {len(transition_list_inv)}')
    # ピニングノード
    pinning_node = np.zeros(config.env.m, dtype=np.bool_)
    pinning_count = np.zeros(config.env.m, dtype=np.uint32)
    for u_idx in controller:
        pinning_node[u_space[u_idx]] = True
        pinning_count[u_space[u_idx]] += 1
    
    
    pbcn.save_pbcn_info(
        info | {
            'controller': controller.tolist(),
            'action_count': ddqn.action_count.tolist(),
            'pinning_count': pinning_count.tolist()
            },
        path=config.save_path / 'pbcn_model.txt'
        )








