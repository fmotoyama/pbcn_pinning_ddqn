# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 20:39:19 2023

@author: fmotoyama
"""

import pickle, itertools, re
import numpy as np

from pbcn_env import drawset, pbcn, def_func
from pbcn_env.pbcn import gym_PBCN as Env


#path = r'.\checkpoints\pbcn_model_3_2023-08-10-22-03-59\q_table.bin'
#with open(path, 'rb') as p:
#    Q_table = pickle.load(p)
#controller = np.argmax(Q_table, axis=1)
#x_space = np.array(list(itertools.product([1,0], repeat=3)), dtype=np.bool_)

#"""
# ランダムに作成
info = def_func.def_f('random2', n=15, max_func_length=3, n_div=5, gamma=1.8, reduce=True)
info['pbcn_model'] = def_func.add_pinning_node(info['pbcn_model'], [0,1,2,3,4,5,6,7,8,9])
#info['target_x'] = np.random.randint(0, 2, len(info['pbcn_model']), dtype=np.bool_).tolist()
pbcn.save_pbcn_info(info)

labels = ['CTLA4','TCR','CREB','IFNG','P2','GPCR','SMAD','Fas','sFas','Ceramide','DISC','Caspase','FLIP','BID','IAP','MCL1','S1P','Apoptosis']
info['pbcn_model'] = [
    [['TCR and not Apoptosis'],[1]],
    [['not (CTLA4 or Apoptosis)'],[1]],
    [['IFNG and not Apoptosis'],[1]],
    [['not (SMAD or P2 or Apoptosis)'],[1]],
    [['(IFNG or P2) and not Apoptosis'],[1]],
    [['SIP and not Apoptosis'],[1]],
    [['GPCR and not Apoptosis'],[1]],
    [['not (sFas or Apoptosis)'],[1]],
    [['SIP and not Apoptosis'],[1]],
    [['Fas and not (SIP or Apoptosis)'],[1]],
    [['(Ceramide or (Fas and not FLIP)) and not Apoptosis'],[1]],
    [['((BID and not IAP) or DISC) and not Apoptosis'],[1]],
    [['not (DISC or Apoptosis)'],[1]],
    [['not (MCL1 or Apoptosis)'],[1]],
    [['not (BID or Apoptosis)'],[1]],
    [['not (DISC or Apoptosis)'],[1]],
    [['not (Ceramide or Apoptosis)'],[1]],
    [['Caspase or Apoptosis'],[1]],
    ]

info['pbcn_model'] = def_func.add_pinning_node(info['pbcn_model'], [6,14,25])
#info['target_x'] = list(map(bool,[0,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0]))

pbcn.save_pbcn_info(info)
#"""

"""
info = pbcn.load_pbcn_info('pbcn_model')

pbcn_model = info['pbcn_model']
target_x = info['target_x']
controller = info['controller']
m = len(pbcn_model)
u_space = np.array(list(itertools.product([1,0], repeat=m)), dtype=np.bool_)

transition_list = pbcn.pbcn_model_to_transition_list(pbcn_model,controller)
#drawset.transition_diagram(transition_list)
controller_func = pbcn.controller_to_func_minimum(m, controller)
a = pbcn.is_controlled(transition_list, target_x)

# ピニングノードを調べる
pinning_node = np.zeros(m, dtype=np.bool_)
pinning_count = np.zeros(m, dtype=np.int32)
for u_idx in controller:
    pinning_node[u_space[u_idx]] = True
    pinning_count[u_space[u_idx]] += 1
"""