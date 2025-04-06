# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:56:21 2023

@author: fmotoyama
"""
import re, itertools
import numpy as np

labels = ['CTLA4','TCR','CREB','IFNG','P2','GPCR','SMAD','Fas','sFas','Ceramide','DISC','Caspase','FLIP','BID','IAP','MCL1','SIP','Apoptosis']
pbcn_model = [
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

for funcs,_ in pbcn_model:
    for i,func in enumerate(funcs):
        for j,label in enumerate(labels):
            #funcs[i] = re.sub(f'(\s*){label}', f'\\1x[{j}]', funcs[i])
            funcs[i] = re.sub(f'^{label}|(\W){label}', f'\\1x[{j}]', funcs[i])

# アトラクタを調べる
def state2idx(state):
    return sum((2**i)*v for i,v in enumerate(np.flip(~state)))
n = len(pbcn_model)
x_space_size = 2 ** n
x_space = np.array(list(itertools.product([1,0], repeat=n)), dtype=np.bool_)
x_sheet = np.zeros(x_space_size, dtype=np.bool_)
attractor = []
for x_idx,x in enumerate(x_space):
    if ~x_sheet[x_idx]:
        
        while True:
            x_next = np.array([eval(funcs[0]) for funcs,probs in pbcn_model], dtype=np.bool_)
            x_next_idx = state2idx(x_next)
            if x_sheet[x_next_idx]:
                if x_idx == x_next_idx:
                    attractor.append(x_idx)
                break
            else:
                x_sheet[x_next_idx] = True
            x = x_next
            x_idx = x_next_idx
        
        print(f'\r{x_idx} / {x_space_size}', end='')


from pbcn_env import pbcn, def_func
pbcn.save_pbcn_info(
    {
     'pbcn_model': def_func.add_pinning_node(pbcn_model,[0,4,10,16,17,1,2,3,5,6]),
     'target_x': x_space[attractor[0]].tolist()
     }
    )
# CTLA4,P2,DISC,SIP,Apoptosisをピニングノードにして制御可能




