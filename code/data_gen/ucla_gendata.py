import argparse
import pickle
from tqdm import tqdm
import sys
from numpy.lib.format import open_memmap
import json

sys.path.extend(['../'])
from data_gen.preprocess import pre_normalization

import numpy as np
import os

view_1_train = '/media/26d532/code/setra/V9/data/UCLA2/view1/train_data.npy'
view_2_train = '/media/26d532/code/setra/V9/data/UCLA2/view2/train_data.npy'
view_3_train = '/media/26d532/code/setra/V9/data/UCLA2/view3/train_data.npy'
view_1_val = '/media/26d532/code/setra/V9/data/UCLA2/view1/val_data.npy'
view_2_val = '/media/26d532/code/setra/V9/data/UCLA2/view2/val_data.npy'
view_3_val = '/media/26d532/code/setra/V9/data/UCLA2/view3/val_data.npy'

view1_train = np.load(view_1_train)
view1_val = np.load(view_1_val)
view2_train = np.load(view_2_train)
view2_val = np.load(view_2_val)
view3_train = np.load(view_3_train)
view3_val = np.load(view_3_val)

v1_data = np.concatenate((view1_train,view1_val))
v2_data = np.concatenate((view2_train,view2_val))
v3_data = np.concatenate((view3_train,view3_val))

view_1_t_label = '/media/26d532/code/setra/V9/data/UCLA2/view1/train_label.pkl'
view_1_v_label = '/media/26d532/code/setra/V9/data/UCLA2/view1/val_label.pkl'
view_2_t_label = '/media/26d532/code/setra/V9/data/UCLA2/view2/train_label.pkl'
view_2_v_label = '/media/26d532/code/setra/V9/data/UCLA2/view2/val_label.pkl'
view_3_t_label = '/media/26d532/code/setra/V9/data/UCLA2/view3/train_label.pkl'
view_3_v_label = '/media/26d532/code/setra/V9/data/UCLA2/view3/val_label.pkl'


with open(view_1_t_label, 'rb') as f1t:
    v1_t_sample_name, v1_t_label = pickle.load(f1t)

with open(view_1_v_label, 'rb') as f1v:
    v1_v_sample_name, v1_v_label = pickle.load(f1v)

with open(view_2_t_label, 'rb') as f2t:
    v2_t_sample_name, v2_t_label = pickle.load(f2t)

with open(view_2_v_label, 'rb') as f2v:
    v2_v_sample_name, v2_v_label = pickle.load(f2v)

with open(view_3_t_label, 'rb') as f3t:
    v3_t_sample_name, v3_t_label = pickle.load(f3t)

with open(view_3_v_label, 'rb') as f3v:
    v3_v_sample_name, v3_v_label = pickle.load(f3v)

v1_label = v1_t_label + v1_v_label
v2_label = v2_t_label + v2_v_label
v3_label = v3_t_label + v3_v_label

# # 保存train_data 和 train_label 是V1和V2的合并

#     with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
#         pickle.dump((sample_name, list(sample_label)), f)

#     np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)