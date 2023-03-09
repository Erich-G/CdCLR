import argparse
import pickle
from tqdm import tqdm
import sys
from numpy.lib.format import open_memmap
import json

sys.path.extend(['../'])
from data_gen.preprocess import pre_normalization

training_cameras = [1, 2]
max_body_true = 1
# max_body_kinect = 3
num_joint = 20
max_frame = 201

import numpy as np
import os

training_cameras = [1, 2]
action_dic = {1:0,2:1,3:2,4:3,5:4,6:5,8:6,9:7,11:8,12:9} 

def read_skeleton(file):
    with open(file, 'r') as f:
        data = json.load(f)
        skeleton = np.array(data['skeletons'])
        label = data['label']
        T,J,C = skeleton.shape
    return skeleton,label,T,J

def get_nonzero_std(s):  # tvc
    index = s.sum(-1).sum(-1) != 0  # select valid frames
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  # three channels
    else:
        s = 0
    return s

def read_xyz(file):
    seq,label,T,J = read_skeleton(file)
    data = np.zeros((1, T, J, 3))
    data[0,:,:,:] = seq

    # select one max energy body
    energy = np.array([get_nonzero_std(x) for x in data])
    index = energy.argsort()[::-1][0:max_body_true]
    data = data[index]

    data = data.transpose(3, 1, 2, 0)    # 3 T J 1
    return data

def gendata(data_path,out_path,benchmark='xview',part ='train'):
    sample_name = []
    sample_label = []   
    for filename in os.listdir(data_path):
        action_class = int(
            filename[filename.find('a') + 1:filename.find('a') + 3])
        subject_id = int(
            filename[filename.find('s') + 1:filename.find('s') + 3])
        camera_id = int(
            filename[filename.find('v') + 1:filename.find('v') + 3])
        environment_id = int(
            filename[filename.find('e') + 1:filename.find('e') + 3])
        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        else:
            raise ValueError()
        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError() 
        if issample:
            sample_name.append(filename)
            sample_label.append(action_dic[action_class]) 

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    fl = open_memmap(
        '{}/{}_num_frame.npy'.format(out_path, part),
        dtype='int',
        mode='w+',
        shape=(len(sample_label),))

    fp = np.zeros((len(sample_label), 3, max_frame, num_joint, max_body_true), dtype=np.float32)

    for i, s in enumerate(tqdm(sample_name)):
        data = read_xyz(os.path.join(data_path, s))
        fp[i, :, 0:data.shape[1], :, :] = data
        fl[i] = data.shape[1] # num_frame

    fp = pre_normalization(fp)
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UCLA Data Converter.')
    parser.add_argument('--data_path', default='/media/26d532/gr/UCLA/')
    parser.add_argument('--out_folder', default='../data/UCLA/')
    benchmark = ['xview']

    #parser.add_argument('--data_path', default='../data/nturgbd_raw_120/nturgb+d_skeletons/')
    #parser.add_argument('--ignored_sample_path',
    #                    default='../data/nturgbd_raw_120/samples_with_missing_skeletons.txt')
    #parser.add_argument('--out_folder', default='../data/NTU-RGB-D-120-AGCN/')
    #benchmark = ['xsub','xsetup', ]

    part = ['train', 'val']
    arg = parser.parse_args()

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            print(b, p)
            gendata(
                arg.data_path,
                out_path,
                benchmark=b,
                part=p)