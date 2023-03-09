# sys
import pickle

# torch
import torch
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
np.set_printoptions(threshold=np.inf)
import random

try:
    from feeder import augmentations
except:
    import augmentations


class Feeder(torch.utils.data.Dataset):
    """ 
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
    """

    def __init__(self,
                 data_path,
                 num_frame_path,
                 l_ratio,
                 input_size,
                 input_representation,
                 mmap=True):

        self.data_path = data_path
        self.num_frame_path= num_frame_path
        self.input_size=input_size
        self.input_representation=input_representation
        self.crop_resize =True
        self.l_ratio = l_ratio


        self.load_data(mmap)
        self.N, self.C, self.T, self.V, self.M = self.data.shape
        print(self.data.shape,len(self.number_of_frames))
        print("l_ratio",self.l_ratio)

    def load_data(self, mmap):
        # data: N C V T M

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        # load num of valid frame length
        self.number_of_frames= np.load(self.num_frame_path)
    
    def get_aug(self,clip,number_of_frames,ta=True,pa=True):

       # get raw input

        # input: C, T, V, M
        start = -1
        data_numpy = clip

        if ta!=False:

            # apply spatio-temporal augmentations to generate  view 1 

            # temporal crop-resize
            start,data_numpy_v1_crop = augmentations.temporal_cropresize(data_numpy, number_of_frames, self.l_ratio, self.input_size)
        else:
            data_numpy_v1_crop = data_numpy

        if pa!=False:

            # randomly select  one of the spatial augmentations 
            flip_prob  = random.random()
            if flip_prob < 0.5:
                    data_numpy_v1 = augmentations.joint_courruption(data_numpy_v1_crop)
            else:
                    data_numpy_v1 = augmentations.pose_augmentation(data_numpy_v1_crop)
        else:
            data_numpy_v1 = data_numpy_v1_crop

        return start,data_numpy_v1


    def __len__(self):
        return self.N

    def __iter__(self):
        return self

    def __getitem__(self, index):

        data_numpy = np.array(self.data[index])
        # number_of_frames = self.number_of_frames[index]
        # 判断帧数
        frame_data = data_numpy.copy()
        frame_data = frame_data.transpose([1, 2, 3, 0])
        frame_data = frame_data.reshape(-1,60)
        frame_sum = np.sum(frame_data,axis=1)
        number_of_frames = np.count_nonzero(frame_sum)

        
        C, T, V, M = data_numpy.shape
        rand_segment = random.randint(0, 1)
        frame_num = number_of_frames         

        # 先进行时域增广
        start_1,frame_idx_1 = self.get_aug(data_numpy,number_of_frames,ta=True,pa=False)
        start_2,frame_idx_2 = self.get_aug(data_numpy,number_of_frames,ta=True,pa=False)
        start_3,frame_idx_3 = self.get_aug(data_numpy,number_of_frames,ta=True,pa=False)

        # 判断时间顺序

        d = {start_1:frame_idx_1,start_2:frame_idx_2,start_3:frame_idx_3}
        s = [start_1,start_2,start_3]
        s.sort()
        f = []
        for i in range(len(s)):
            f.append(d[s[i]])

        if rand_segment == 0:
            _,frame1_aug1 = self.get_aug(f[0],number_of_frames,ta=False,pa=True)
            _,frame1_aug2 = self.get_aug(f[0],number_of_frames,ta=False,pa=True)
            _,frame2_aug = self.get_aug(f[1],number_of_frames,ta=False,pa=True)
            _,frame3_aug = self.get_aug(f[2],number_of_frames,ta=False,pa=True)
        else:
            _,frame2_aug = self.get_aug(f[0],number_of_frames,ta=False,pa=True)
            _,frame3_aug = self.get_aug(f[1],number_of_frames,ta=False,pa=True)
            _,frame1_aug1 = self.get_aug(f[2],number_of_frames,ta=False,pa=True)
            _,frame1_aug2 = self.get_aug(f[2],number_of_frames,ta=False,pa=True)

        if self.input_representation == "seq-based": 

             #Input for sequence-based representation
             # two person  input ---> shpae (64 X 150)
            frame1_aug1 = frame1_aug1.transpose(1,2,0,3)
            
            frame1_aug1 = frame1_aug1.reshape(-1,60).astype('float32')
            frame1_aug2 = frame1_aug2.transpose(1,2,0,3)
            frame1_aug2 = frame1_aug2.reshape(-1,60).astype('float32')
            frame2_aug = frame2_aug.transpose(1,2,0,3)
            frame2_aug = frame2_aug.reshape(-1,60).astype('float32')
            frame3_aug = frame3_aug.transpose(1,2,0,3)
            frame3_aug = frame3_aug.reshape(-1,60).astype('float32')

        elif self.input_representation == "graph-based" or self.input_representation == "image-based": 

             #input for graph-based or image-based representation
             # two person input --->  shape (3, 64, 25, 2)
            frame1_aug1 = frame1_aug1.astype('float32')
            frame1_aug2 = frame1_aug2.astype('float32')
            frame2_aug = frame2_aug.astype('float32')
            frame3_aug = frame3_aug.astype('float32')

        
        return frame1_aug1, frame1_aug2, frame2_aug, frame3_aug, rand_segment
