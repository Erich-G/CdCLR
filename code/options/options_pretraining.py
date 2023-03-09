# graph based model arguments
agcn_model_arguments = {
   "num_class": 128,
   "num_point": 20,
   "num_person": 1,
   'graph_args': {
     'labeling_mode': 'spatial'}
}

#image based model arguments
hcn_model_arguments = {
   "in_channel":3,
   "out_channel":64,
   "window_size":64,
   "num_joint":20,
   "num_person":1,
   "num_class":128
 }

#Sequence based model arguments
bi_gru_model_arguments = {
   "en_input_size":60,
   "en_hidden_size":512,
   "en_num_layers":1,
   "num_class":128
 }

class  opts_ucla():

  def __init__(self):

   self.agcn_model_args = agcn_model_arguments

   self.hcn_model_args = hcn_model_arguments

   self.bi_gru_model_args = bi_gru_model_arguments
   
   # feeder
   self.train_feeder_args = {
     'data_path': './data/UCLA/xview/train_data_joint.npy',
     'num_frame_path': './data/UCLA/xview/train_num_frame.npy',
     'l_ratio': [0.1,1],
     'input_size': 64
   }