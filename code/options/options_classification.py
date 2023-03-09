class  opts_ucla():

  def __init__(self):

  # graph based model
   self.agcn_model_args = {
      "num_class": 10,
      "num_point": 20,
      "num_person": 1,
      'graph_args': {
        'labeling_mode': 'spatial'}
   }

   #image based model
   self.hcn_model_args = {
      "in_channel":3,
      "out_channel":64,
      "window_size":64,
      "num_joint":20,
      "num_person":1,
      "num_class":10
    }

   #Sequence based model
   self.bi_gru_model_args = {
      "en_input_size":60,
      "en_hidden_size":512,
      "en_num_layers":1,
      "num_class":10,
    }
   
   # feeder
   self.train_feeder_args = {
     'data_path': './data/UCLA/xview/train_data_joint.npy',
     'label_path': './data/UCLA/xview/train_label.pkl',
     'num_frame_path': './data/UCLA/xview/train_num_frame.npy',
     'l_ratio': [0.5,1.0],
     'input_size': 64
   }
   
   self.test_feeder_args = {

     'data_path': './data/UCLA/xview/val_data_joint.npy',
     'label_path': './data/UCLA/xview/val_label.pkl',
     'num_frame_path': './data/UCLA/xview/val_num_frame.npy',
     'l_ratio': [0.95],
     'input_size': 64
   }