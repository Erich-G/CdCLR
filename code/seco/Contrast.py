import torch
import torch.nn as nn
import math
from .GRU import BIGRU
from .AGCN import Model as AGCN
from .HCN import HCN

# initilize weight
def weights_init_gru(model):
    with torch.no_grad():
        for child in list(model.children()):
            print(child)
            for param in list(child.parameters()):
                  if param.dim() == 2:
                        nn.init.xavier_uniform_(param)
    print('GRU weights initialization finished!')

class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        label = torch.zeros([x.shape[0]]).long().to(x.device)
        return self.criterion(x, label)


class MemorySeCo(nn.Module):
    """Fixed-size queue with momentum encoder"""
    def __init__(self,skeleton_representation,args_bi_gru, args_agcn, args_hcn, feature_dim=128, queue_size=65536,m=0.999,temperature=0.07, temperature_intra=0.07,mlp=False):
        super(MemorySeCo, self).__init__()
        self.queue_size = queue_size
        self.m = m
        self.temperature = temperature
        self.temperature_intra = temperature_intra
        mlp = mlp
        self.index = 0
        print(" secco parameters",queue_size,temperature,temperature_intra,mlp)

        if skeleton_representation=='seq-based':
            self.encoder_q = BIGRU(**args_bi_gru)
            self.encoder_k = BIGRU(**args_bi_gru)
            weights_init_gru(self.encoder_q)
            weights_init_gru(self.encoder_k)
        elif skeleton_representation=='graph-based':
            self.encoder_q = AGCN(**args_agcn)
            self.encoder_k = AGCN(**args_agcn)
        elif skeleton_representation=='image-based':
            self.encoder_q = HCN(**args_hcn)
            self.encoder_k = HCN(**args_hcn)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # noinspection PyCallingNonCallable
        self.register_buffer('params', torch.tensor([-1]))
        stdv = 1. / math.sqrt(feature_dim / 3)
        memory = torch.rand(self.queue_size, feature_dim, requires_grad=False).mul_(2 * stdv).add_(-stdv)
        self.register_buffer('memory', memory)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, k_all):

        all_size = k_all.shape[0]
        out_ids = torch.fmod(torch.arange(all_size, dtype=torch.long).cuda() + self.index, self.queue_size)
        self.memory.index_copy_(0, out_ids, k_all)
        self.index = (self.index + all_size) % self.queue_size
    
    def _out_inter_intra(self, q, k_sf, k_df1, k_df2, k_all, inter=True):
        # compute logits
        l_pos_sf = (q * k_sf.detach()).sum(dim=-1, keepdim=True)  # shape: (batchSize, 1)
        l_pos_df1 = (q * k_df1.detach()).sum(dim=-1, keepdim=True)  # shape: (batchSize, 1)
        l_pos_df2 = (q * k_df2.detach()).sum(dim=-1, keepdim=True)  # shape: (batchSize, 1)
        
        if inter:
            # TODO: remove clone. need update memory in backwards
            l_neg = torch.mm(q, self.memory.clone().detach().t())
            out = torch.cat((torch.cat((l_pos_sf, l_pos_df1, l_pos_df2), dim=0), l_neg.repeat(3, 1)), dim=1)
            out = torch.div(out, self.temperature).contiguous()
            self._dequeue_and_enqueue(k_all)
        else:
            # out intra-frame similarity
            out = torch.div(torch.cat((l_pos_sf.repeat(2, 1), torch.cat((l_pos_df1, l_pos_df2), dim=0)), dim=-1), self.temperature_intra).contiguous()
        return out

    def forward(self, xq, x1, x2, x3):
    # def forward(self, q, k_sf, k_df1, k_df2, k_all, inter=True):
        # compute key features
        with torch.no_grad():
            x1_feat_inter, x1_feat_intra, x1_feat_order = self.encoder_k(x1)
            x2_feat_inter, x2_feat_intra, x2_feat_order = self.encoder_k(x2)
            x3_feat_inter, x3_feat_intra, x3_feat_order = self.encoder_k(x3)            

        # compute query features
        xq_feat_inter, xq_feat_intra, xq_feat_order, xq_logit_order = self.encoder_q(xq, order_feat=torch.cat([x2_feat_order.detach(), x3_feat_order.detach()], dim=1))        

        # inter_out
        out_inter = self._out_inter_intra(xq_feat_inter,
                                          x1_feat_inter,x2_feat_inter,x3_feat_inter,
                                          torch.cat([x1_feat_inter, x2_feat_inter, x3_feat_inter], dim=0), inter=True)

        # intra_out
        out_intra = self._out_inter_intra(xq_feat_intra,
                                          x1_feat_intra, x2_feat_intra, x3_feat_intra, None, inter=False)        

        # labels: positive key indicators
        labels = torch.zeros(out_inter.shape[0], dtype=torch.long).cuda()
        self._momentum_update_key_encoder()

        return out_inter, out_intra,labels,xq_logit_order

