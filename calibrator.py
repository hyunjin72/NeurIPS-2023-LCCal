import numpy as np
import torch
from torch.nn import functional as F
import torch_sparse
from torch_sparse import SparseTensor
from torch_geometric.utils import degree

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LCCal():
    def __init__(self, args):
        super(LCCal, self).__init__()
        self.args = args
    
    def compute_LCC(self, data, logits):
        # calculate inverse degree.
        deg = degree(data.edge_index[0], num_nodes=self.args.n_nodes).to(device) # (n_nodes,)
        inv_deg = 1/deg
        inv_deg[inv_deg==float('inf')] = 0.
        inv_deg[inv_deg.isnan()] = 0.
        
        # obtain probability matrix and confidence/prediction vectors.
        probs = F.softmax(logits, dim=1)
        confs, preds = probs.max(1)
        
        # compute (predicted) label matrix Y.
        Y = torch.zeros(self.args.n_nodes, self.args.n_classes).to(device)
        Y[torch.arange(self.args.n_nodes), preds] = 1
        
        # compute p_{i,yi}.
        my_probs = confs.clone()
        
        # compute normalized node features.
        x_norm = probs / (probs.norm(dim=1, keepdim=True) + 1e-8)
        x_norm = x_norm.cpu()
        # compute dot product between all pairs of node features.
        all_sim = torch.matmul(x_norm, x_norm.t())
        values = all_sim[data.edge_index[0], data.edge_index[1]].to(device)
        # create weighted adj_t.
        w_adj_t = SparseTensor.from_edge_index(data.edge_index, values, sparse_sizes=(data.num_nodes, data.num_nodes)).to(device)

        del (all_sim, values, x_norm)
        torch.cuda.empty_cache()

        # obtain LCC.
        AP = torch_sparse.matmul(w_adj_t, probs)
        weighted_agg = (Y * AP).sum(1) * inv_deg
        LCC = (my_probs + weighted_agg)/2
        return LCC, confs
    
    def calibrate(self, data, logits):
        LCC, confs = self.compute_LCC(data, logits)
        T = torch.zeros(logits.size(0)).to(device)
        mask = LCC<=self.args.default_l
        left = torch.exp(confs-LCC) - 1 + self.args.b_over + confs[mask].mean()
        right = (1-confs) * (torch.exp(-LCC) - 1) + self.args.b_under + confs[~mask].mean()
        
        T[mask] = left[mask]
        T[~mask] = right[~mask]
        T[T<=self.args.eps] = self.args.eps
        assert len(torch.nonzero(T<=0).flatten()) == 0
        assert len(torch.nonzero(T.isnan()).flatten()) == 0
        
        cal_logits = logits / T.view(-1,1)
        return cal_logits
