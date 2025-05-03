import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv   # or GATConv, TransformerConv, etc.

class PathLSTMEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid_dim, batch_first=True, num_layers = 2,
                            bidirectional=True)

    def forward(self, C):  # C: [N, P, L, F+1]
        N,P,L,D = C.shape
        x = C.view(N*P, L, D)              # [N*P, L, D]
        _, (h, _) = self.lstm(x)          # h: [2, N*P, hid_dim]
        h = torch.cat([h[0], h[1]], dim=-1)  # [N*P, 2*hid_dim]
        h = h.view(N, P, -1)              # [N, P, 2*hid_dim]
        return h.mean(dim=1)              # [N, 2*hid_dim]


class GNN_LSTM_Classifier(nn.Module):
    def __init__(self,
                 node_feat_dim: int,    # e.g. 70
                 lstm_hidden:  int,     # e.g. 64
                 gnn_hidden:  int,      # e.g. 64
                 num_classes: int       # e.g. 4
                ):
        super().__init__()
        # 1) Static, 1-hop GNN that *does* use edge_attr as you already have:
        self.gnn = GCNConv(node_feat_dim, gnn_hidden)

        # 2) Path‐LSTM over your [node_feats + weight] sequence, 
        #    in_dim = original node‐features (70) + 1 weight
        self.path_enc = PathLSTMEncoder(in_dim=node_feat_dim+1,
                                        hid_dim=lstm_hidden)

        # 3) Final fusion: concat GNN‐embedding + 2*lstm_hidden  
        self.fuse = nn.Linear(gnn_hidden + 2*lstm_hidden, num_classes)

    def forward(self,
                X: torch.Tensor,             # [N, node_feat_dim]
                edge_index: torch.LongTensor, # [2, E]
                edge_attr: torch.Tensor,     # [E, 1]
                C: torch.Tensor               # [N, P, L, node_feat_dim+1]
               ) -> torch.Tensor:
        # — Static GNN branch —
        h0 = F.relu(self.gnn(X, edge_index, edge_attr))  
        # h0: [N, gnn_hidden]

        # — Contagion LSTM branch —
        hp = self.path_enc(C)  
        # hp: [N, 2*lstm_hidden]

        # — Fusion & Classification —
        h  = torch.cat([h0, hp], dim=-1)  
        # h: [N, gnn_hidden + 2*lstm_hidden]

        return self.fuse(h)  
        # output: [N, num_classes], log‐probs if you wrap it in LogSoftmax later
