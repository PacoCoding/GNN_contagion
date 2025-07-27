import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv   # or GATConv, TransformerConv, etc.

class SpatioTemporalGNN(nn.Module):
        def __init__(self, in_feats, mlp_feats, gcn_feats, mlp_hidden2, out_classes, p_drop):
            super().__init__()
            self.pre_mlp = nn.Sequential(
                nn.Linear(in_feats, mlp_feats), nn.BatchNorm1d(mlp_feats), nn.ReLU(), nn.Dropout(p_drop),
                nn.Linear(mlp_feats, gcn_feats), nn.BatchNorm1d(gcn_feats), nn.ReLU(), nn.Dropout(p_drop)
            )
            self.gcns = nn.ModuleList([GCNConv(gcn_feats, gcn_feats) for _ in range(3)])
            self.gcn_norms = nn.ModuleList([nn.BatchNorm1d(gcn_feats) for _ in range(3)])
            self.gcn_dropout = nn.Dropout(p_drop)
            self.post_mlp = nn.Sequential(
                nn.Linear(gcn_feats, gcn_feats), nn.BatchNorm1d(gcn_feats), nn.ReLU(), nn.Dropout(p_drop)
            )
            self.gru = nn.GRU(gcn_feats, gcn_feats, batch_first=True)
            self.classifier = nn.Sequential(
                nn.Linear(gcn_feats, mlp_hidden2), nn.BatchNorm1d(mlp_hidden2), nn.ReLU(), nn.Dropout(p_drop),
                nn.Linear(mlp_hidden2, out_classes)
            )

        def forward(self, data_seq):
            h_steps = []
            for data in data_seq:
                x = self.pre_mlp(data.x)
                h = x
                for gcn, bn in zip(self.gcns, self.gcn_norms):
                    h_new = gcn(h, data.edge_index, edge_weight=data.edge_weight)
                    h_new = bn(h_new)
                    h_new = F.relu(h_new)
                    h_new = self.gcn_dropout(h_new)
                    h = h + h_new
                h2 = self.post_mlp(h)
                h_steps.append(h2.unsqueeze(0))
            H = torch.cat(h_steps, 0).permute(1,0,2)
            _, h_T = self.gru(H)
            h_T = h_T.squeeze(0)
            return self.classifier(h_T)

class TemporalLSTM(torch.nn.Module):
    def __init__(self, in_feats, lstm_hidden=128, lstm_layers=2,
                 mlp_hidden=64, num_classes=4, dropout=0.1):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=in_feats,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(lstm_hidden, mlp_hidden),
            torch.nn.BatchNorm1d(mlp_hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(mlp_hidden, num_classes)
        )

    def forward(self, x_seq):
        # x_seq: [N, T, F]
        H, _ = self.lstm(x_seq)      # H: [N, T, lstm_hidden]
        h_last = H[:, -1, :]         # [N, lstm_hidden]
        return self.classifier(h_last)  # [N, num_classes]

class MLP(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes, dropout=0.1):
        super().__init__()
        self.lin1 = nn.Linear(in_feats, hidden_feats)
        self.lin2 = nn.Linear(hidden_feats, hidden_feats)
        self.lin3 = nn.Linear(hidden_feats, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = F.relu(self.lin2(x))
        x = self.dropout(x)
        x = self.lin3(x)
        return x


class CoreGCN(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, p_drop=0.1):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_feats)
        self.bn1   = BatchNorm(hidden_feats)
        self.conv2 = GCNConv(hidden_feats, hidden_feats)
        self.bn2   = BatchNorm(hidden_feats)
        self.p_drop = p_drop

    def forward(self, x, edge_index, edge_weight=None):
        # 1) Raw edge-weight stats
        if edge_weight is None:
            print(" ▶ raw edge_weight: None")
        else:
            print(f" ▶ raw edge_weight: min={edge_weight.min():.4g}, "
                  f"max={edge_weight.max():.4g}, nan={edge_weight.isnan().any().item()}")

        # 2) Normalize exactly as GCNConv does under the hood
        norm_ei, norm_ew = gcn_norm(
            edge_index, edge_weight,
            num_nodes=x.size(0),
            improved=False,
            add_self_loops=True,
            dtype=x.dtype,
        )

        # # 3) Compute “degree” from the normalized weights
        # deg = scatter(norm_ew, norm_ei[1], dim=0, reduce="sum")
        # print(f" ▶ normalized edges: {norm_ei.size(1)} edges, "
        #       f"deg: min={deg.min():.4g}, max={deg.max():.4g}, nan={deg.isnan().any().item()}")

        # 4) Now apply the two GCN layers
        x = self.conv1(x, norm_ei, norm_ew)
        x = F.relu(self.bn1(x))
        x = F.dropout(x, p=self.p_drop, training=self.training)

        x = self.conv2(x, norm_ei, norm_ew)
        x = F.relu(self.bn2(x))
        return F.dropout(x, p=self.p_drop, training=self.training)

class OutputMLP(torch.nn.Module):
    def __init__(self, in_feats, h1=128, h2=64, n_classes=4, p_drop=0.1):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_feats, h1)
        self.bn1 = torch.nn.BatchNorm1d(h1)
        self.fc2 = torch.nn.Linear(h1, h2)
        self.bn2 = torch.nn.BatchNorm1d(h2)
        self.fc3 = torch.nn.Linear(h2, n_classes)
        self.p_drop = p_drop

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=self.p_drop, training=self.training)
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, p=self.p_drop, training=self.training)
        return self.fc3(x)

class InterbankNetGCN(torch.nn.Module):
    def __init__(self, in_feats, gcn_h, mlp_h1, mlp_h2, n_classes, p_drop=0.1):
        super().__init__()
        self.input_mlp  = InputMLP(in_feats,  mlp_h1, mlp_h2, p_drop)
        self.core_gcn   = CoreGCN(mlp_h2, gcn_h, p_drop)
        self.output_mlp = OutputMLP(gcn_h, mlp_h1, mlp_h2, n_classes, p_drop)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.input_mlp(x)
        x = self.core_gcn(x, edge_index, edge_weight)
        return self.output_mlp(x)



