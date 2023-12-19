from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
from torch_scatter import scatter
from collections import defaultdict
class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, base_model=GCNConv, k: int = 2, skip=False):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k 
        self.skip = skip
        if not self.skip:
            self.conv = [base_model(in_channels, 2 * out_channels).jittable()]
            for _ in range(1, k - 1):
                self.conv.append(base_model(2 * out_channels, 2 * out_channels))
            self.conv.append(base_model(2 * out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation
        else:
            self.fc_skip = nn.Linear(in_channels, out_channels)
            self.conv = [base_model(in_channels, out_channels)]
            for _ in range(1, k):
                self.conv.append(base_model(out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        if not self.skip:
            for i in range(self.k):
                x = self.activation(self.conv[i](x, edge_index))
            return x
        else:
            h = self.activation(self.conv[0](x, edge_index))
            hs = [self.fc_skip(x), h]
            for i in range(1, self.k):
                u = sum(hs)
                hs.append(self.activation(self.conv[i](u, edge_index)))
            return hs[-1]


class DCMSL(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int, tau: float = 0.5):
        super(DCMSL, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

        self.num_hidden = num_hidden

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    

   
    def community_contrastive(self, z1, z2, communities,delta,gamma) -> torch.Tensor:
        communities = torch.tensor(communities, device=z1.device)
        unique_communities, community_counts = torch.unique(communities, return_counts=True)
        community_sums_z1 = torch.zeros(len(unique_communities), z1.size(1), device=z1.device)
        community_sums_z2 = torch.zeros(len(unique_communities), z2.size(1), device=z2.device)
        community_sums_z1.scatter_add_(0, communities.unsqueeze(1).expand(-1, z1.size(1)), z1)
        community_sums_z2.scatter_add_(0, communities.unsqueeze(1).expand(-1, z2.size(1)), z2)
        readout_z1 = community_sums_z1 / community_counts.unsqueeze(1).to(z1.device)
        readout_z2 = community_sums_z2 / community_counts.unsqueeze(1).to(z2.device)
        temp = lambda x: torch.exp(x / self.tau)
        refl_sim = temp(self.sim(readout_z1, readout_z1))
        between_sim = temp(self.sim(readout_z1, readout_z2))
        clgcl = -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        refl_sim_node = temp(self.sim(z1, z1))
        between_sim_node = temp(self.sim(z1, z2))
        mask = (communities.unsqueeze(0) == communities.unsqueeze(1)).float()
        mask[mask==0]=torch.tensor(delta).to(z1.device)
        refl_sim_node=refl_sim_node*mask
        between_sim_node=between_sim_node*mask
        pfgcl=-torch.log(between_sim_node.diag() / (refl_sim_node.sum(1) + between_sim_node.sum(1) - refl_sim_node.diag()))
        return torch.mean(pfgcl)+gamma*torch.mean(clgcl)
        
   
    def msgcl(self,
                     z1: torch.Tensor,
                     z2: torch.Tensor,
                     com,delta,gamma) -> torch.Tensor:

        h1 = self.projection(z1)
        h2 = self.projection(z2)    
        l1 = self.community_contrastive(h1, h2,com,delta,gamma)
        l2 = self.community_contrastive(h2, h1,com,delta,gamma)
        ret = (l1 + l2) * 0.5
        ret = ret.mean()
        return ret
    



class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret
