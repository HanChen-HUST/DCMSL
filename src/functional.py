import numpy as np
import networkx as nx
import torch
from torch_geometric.utils import degree, to_undirected
from cdlib.utils import convert_graph_formats
import json




def NCED(edge_index, edge_weight, p, threshold=1.):
    edge_weight = edge_weight / edge_weight.mean() * (1. - p)
    edge_weight = edge_weight.where(edge_weight > (1. - threshold), torch.ones_like(edge_weight) * (1. - threshold)) 
    edge_weight = edge_weight.where(edge_weight < 1, torch.ones_like(edge_weight) * 1) 
    sel_mask = torch.bernoulli(edge_weight).to(torch.bool)
    return edge_index[:, sel_mask] 


def NCNAM(feature, nc, p, max_threshold: float = 0.7):
    x = feature.abs() 
    device = feature.device
    w = x.t() @ torch.tensor(nc,dtype=torch.float).to(device)
    w = w.log() 
    w = (w.max() - w) / (w.max() - w.min()) 
    w = w / w.mean() * p
    w = w.where(w < max_threshold, max_threshold * torch.ones(1).to(device))
    w = w.where(w > 0, torch.zeros(1).to(device))
    drop_mask = torch.bernoulli(w).to(torch.bool)
    feature = feature.clone()
    feature[:, drop_mask] = 0. 
    return feature

def transition(communities, num_nodes):
    classes = np.full(num_nodes, -1)
    for i, node_list in enumerate(communities):
        classes[np.asarray(node_list)] = i
    return classes

def dynamic_community_strength(g, communities, json_file, alpha, beta):
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    community_dict = {}
    node_strengths = np.zeros(g.number_of_nodes())

    for node, community in enumerate(communities):
        key = tuple(community) if isinstance(community, list) else community
        if key not in community_dict:
            community_dict[key] = []
        community_dict[key].append(node)

    community_strengths = {}
    for community, nodes in community_dict.items():
        community_subgraph = g.subgraph(nodes)

        if community_subgraph.number_of_edges() > 0:
            conductance = nx.conductance(g, community_subgraph)
        else:
            conductance = 1
        community_pagerank_sum = sum(data[str(node)]['pagerank'] for node in nodes)
        community_pagerank_avg = community_pagerank_sum / len(nodes)
        community_strengths[community] = alpha * conductance + beta *1e4*community_pagerank_avg
    for node in g.nodes():
        for community in community_dict.keys():
            if node in list(community):
                
                node_hub_score = data[str(node)]['hub_score']

                node_community_strength = community_strengths[community]
                node_strengths[node] = node_community_strength * node_hub_score
                break

    return community_strengths, node_strengths    

def get_edge_weight(edge_index: torch.Tensor,
                    com: np.ndarray,
                    nc:np.ndarray) -> torch.Tensor:
    edge_mod =lambda x: float(nc[x[0]])/2+float(nc[x[1]])/2 if x[0] == x[1] else -(float(nc[x[0]])/2 + float(nc[x[1]]))/2
    normalize = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    edge_weight = np.asarray([edge_mod([com[u.item()], com[v.item()]]) for u, v in edge_index.T])
    edge_weight = normalize(edge_weight)
    return torch.from_numpy(edge_weight).to(edge_index.device)