import os.path as osp
from torch_geometric.datasets import Planetoid, WikiCS, Coauthor, Amazon
import torch_geometric.transforms as T




def get_dataset(path, name):
    assert name in ['WikiCS', 'Coauthor-CS', 'Amazon-Computers', 'Amazon-Photo']
    root_path = './datasets'

    if name == 'Coauthor-CS':
        return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())
    if name == 'WikiCS':
        return WikiCS(root=path)
    if name == 'Amazon-Computers':
        return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())
    if name == 'Amazon-Photo':
        return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())