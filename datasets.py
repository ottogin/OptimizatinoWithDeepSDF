import os
import sys
import time
import numpy as np
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import (NNConv, GMMConv, GraphConv, Set2Set)
from torch_geometric.nn import (SplineConv, graclus, max_pool, max_pool_x, global_mean_pool)

#from neuralnet_pytorch.metrics import chamfer_loss

import trimesh

from visualization_utils import plot_mesh_3d

from models import *

from sklearn.neighbors import KDTree

def make_data_instance_from_fld(fld_path, k=10, replace_inf=10e5, data_step=1) -> torch_geometric.data.Data:
        ''' Takes a path to generated fld file with following colomns: x,y,z,p,k,omega,nut 
            and converts it into a geometric data instance.
        '''
        
        fld = np.genfromtxt(fld_path, delimiter=',', skip_header=1)
        np.random.shuffle(fld)
        fld = fld[::data_step]
        fld[fld > 10e5] = np.nan
        fld = fld[~np.isnan(fld).any(axis=1)]
    
        answers = fld[:, 3:]
        
        mean_values = [-1.90357614e+00, 9.55119907e-02, 2.05472217e+02, 5.53618263e-05]
        std_values = [3.71674873e+00, 4.93675056e-02, 1.10871494e+02, 2.63155496e-05]
        for f in range(answers.shape[1]):
            answers[:, f] = (answers[:, f] - mean_values[f]) / std_values[f]
        
        if k > 0:
            # find correspondences
            fld_tree = KDTree(fld[:, :3])
            distances, indeces = fld_tree.query(fld[:, :3], k=10)
            # create edge indicies
            edge_indices = np.array([[(idx, indeces[idx][i]) for i in range(k)] for idx in range(len(fld))])
            edge_indices = edge_indices.reshape( k * len(fld), 2)
            edge_attr    = np.array([[fld[idx, :3] - fld[indeces[idx][i]][:3] for i in range(k)] 
                                     for idx in range(len(fld))])
            edge_attr    = edge_attr.reshape( k * len(fld), 3)
            
        else:
            edge_indices = np.array([[0,0]])
            edge_attr    = np.array([0])
            
        data = torch_geometric.data.Data(x  = torch.tensor(fld[:, :3], dtype=torch.float), 
                                         pos= torch.tensor(fld[:, :3], dtype=torch.float), 
                                         y  = torch.tensor(fld[:, 3:], dtype=torch.float),
                                         edge_attr = torch.tensor(edge_attr, dtype=torch.float),
                                         edge_index= torch.tensor(edge_indices, dtype=torch.long).t().contiguous())
        
        scr_path = fld_path.replace('fld', 'scr', 1).replace('fld', 'json')
        with open(scr_path) as scr_file:
            scr_data = json.load(scr_file)
            
        global_means = [ 7.15702765e-01, 9.76291022e-03, -1.97037022e-04,
                         4.33680848e-02, 2.71446501e-03,  2.42610894e-05,
                        -1.63100377e-05, -1.20658604e-03, 2.01998814e-01,
                        -2.94244062e-06, 1.35224581e-05, -6.22179022e-04]
        global_std =   [ 3.12011511e-01, 2.76790047e-01, 4.93472812e-02,
                         7.02184919e-03, 1.78783928e-03, 8.31190054e-04,
                         4.32590171e-03, 1.71780821e-02, 1.01220579e-01,
                         1.13513395e-04, 2.45400068e-04, 1.05765967e-03]

        data.pressure_drag = torch.tensor((scr_data['pressure_drag'] - global_means[:3]) / 
                                                global_std[:3], dtype=torch.float)
        data.viscous_drag = torch.tensor((scr_data['viscous_drag'] - global_means[3:6]) / 
                                                global_std[3:6], dtype=torch.float)
        data.pressure_moment = torch.tensor((scr_data['pressure_moment'] - global_means[6:9]) / 
                                                global_std[6:9], dtype=torch.float)
        data.viscous_moment = torch.tensor((scr_data['viscous_moment'] - global_means[9:]) / 
                                                global_std[9:], dtype=torch.float)

        return data


class CDFDataset(torch_geometric.data.Dataset):
    
    def __init__(self, root, transform=None, pre_transform=None, 
                             train=True, delimetr=0.9, connectivity=10, data_step=1):
        super(CDFDataset, self).__init__(root, transform, pre_transform)
        
        self.objects = list()
        for (dirpath, dirnames, filenames) in os.walk(root):
            self.objects += [os.path.join(dirpath, file) for file in filenames if file[-4:] == '.fld']
        
        delimetr = int(delimetr * len(self.objects))
        if train:
            self.objects = self.objects[:delimetr]
        else:
            self.objects = self.objects[delimetr:]
        
        self.connectivity = connectivity
        self.data_step = data_step

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []
    
    def __len__(self):
        return len(self.objects)

    def get(self, idx):
        return make_data_instance_from_fld(self.objects[idx], self.connectivity,  data_step=self.data_step)