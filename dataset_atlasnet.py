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

sys.path.append("/cvlabdata2/home/artem/DeepSDF/atlasnet_retrained")

import dependencies.dataset_shapenet as dataset_shapenet
from dependencies.model import EncoderDecoder
import dependencies.argument_parser as argument_parser

from easydict import EasyDict
import pdb
import pymesh

#from neuralnet_pytorch.metrics import chamfer_loss

import trimesh

from visualization_utils import plot_mesh_3d

from models import *

from sklearn.neighbors import KDTree

def generateNmaesMapping(root):
    mapping = {}
    objects = []
    for (dirpath, dirnames, filenames) in os.walk(os.path.join(root, 'scr')):
        objects += [(os.path.join(dirpath, file), file[:-5]) for file in filenames if file[-5:] == '.json' and file[0] != '.']
    for path, name in objects:
        with open(path, 'r') as json_data:
             origId = json.load(json_data)['stl_id']
        mapping[name] = origId
    return mapping


def make_mesh_from_mesh_atlas(points, faces):
    transformed_points = points

    edges = trimesh.geometry.faces_to_edges(faces)
    np_points = transformed_points.clone().cpu().detach().numpy()
    edge_attr = [np_points[a] - np_points[b] for a, b in edges]

    data = torch_geometric.data.Data(x  = transformed_points, 
                                     pos= transformed_points, 
                                     face = torch.tensor(faces, 
                                                     dtype=torch.long).to('cpu').t(),
                                     edge_attr=torch.tensor(edge_attr, dtype=torch.float).to('cpu'),
                                     edge_index=torch.tensor(edges, dtype=torch.long).to('cpu').t().contiguous())
    return data

def make_data_instance_from_atlas_rec(decoder, fld_path, replace_inf=10e5, 
                                      batch_norm=False, normilize=True) -> torch_geometric.data.Data:
    
    latent_idx = fld_path.split('/')[-1][:4]
    
    # init
    latent_filename = os.path.join(
        "/cvlabdata2/home/artem/DeepSDF/atlasnet_retrained/reconstructions/codes", latent_idx + ".pth"
    )
    latent = torch.load(latent_filename).squeeze(0).clone().detach().requires_grad_(True)
    latent.requires_grad = True
    
    verts, faces = decoder.module.generate_mesh(latent.unsqueeze(0))
    mesh = make_mesh_from_mesh_atlas(verts, faces)
    
    fld = np.genfromtxt(fld_path, delimiter=',', skip_header=1)
    np.random.shuffle(fld)
    fld[fld > 10e5] = np.nan
    fld = fld[~np.isnan(fld).any(axis=1)]

    answers = fld[:, 3:]

    if normilize:
        for f in range(answers.shape[1]):
            answers[:, f] = (answers[:, f] - np.mean(answers[:, f])) / np.std(answers[:, f])


    fld_tree = KDTree(fld[:, :3])
    distances, indeces = fld_tree.query(mesh.x.cpu().detach().numpy(), k=3)
    interpolations = np.mean(answers[indeces], axis=1)
    
    mesh.y = torch.tensor(interpolations, dtype=torch.float)
    
    return mesh

def make_data_instance_from_atlas_cached(fld_path, mapping,replace_inf=10e5, 
                                      batch_norm=False, normilize=True) -> torch_geometric.data.Data:
    
    fld_lat_idx = fld_path.split('/')[-1][:9]
    latent_idx = '%04d' % int(mapping[fld_lat_idx])
    
#     print("Mapped: ", fld_lat_idx, " to ", latent_idx)
    mesh_ply_path = '/cvlabdata2/home/artem/DeepSDF/atlasnet_retrained/reconstructions/meshes/' + latent_idx + '.ply'
    mesh = trimesh.load(mesh_ply_path)
    
    fld = np.genfromtxt(fld_path, delimiter=',', skip_header=1)
    np.random.shuffle(fld)
    fld[fld > 10e5] = np.nan
    fld = fld[~np.isnan(fld).any(axis=1)]

    answers = fld[:, 3:]

    if normilize:
        for f in range(answers.shape[1]):
            answers[:, f] = (answers[:, f] - np.mean(answers[:, f])) / np.std(answers[:, f])

    fld_tree = KDTree(fld[:, :3])
    distances, indeces = fld_tree.query(mesh.vertices, k=3)
    interpolations = np.mean(answers[indeces], axis=1)
    
    edge_attr = [mesh.vertices[a] - mesh.vertices[b] for a, b in mesh.edges]
    data = torch_geometric.data.Data(x  = torch.tensor(mesh.vertices, dtype=torch.float), 
                                     pos= torch.tensor(mesh.vertices, dtype=torch.float),
                                     face = torch.tensor(mesh.faces, dtype=torch.long).t(),
                                     y  = torch.tensor(interpolations, dtype=torch.float),
                                     edge_attr = torch.tensor(edge_attr, dtype=torch.float),
                                     edge_index= torch.tensor(mesh.edges, dtype=torch.long).t().contiguous())
    
    return data



class CDFAtlasDatasetInMemory(torch_geometric.data.InMemoryDataset):
    
    def __init__(self, root, transform=None, pre_transform=None, 
                             train=True, delimetr=0.95):
        self.delimetr = delimetr
        self.train = train
        
        super(CDFAtlasDatasetInMemory, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        if self.train:
            return ['data15_atlas_mid_cor_map_train.pt']
        else:
            return ['data15_atlas_mid_cor_map_test.pt']

    def process(self):
  
        self.objects = list()
        for (dirpath, dirnames, filenames) in os.walk(self.root):
            self.objects += [os.path.join(dirpath, file) for file in filenames if file[-4:] == '.fld']
        
        delimetr = int(self.delimetr * len(self.objects))
        if self.train:
            self.objects = self.objects[:delimetr]
        else:
            self.objects = self.objects[delimetr:]
        
        print('Generating ', len(self.objects), " objects")
#         sys.argv = ['foo']
#         opt = argument_parser.parser()
#         torch.cuda.set_device(opt.multi_gpu[0])

#         """
#         load pre-trained model
#         """
#         OUT_PATH = "/cvlabdata2/home/artem/DeepSDF/atlasnet_retrained/optimizations/"
#         MODEL_PATH = "/cvlabdata2/home/artem/DeepSDF/atlasnet_retrained/trained/network.pth"

#         if torch.cuda.is_available():
#             opt.device = torch.device(f"cuda:{opt.multi_gpu[0]}")
#         else:
#             # Run on CPU
#             opt.device = torch.device(f"cpu")

#         network = EncoderDecoder(opt)
#         network = nn.DataParallel(network, device_ids=opt.multi_gpu)
#         # load weights
#         network.load_state_dict(torch.load(MODEL_PATH, map_location='cuda:0'))
#         # finally keep only decoder
#         decoder = nn.DataParallel(network.module.decoder)
#         # and put in eval mode
#         decoder = decoder.eval()
        
        mapping = generateNmaesMapping(self.root[:-3])
        data_list = []
        for obj in self.objects:
            if int(mapping[obj.split('/')[-1][:9]]) > 800:
                continue
            data_list += [ make_data_instance_from_atlas_cached(obj, mapping) ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        
        
        
        
# class CDFDatasetAtlas(torch_geometric.data.Dataset):
#     def __init__(self, root, mapping, transform=None, pre_transform=None, 
#                              train=True, delimetr=0.9, connectivity=10, data_step=1, split=None):
#         super(CDFDatasetAtlas, self).__init__(root, transform, pre_transform)
        
#         self.objects = list()
#         if split is not None:
#             with open(split, 'r') as json_data:
#                 objNmae = json.load(json_data)['ShapeNetV2']['02958343']
#                 self.objects += [os.path.join(root, objName + '.fld')]
#         else:
#             for (dirpath, dirnames, filenames) in os.walk(root):
#                 self.objects += [os.path.join(dirpath, file) for file in filenames if file[-4:] == '.fld']

#         delimetr = int(delimetr * len(self.objects))
#         if train:
#             self.objects = self.objects[:delimetr]
#         else:
#             self.objects = self.objects[delimetr:]
        
#         self.connectivity = connectivity
#         self.data_step = data_step

#     @property
#     def raw_file_names(self):
#         return []

#     @property
#     def processed_file_names(self):
#         return []
    
#     def __len__(self):
#         return len(self.objects)

#     def get(self, idx):
#         #return make_data_instance_from_fld(self.objects[idx], self.connectivity,  data_step=self.data_step)
#         return make_data_instance_from_atlas_rec(self.objects[idx], self.connectivity,  data_step=self.data_step)

