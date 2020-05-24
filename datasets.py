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


def compute_traingle_area(a, b, c):
    side1 = torch.sum((a - b) ** 2) ** (1/2)
    side2 = torch.sum((a - c) ** 2) ** (1/2)
    side3 = torch.sum((c - b) ** 2) ** (1/2)
    p = (side1 + side2 + side3) / 2
    return (p * (p - side1) * (p - side2) * (p - side3)) ** (1/2)

def compute_normal(a, b, c, mc):
    norm = torch.cross(a - b, a - c)
    if torch.dot(mc - a, norm) > 0:
        norm = -norm
    return norm

def compute_lift(data_instance, answers):
    
    fld_tree = KDTree(data_instance.x.cpu().detach())
    mass_center = torch.mean(data_instance.x, axis=0)
    distances, indeces = fld_tree.query(data_instance.x.cpu().detach(), k=4)
    areas = [compute_traingle_area(data_instance.x[a], data_instance.x[b], data_instance.x[c]) 
                for a, b, c in indeces[:, 1:]]
    normals = [compute_normal(data_instance.x[a], data_instance.x[b], data_instance.x[c], mass_center) 
                for a, b, c in indeces[:, 1:]]

    mult = torch.tensor([a * n[2] for a, n in zip(areas, normals)]).to('cuda:0')
    lift = answers[:, 0] * mult
    return torch.sum(lift[~torch.isnan(lift)])


#     areas = [torch.mean(torch.tensor([compute_traingle_area(data_instance.x[data_instance.faces[f][0]], 
#                                             data_instance.x[data_instance.faces[f][1]], 
#                                             data_instance.x[data_instance.faces[f][2]]) 
#                 for f in  faces if f > 0])) for faces in data_instance.vertex_faces[0] ]


def compute_lift_faces(data_instance, answers, axis=0):
    
    mesh = trimesh.Trimesh(vertices=data_instance.x.cpu().detach(), faces=data_instance.face.t().cpu().detach())
    #mass_center = torch.mean(data_instance.x, axis=0)
    
    pressures = torch.mean(answers[data_instance.face, 0], axis=0) 
    mult = torch.tensor( - mesh.area_faces * mesh.face_normals[:, axis], dtype=torch.float).to('cuda:0')
    lift = torch.mul(pressures, mult)
    return torch.sum(lift[~torch.isnan(lift)])

def compute_lift_faces_signs(data_instance, answers, axis=0):
    mesh = trimesh.Trimesh(vertices=data_instance.x.cpu().detach(), faces=data_instance.face.t().cpu().detach())
    #mass_center = torch.mean(data_instance.x, axis=0)

    pressures = torch.mean(answers[data_instance.face, 0], axis=0) 
    signs = np.sign(np.sum(np.sum(mesh.vertices[mesh.faces], axis=1) * mesh.face_normals, axis=1))
    mult = torch.tensor( - mesh.area_faces * mesh.face_normals[:, axis] * signs, dtype=torch.float).to('cuda:0')
    lift = torch.mul(pressures, mult)
    return torch.sum(lift[~torch.isnan(lift)])


def compute_lift_faces_diff(data_instance, answers, axis=0):
    pressures = torch.mean(answers[data_instance.face, 0], axis=0)

    # TODO: cahnge to x if needed
    pos = data_instance.x
    cross_prod = (pos[data_instance.face[1]] - pos[data_instance.face[0]]).cross(
                  pos[data_instance.face[2]] - pos[data_instance.face[0]])
    mult = - cross_prod[:, axis] / 2
    lift = torch.mul(pressures, mult)
    return torch.sum(lift[~torch.isnan(lift)])

def compute_lift_faces_diff_signs(data_instance, answers, axis=0):
    pressures = torch.mean(answers[data_instance.face, 0], axis=0)

    # TODO: cahnge to x if needed
    pos = data_instance.x
    cross_prod = (pos[data_instance.face[1]] - pos[data_instance.face[0]]).cross(
                  pos[data_instance.face[2]] - pos[data_instance.face[0]])
    
    signs = torch.sign(torch.sum(pos[data_instance.face[0]] * cross_prod, axis=1))
    mult = - cross_prod[:, axis] * signs / 2
    lift = torch.mul(pressures, mult)
    return torch.sum(lift[~torch.isnan(lift)])

def compute_signs_for_loss(data_instance, normals):
    pos = data_instance.x
    cross_prod = (pos[data_instance.face[1]] - pos[data_instance.face[0]]).cross(
                  pos[data_instance.face[2]] - pos[data_instance.face[0]])
    face_normals = torch.mean(normals[faces], axis=1)
    return torch.sign(torch.sum(face_normals * cross_prod, axis=1))

def compute_lift_faces_diff_mem_signs(data_instance, answers, signs, axis=0):
    pressures = torch.mean(answers[data_instance.face, 0], axis=0)

    # TODO: cahnge to x if needed
    pos = data_instance.x
    cross_prod = (pos[data_instance.face[1]] - pos[data_instance.face[0]]).cross(
                  pos[data_instance.face[2]] - pos[data_instance.face[0]])
    mult = - cross_prod[:, axis] * signs / 2
    lift = torch.mul(pressures, mult)
    return torch.sum(lift[~torch.isnan(lift)])


def make_data_instance_from_stl(fld_path, replace_inf=10e5, batch_norm=False) -> torch_geometric.data.Data:
        ''' Takes a path to generated fld file with following colomns: x,y,z,p,k,omega,nut 
            and converts it into a geometric data instance.
        '''
        
        fld = np.genfromtxt(fld_path, delimiter=',', skip_header=1)
        np.random.shuffle(fld)
        fld[fld > 10e5] = np.nan
        fld = fld[~np.isnan(fld).any(axis=1)]
    
        answers = fld[:, 3:]

#         if (batch_norm):
#             mean_values = answers.mean(axis=0)
#             std_values = answers.std(axis=0)
#         else:

        mean_values = [-1.15994242e+01, 9.01274307e-01,  1.83840398e+03,  6.36532838e-05]
        std_values = [4.78920149e+01, 3.70121534e-01, 7.36068558e+02, 2.35466637e-05]
        for f in range(answers.shape[1]):
            #answers[:, f] = (answers[:, f] - mean_values[f]) / std_values[f]
            answers[:, f] = (answers[:, f] - np.mean(answers[:, f])) / np.std(answers[:, f])
            
        stl_path = fld_path.replace('fld', 'stl', 1)[:-9] + '.stl'
        mesh = trimesh.load(stl_path)
        
        fld_tree = KDTree(fld[:, :3])
        distances, indeces = fld_tree.query(mesh.vertices, k=1)
        interpolations = answers[indeces].squeeze()
        
        edge_attr = [mesh.vertices[a] - mesh.vertices[b] for a, b in mesh.edges]
        data = torch_geometric.data.Data(x  = torch.tensor(mesh.vertices, dtype=torch.float), 
                                         pos= torch.tensor(mesh.vertices, dtype=torch.float),
                                         face = torch.tensor(mesh.faces, dtype=torch.long).t(),
                                         y  = torch.tensor(interpolations, dtype=torch.float),
                                         edge_attr = torch.tensor(edge_attr, dtype=torch.float),
                                         edge_index= torch.tensor(mesh.edges, dtype=torch.long).t().contiguous())
        
        scr_path = fld_path.replace('fld', 'scr', 1).replace('fld', 'json')
        with open(scr_path) as scr_file:
            scr_data = json.load(scr_file)
            
        global_mean = np.array([ 7.15702765e-01, 9.76291022e-03, -1.97037022e-04,
                                 4.33680848e-02, 2.71446501e-03,  2.42610894e-05,
                                -1.63100377e-05, -1.20658604e-03, 2.01998814e-01,
                                -2.94244062e-06, 1.35224581e-05, -6.22179022e-04] )
        global_std =  np.array([ 3.12011511e-01, 2.76790047e-01, 4.93472812e-02,
                                 7.02184919e-03, 1.78783928e-03, 8.31190054e-04,
                                 4.32590171e-03, 1.71780821e-02, 1.01220579e-01,
                                 1.13513395e-04, 2.45400068e-04, 1.05765967e-03] )

        data.pressure_drag = torch.tensor((scr_data['pressure_drag'] - global_mean[None, :3]) / 
                                                global_std[None, :3], dtype=torch.float)
        data.viscous_drag = torch.tensor((scr_data['viscous_drag'] - global_mean[3:6]) / 
                                                global_std[None, 3:6], dtype=torch.float)
        data.pressure_moment = torch.tensor((scr_data['pressure_moment'] - global_mean[6:9]) / 
                                                global_std[None, 6:9], dtype=torch.float)
        data.viscous_moment = torch.tensor((scr_data['viscous_moment'] - global_mean[9:]) / 
                                                global_std[None, 9:], dtype=torch.float)
        data.path = fld_path

        return data


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
            
        global_mean = np.array([ 7.15702765e-01, 9.76291022e-03, -1.97037022e-04,
                                 4.33680848e-02, 2.71446501e-03,  2.42610894e-05,
                                -1.63100377e-05, -1.20658604e-03, 2.01998814e-01,
                                -2.94244062e-06, 1.35224581e-05, -6.22179022e-04] )
        global_std =  np.array([ 3.12011511e-01, 2.76790047e-01, 4.93472812e-02,
                                 7.02184919e-03, 1.78783928e-03, 8.31190054e-04,
                                 4.32590171e-03, 1.71780821e-02, 1.01220579e-01,
                                 1.13513395e-04, 2.45400068e-04, 1.05765967e-03] )

        data.pressure_drag = torch.tensor((scr_data['pressure_drag'] - global_mean[None, :3]) / 
                                                global_std[None, :3], dtype=torch.float)
        data.viscous_drag = torch.tensor((scr_data['viscous_drag'] - global_mean[3:6]) / 
                                                global_std[None, 3:6], dtype=torch.float)
        data.pressure_moment = torch.tensor((scr_data['pressure_moment'] - global_mean[6:9]) / 
                                                global_std[None, 6:9], dtype=torch.float)
        data.viscous_moment = torch.tensor((scr_data['viscous_moment'] - global_mean[9:]) / 
                                                global_std[None, 9:], dtype=torch.float)
        data.path = fld_path

        return data
    

def generateNamesMapping(root):
    mapping = {}
    objects = []
    for (dirpath, dirnames, filenames) in os.walk(os.path.join(root, 'scr')):
        objects += [(os.path.join(dirpath, file), file[:-5]) for file in filenames if file[-5:] == '.json' and file[0] != '.']
    for path, name in objects:
        with open(path, 'r') as json_data:
             origId = json.load(json_data)['stl_id']
        mapping[origId] = name

    return mapping

class CDFDataset(torch_geometric.data.Dataset):
    
    def __init__(self, root, transform=None, pre_transform=None, 
                             train=True, delimetr=0.9, connectivity=10, data_step=1, split=None):
        super(CDFDataset, self).__init__(root, transform, pre_transform)
        
        self.objects = list()
        if split is not None:
            with open(split, 'r') as json_data:
                objNmae = json.load(json_data)['ShapeNetV2']['02958343']
                self.objects += [os.path.join(root, objName + '.fld')]
        else:
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
        #return make_data_instance_from_fld(self.objects[idx], self.connectivity,  data_step=self.data_step)
        return make_data_instance_from_stl(self.objects[idx], self.connectivity,  data_step=self.data_step)
    

class CDFDatasetInMemory(torch_geometric.data.InMemoryDataset):
    
    def __init__(self, root, transform=None, pre_transform=None, 
                             train=True, delimetr=0.95):
        self.delimetr = delimetr
        self.train = train
        
        super(CDFDatasetInMemory, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        if self.train:
            return ['data15_train_full_normalized.pt']
        else:
            return ['data15_test_full_normalized.pt']

    def process(self):
        # Get list of meshes
#         if self.split is not None:
#             mapping = generateNamesMapping(self.root)
#             with open(self.split, 'r') as json_data:
#                 objNames = json.load(json_data)['ShapeNetV2']['02958343']
#             self.objects = [os.path.join(self.root, 'fld/' + mapping[name] + '.fld') for name in objNames if name in mapping.keys()]
#             print('Taken ' + str(len(self.objects)) + ' out of ' + str(len(objNames)))
      
        self.objects = list()
        for (dirpath, dirnames, filenames) in os.walk(self.root):
            self.objects += [os.path.join(dirpath, file) for file in filenames if file[-4:] == '.fld']
        
        delimetr = int(self.delimetr * len(self.objects))
        if self.train:
            self.objects = self.objects[:delimetr]
        else:
            self.objects = self.objects[delimetr:]
            
        print(len(self.objects))
    
        
        data_list = [ make_data_instance_from_stl(obj) for obj in self.objects]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
