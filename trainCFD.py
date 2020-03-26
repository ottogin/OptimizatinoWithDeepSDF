import os
import sys
import time
import numpy as np
import json
import argparse
from tqdm import tqdm

import matplotlib.pyplot as plt 
%matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import (NNConv, GMMConv, GraphConv, Set2Set)
from torch_geometric.nn import (SplineConv, graclus, max_pool, max_pool_x, global_mean_pool)

import trimesh

from visualization_utils import plot_mesh_3d

from models import *
from datasets import *


def train(epoch, model, train_loader, device, optimizer):
    model.train()

    for data in tqdm(train_loader, leave=False):
        data = data.to(device)
        optimizer.zero_grad()
        verticies_preds, global_preds = model(data)
        
        vert_loss = F.mse_loss(verticies_preds, data.y)
        global_loss = F.mse_loss(global_preds[:, :3], data.pressure_drag.reshape(-1, 3)) +\
                      F.mse_loss(global_preds[:, 3:6], data.viscous_drag.reshape(-1, 3)) +\
                      F.mse_loss(global_preds[:, 6:9], data.pressure_moment.reshape(-1, 3)) +\
                      F.mse_loss(global_preds[:, 9:], data.viscous_moment.reshape(-1, 3))

        loss = (vert_loss + global_loss) / 2
        
        loss.backward()
        optimizer.step()

        #print('Train Loss: {:.4f}'.format(loss))
    del data, loss, responce

def validate(model, test_loader, device):
    model.eval()
    loss = 0

    for data in tqdm(test_loader):
        data = data.to(device)
        
        verticies_preds, global_preds = model(data)

        vert_loss = F.mse_loss(verticies_preds, data.y)
        global_loss = F.mse_loss(global_preds[:, :3], data.pressure_drag.reshape(-1, 3)) +\
                      F.mse_loss(global_preds[:, 3:6], data.viscous_drag.reshape(-1, 3)) +\
                      F.mse_loss(global_preds[:, 6:9], data.pressure_moment.reshape(-1, 3)) +\
                      F.mse_loss(global_preds[:, 9:], data.viscous_moment.reshape(-1, 3))

        curr_loss = (vert_loss + global_loss) / 2
        
        loss += curr_loss.cpu().detach().numpy()
    return loss / len(test_loader)

def process_model(network, out_file_name, train_loader, validation_loader,
                  init_lr=0.1, num_epochs=150):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = network(train_loader.dataset.num_features).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.8, patience=3, min_lr=0.00001, verbose=True)
   
    start_time = time.time()
    print('Start training')
    for epoch in tqdm(range(num_epochs)):
        train(epoch, model, train_loader, device, optimizer)
        test_acc = validate(model, validation_loader, device)
        scheduler.step(test_acc)
        with open("Expirements/" + out_file_name, 'a') as file:
            print('Epoch: {:02d}, Time: {:.4f}, Validation Accuracy: {:.4f}'\
                  .format(epoch, time.time() - start_time, test_acc), file=file)
            
        torch.save(model.state_dict(), "Expirements/" + out_file_name + ".nn")

#         start_time = time.time()
#         test_acc = validate(model, test_loader, device)
#         print('Test, Time: {:.4f}, Accuracy: {:.4f}'\
#               .format(time.time() - start_time, test_acc))

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a CFD Network')
    parser.add_argument('integers', metavar='N', type=int, nargs='+',
                        help='an integer for the accumulator')
    parser.add_argument('--sum', dest='accumulate', action='store_const',
                        const=sum, default=max,
                        help='sum the integers (default: find the max)')

    args = parser.parse_args()
    print(args.accumulate(args.integers))

    train_dataset = CDFDataset('/cvlabsrc1/cvlab/dataset_shapenet/code/foam_npy/fld', 
                               connectivity=10, data_step=10) #, delimetr=0.002
    val_dataset   = CDFDataset('/cvlabsrc1/cvlab/dataset_shapenet/code/foam_npy/fld',
                               connectivity=10, train=False, data_step=10) #, delimetr=0.999

    train_loader = torch_geometric.data.DataLoader(train_dataset, batch_size=2, shuffle=False)
    val_loader = torch_geometric.data.DataLoader(val_dataset, batch_size=2, shuffle=False)