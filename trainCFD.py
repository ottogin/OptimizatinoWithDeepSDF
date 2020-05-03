import os
import sys
import time
import numpy as np
import json
import argparse
from tqdm import tqdm

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


ALPHA = 1/4


def getR2Score(y, pred):
    mean_data = np.mean(y.numpy())
    sstot = np.sum((y.numpy() - mean_data) ** 2)
    ssreg = np.sum((pred.numpy() - mean_data) ** 2)
    ssres = np.sum((pred.numpy() - y.numpy()) ** 2)
    return 1 - ssres / sstot 

def train(epoch, model, train_loader, device, optimizer):
    model.train()

    for data in tqdm(train_loader, leave=False):
        data = data.to(device)
        optimizer.zero_grad()
        #verticies_preds, global_preds = model(data)
        verticies_preds = model(data)
        
        vert_loss = F.mse_loss(data.y, verticies_preds)
#         global_loss = F.mse_loss(global_preds[:, :3], data.pressure_drag.reshape(-1, 3)) +\
#                       F.mse_loss(global_preds[:, 3:6], data.viscous_drag.reshape(-1, 3)) +\
#                       F.mse_loss(global_preds[:, 6:9], data.pressure_moment.reshape(-1, 3)) +\
#                       F.mse_loss(global_preds[:, 9:], data.viscous_moment.reshape(-1, 3))

        loss = vert_loss
        
        loss.backward()
        optimizer.step()

#         print('Train Loss: {:.4f}'.format(loss))
    del data, loss, verticies_preds

def validate(model, test_loader, device):
    model.eval()
    loss = 0
    r2_score = 0

    for data in tqdm(test_loader):
        data = data.to(device)
        
        #verticies_preds, global_preds = model(data)
        verticies_preds = model(data)

        vert_loss = F.mse_loss(verticies_preds, data.y)
#         global_loss = F.mse_loss(global_preds[:, :3], data.pressure_drag.reshape(-1, 3)) +\
#                       F.mse_loss(global_preds[:, 3:6], data.viscous_drag.reshape(-1, 3)) +\
#                       F.mse_loss(global_preds[:, 6:9], data.pressure_moment.reshape(-1, 3)) +\
#                       F.mse_loss(global_preds[:, 9:], data.viscous_moment.reshape(-1, 3))

        r2_score += getR2Score(data.y[:, 0].cpu(), verticies_preds[:, 0].cpu().detach())
        curr_loss = vert_loss #(vert_loss + ALPHA * global_loss) / 2
        
        loss += curr_loss.cpu().detach().numpy()
    return loss / len(test_loader), r2_score / len(test_loader)

def process_model(network, out_file_name, train_loader, validation_loader,
                  init_lr=0.1, num_epochs=150, cont=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = network(train_loader.dataset.num_features).to(device)
    
    if cont:
        model_path = "Expirements/" + out_file_name + ".nn"
        model.load_state_dict(torch.load(model_path))
        print('Continuing model from ', model_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.8, patience=3, min_lr=0.00001, verbose=True)
   
    start_time = time.time()
    print('Start training')
    for epoch in tqdm(range(num_epochs)):
        train(epoch, model, train_loader, device, optimizer)
        test_acc, r2_test = validate(model, validation_loader, device)
        scheduler.step(test_acc)
        with open("Expirements/" + out_file_name, 'a') as file:
            print('Epoch: {:02d}, Time: {:.4f}, Validation Accuracy: {:.4f}, R2 score pressure {:.4f}'\
                  .format(epoch, time.time() - start_time, test_acc, r2_test), file=file)
            
        torch.save(model.state_dict(), "Expirements/" + out_file_name + ".nn")

#         start_time = time.time()
#         test_acc = validate(model, test_loader, device)
#         print('Test, Time: {:.4f}, Accuracy: {:.4f}'\
#               .format(time.time() - start_time, test_acc))

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a CFD Network')
    parser.add_argument('-b', metavar='batch_size', type=int, default=1,
                        help='batch size')
    parser.add_argument('-c', help='Continue training', action='store_true', default=False)
    
    parser.add_argument('model', metavar='model', type=str,
                        help='model class name')
    parser.add_argument('name', metavar='name', type=str,
                        help='name of the expirement')
    
    args = parser.parse_args()
    

#     train_dataset = CDFDatasetInMemory('/cvlabdata2/home/artem/Data/cars_refined/simulated', 
#                                        split='examples/splits/sv2_cars_clear_train.json') #, delimetr=0.002
#     val_dataset   = CDFDatasetInMemory('/cvlabdata2/home/artem/Data/cars_refined/simulated', 
#                                        split='examples/splits/sv2_cars_clear_test.json', 
#                                        train=False) #, delimetr=0.999
    train_dataset = CDFDatasetInMemory('/cvlabdata2/home/artem/Data/cars_orig/simulated', 
                                       split='examples/splits/sv2_cars_clear_train.json') #, delimetr=0.002
    val_dataset   = CDFDatasetInMemory('/cvlabdata2/home/artem/Data/cars_orig/simulated', 
                                       split='examples/splits/sv2_cars_clear_train.json', 
                                       train=False) #, delimetr=0.999

    train_loader = torch_geometric.data.DataLoader(train_dataset, batch_size=args.b, shuffle=False)
    val_loader = torch_geometric.data.DataLoader(val_dataset, batch_size=args.b, shuffle=False)
    
    exec("model_class = %s" % args.model)
    
    model = process_model(model_class, args.name, train_loader, val_loader, init_lr=0.02, cont=args.c)
    
    print("FINISHED!")
    