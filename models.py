import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import (NNConv, GMMConv, GraphConv, Set2Set)
from torch_geometric.nn import (SplineConv, graclus, max_pool, max_pool_x, global_mean_pool)



class MoNet(nn.Module):
    def __init__(self, num_features, kernel=3, dim=3):
        super(MoNet, self).__init__()
        self.conv1 = GMMConv(in_channels=num_features, out_channels=8, dim=dim, kernel_size=kernel)
        self.conv2 = GMMConv(in_channels=8, out_channels=16, dim=dim, kernel_size=kernel)
        self.conv3 = GMMConv(in_channels=16, out_channels=8, dim=dim, kernel_size=kernel)
        self.conv4 = GMMConv(in_channels=8, out_channels=4, dim=dim, kernel_size=kernel)
        
    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv3(data.x, data.edge_index, data.edge_attr))
        data.x = self.conv4(data.x, data.edge_index, data.edge_attr)
        return data.x


class SplineBlock(nn.Module):
    def __init__(self, num_in_features, num_outp_features, mid_features, kernel=3, dim=3):
        super(SplineBlock, self).__init__()
        self.conv1 = SplineConv(num_in_features, mid_features, dim, kernel, is_open_spline=False)
        self.batchnorm1 = torch.nn.BatchNorm1d(mid_features)
        self.conv2 = SplineConv(mid_features, 2 * mid_features, dim, kernel, is_open_spline=False)
        self.batchnorm2 = torch.nn.BatchNorm1d(2 * mid_features)
        self.conv3 = SplineConv(2 * mid_features + 3, num_outp_features, dim, kernel, is_open_spline=False)
  
    def forward(self, data):
        res = F.elu(self.batchnorm1(self.conv1(data.x, data.edge_index, data.edge_attr)))
        res = F.elu(self.batchnorm2(self.conv2(res, data.edge_index, data.edge_attr)))
        res = torch.cat([res, data.pos], dim=1)
        res = self.conv3(res, data.edge_index, data.edge_attr)
        return res


class SplineCNN(nn.Module):
    def __init__(self, num_features, kernel=3, dim=3):
        super(SplineCNN, self).__init__()
        self.block1 = SplineBlock(num_features, 4, 2, kernel, dim)

    def forward(self, data):
        res = self.block1(data)
        return res
    
class SplineCNN2(nn.Module):
    def __init__(self, num_features, kernel=3, dim=3):
        super(SplineCNN2, self).__init__()
        self.block1 = SplineBlock(num_features, 16, 4, kernel, dim)
        self.block2 = SplineBlock(16, 4, 32, kernel, dim)

    def forward(self, data):
        data.x = self.block1(data)
        data.x = self.block2(data)
        return data.x
    
class SplineCNN4(nn.Module):
    def __init__(self, num_features, kernel=3, dim=3):
        super(SplineCNN4, self).__init__()
        self.block1 = SplineBlock(num_features, 16, 8, kernel, dim)
        self.block2 = SplineBlock(16, 64, 32, kernel, dim)
        self.block3 = SplineBlock(64, 64, 128, kernel, dim)
        self.block4 = SplineBlock(64, 4, 16, kernel, dim)

    def forward(self, data):
        data.x = self.block1(data)
        data.x = self.block2(data)
        data.x = self.block3(data)
        data.x = self.block4(data)
        return data.x

class MoNet(nn.Module):
    def __init__(self, num_features, kernel=3, dim=3):
        super(MoNet, self).__init__()
        self.conv1 = GMMConv(in_channels=num_features, out_channels=8, dim=dim, kernel_size=kernel)
        self.conv2 = GMMConv(in_channels=8, out_channels=16, dim=dim, kernel_size=kernel)
        self.conv3 = GMMConv(in_channels=16, out_channels=8, dim=dim, kernel_size=kernel)
        self.conv4 = GMMConv(in_channels=8, out_channels=4, dim=dim, kernel_size=kernel)
        
    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        data.x = F.elu(self.conv3(data.x, data.edge_index, data.edge_attr))
        data.x = self.conv4(data.x, data.edge_index, data.edge_attr)
        return data.x
    
    
class SplineCNN4Pooling(nn.Module):
    def __init__(self, num_features, kernel=3, dim=3):
        super(SplineCNN4Pooling, self).__init__()
        self.block1 = SplineBlock(num_features, 16, 8, kernel, dim)
        self.block2 = SplineBlock(16, 64, 32, kernel, dim)
        self.block3 = SplineBlock(64, 64, 128, kernel, dim)
        self.block4 = SplineBlock(64, 4, 16, kernel, dim)
        self.fc1 = torch.nn.Linear(64, 80)
        self.fc2 = torch.nn.Linear(80, 12)

    def forward(self, data):
        data.x = self.block1(data)
        data.x = self.block2(data)
        data.x = self.block3(data)
        verticies_predictions = self.block4(data)
        
        global_prediction = global_mean_pool(data.x, data.batch)
        global_prediction = F.elu(self.fc1(global_prediction))
        global_prediction = F.elu(self.fc2(global_prediction))
       
        return verticies_predictions, global_prediction
