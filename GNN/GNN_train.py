# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.data import Data
import random
import numpy as np
import copy
import json

city = 'newyork'
mode = 'tunning' #'init' or 'tunning'

# Chargement graph data
def loadData(path = 'graph_'+city+'.json'):

    file = open(path, 'rb')
    data = json.load(file)
    file.close()

    adj_matrix = data['adj']
    number_of_nodes = int(data['number_of_nodes'])
    shortest_costs = data['shortest_costs']
    
    print('Data loaded : ', path)
    
    return adj_matrix, number_of_nodes, shortest_costs


def generateTensors(start, end, waypoints, number_of_nodes, shortest_costs):
    node_list = []
    for ind in range(number_of_nodes):
        if ind == start:
            node_list.append((1.0, 0.0, 0.0, 1.0, shortest_costs[ind][end] ))
            continue
        if ind == end:
            node_list.append((0.0, 1.0, 0.0, shortest_costs[start][ind], 1.0))
            continue
        if ind in waypoints:           
            node_list.append((0.0, 0.0, 1.0, shortest_costs[start][ind], shortest_costs[ind][end]))
            continue
        node_list.append((0.0, 0.0, 0.0, shortest_costs[start][ind], shortest_costs[ind][end]))
    
    x = torch.tensor(node_list, dtype=torch.float)
    return x
    

class GCN(torch.nn.Module):
    def __init__(self, features):
        super().__init__()
    
        out_sizes = range(10, 200, 15)
        last_layer = 3
        self.core_features = out_sizes[last_layer]
    
        self.convs = torch.nn.ModuleList()
        
        prev_size = features
        for depth, size in enumerate(out_sizes[:last_layer+1]):
            self.convs.append(GraphConv(prev_size, size))
            prev_size = size

        self.relu = nn.ReLU()
        self.bn_node  = nn.BatchNorm1d(out_sizes[last_layer], momentum = 0.1, track_running_stats = False)

        hidden_size = 2048

        self.sigm = nn.Sigmoid()
        
        self.drop = nn.Dropout(p=0.2)
        self.edge_fc1 = nn.Linear(self.core_features*2, hidden_size)
        self.edge_fc2 = nn.Linear(hidden_size, 100)
        
        self.pooling = nn.MaxPool1d(100)
        
        
    def forward(self, data, waypoints):
        x, edge_index, edge_w = data.x, data.edge_index, data.edge_attr

        for unit_conv in self.convs:
            x = unit_conv(x, edge_index, edge_w)
            x = self.sigm(x)

        x = self.bn_node(x)
        nb_waypoints = len(waypoints)
        
        selected = torch.index_select(x, 0, waypoints)
        
        row = selected.repeat(1, nb_waypoints)
        row = row.view(nb_waypoints,nb_waypoints,self.core_features)
        col = selected.view(1,nb_waypoints*self.core_features)
        col = col.repeat(1, nb_waypoints, 1)
        col = col.view(nb_waypoints,nb_waypoints,self.core_features)
        vect = torch.stack((row, col),2)
        vect = vect.view(nb_waypoints,nb_waypoints,self.core_features*2)
        vect = self.edge_fc1(vect)
        vect = self.sigm(vect)
        vect = self.drop(vect)
        vect = self.edge_fc2(vect)
        vect = self.sigm(vect)
        vect = self.pooling(vect)
        for ind in range(nb_waypoints):
            vect[ind][ind] = torch.tensor([0.0]).to(torch.device('cuda'))
        
        vect = vect.transpose(0,2)        
        return vect

def main():
    adj_matrix, number_of_nodes, shortest_costs = loadData()
    
    print(number_of_nodes)
    
    edge_list = []
    edge_val = []
    
    for i in range(number_of_nodes):
        for j in range(number_of_nodes):
            if adj_matrix[i][j] > 0:
                edge_list.append((i,j))
                edge_list.append((j,i))
                edge_val.append(adj_matrix[i][j])
                edge_val.append(adj_matrix[i][j])
                
    edge_index = torch.tensor(edge_list, dtype=torch.long)
    edge_index = edge_index.t().contiguous()
    edge_attr = torch.tensor(edge_val, dtype=torch.float)
    

    database = []
    db_file = open('DB_train_'+city+'.csv', 'r')

    for line in db_file.readlines():
        elems = line.split(';')
        distance, start, end = float(elems[0]),int(elems[1]),int(elems[2])
        wps = [int(x) for x in elems[3:]]
        random.shuffle(wps) # in place shuffle of waypoints to reach
        x = generateTensors(start, end, wps, number_of_nodes, shortest_costs)
        
        truth_order = [int(x) for x in elems[3:]]
        waypoints = sorted(truth_order)
        edge_matrix = torch.zeros(len(waypoints), len(waypoints), dtype = torch.float32)
        for ind1, node1 in enumerate(waypoints):
            for ind2, node2 in enumerate(waypoints):
                if truth_order.index(node1) == truth_order.index(node2) + 1:
                    edge_matrix[ind1][ind2] = 1
                    edge_matrix[ind2][ind1] = 1
        edge_matrix = torch.unsqueeze(edge_matrix, 0)
        
        database.append((x, waypoints, edge_matrix, distance, start, end))
        
    db_file.close()

    database_test = []
    db_test_file = open('DB_'+city+'.csv', 'r')

    for line in db_test_file.readlines():
        elems = line.split(';')
        distance, start, end = float(elems[0]),int(elems[1]),int(elems[2])
        wps = [int(x) for x in elems[3:]]
        random.shuffle(wps) # in place shuffle of waypoints to reach
        x = generateTensors(start, end, wps, number_of_nodes, shortest_costs)
        
        truth_order = [int(x) for x in elems[3:]]
        waypoints = sorted(truth_order)
        edge_matrix = torch.zeros(len(waypoints), len(waypoints), dtype = torch.float32)
        for ind1, node1 in enumerate(waypoints):
            for ind2, node2 in enumerate(waypoints):
                if truth_order.index(node1) == truth_order.index(node2) + 1:
                    edge_matrix[ind1][ind2] = 1
                    edge_matrix[ind2][ind1] = 1
        edge_matrix = torch.unsqueeze(edge_matrix, 0)
        
        database_test.append((x, waypoints, edge_matrix, distance, start, end))
        
    db_test_file.close()

    print('data base generated', len(database), len(database_test))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)    
    
    model = GCN(5).to(device)
    
    print(model)
    
    if mode == 'init':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # faster init from epoch 0 to 35
        epochs = range(0,36)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # tunning from epoch 36 to 100
        epochs = range(36,100)
        model.load_state_dict(torch.load('w_'+city+'_35.bin'))
        optimizer.load_state_dict(torch.load('Opt_'+city+'_35.bin'))
       
    myGraph = Data(edge_index=edge_index, edge_attr = edge_attr).to(device)
        
    for epoch in epochs:
        losses = []
        model.train()
        
        indexes = list(range(len(database)))
        random.shuffle(indexes)
        
        for index in indexes:
        
            x, waypoints, truth_matrix, truth_distance, start, end = database[index]
            myGraph.x = x.to(device)

            optimizer.zero_grad()
            out = model(myGraph, torch.tensor(waypoints).to(device))
            
            loss_val = F.binary_cross_entropy(out, truth_matrix.to(device))
            
            losses.append(loss_val.item())
            loss_val.backward()
            optimizer.step()    
        
        
        losses_test = []
        model.eval()
        
        for x, waypoints, truth_matrix, truth_distance, start, end in database_test:
        
            myGraph.x = x.to(device)

            out = model(myGraph, torch.tensor(waypoints).to(device))
            
            loss_val = F.binary_cross_entropy(out, truth_matrix.to(device))
            
            losses_test.append(loss_val.item())
        
        print(epoch, 'TRAIN', np.mean(losses), 'TEST', np.mean(losses_test))        
        
        if epoch % 5 == 0:
            torch.save(model.state_dict(), 'w_'+city+'_'+str(epoch)+'.bin')
            torch.save(optimizer.state_dict(), 'Opt_'+city+'_'+str(epoch)+'.bin')
        
main()
