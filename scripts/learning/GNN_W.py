# -*- coding: utf-8 -*-

'''
# Install required packages.
!pip install -q torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
!pip install -q torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
!pip install -q git+https://github.com/pyg-team/pytorch_geometric.git
'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.data import Data
import random
import numpy as np
import copy
from torch.utils.tensorboard import SummaryWriter
import json

# Chargement graph data
def loadData(path):

    file = open(path, 'rb')

    data = json.load(file)
  
    file.close()

    adj_matrix = data['adj']
    number_of_nodes = int(data['number_of_nodes'])
    shortest_costs = data['shortest_costs']
    
    print('Data loaded')
    
    return adj_matrix, number_of_nodes, shortest_costs



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class GCN_in_G(torch.nn.Module):
    def __init__(self, features, last_layer = 10):
        super().__init__()
    
        out_sizes = range(10, 2000, 15)
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
        # self.edge_fc2 = nn.Linear(hidden_size, 1)

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
        vect = self.pooling(vect)
        # vect = self.sigm(vect).clone()

        for ind in range(nb_waypoints):
            vect[ind][ind] = torch.tensor([-10000.0]).to(torch.device('cuda'))
        
        vect = vect.transpose(0,2)        
        return vect



def read_problems(path, max_len):
    '''
    Read information from database file.
    File structure dependencies should be kepts in this function.
    '''
    problem_list = []
    
    db_file = open(path, 'r')

    line_counter = 0
    for line in db_file.readlines():
        elems = line.split(';')
        if '.' in elems[1]:
            start, end = int(elems[2]),int(elems[3])
            sequence = [int(x) for x in elems[4:]]
        else:
            start, end = int(elems[1]),int(elems[2])
            sequence = [int(x) for x in elems[3:]]
        problem_list.append((start, end, sequence))
        
        if max_len != None and line_counter >= max_len-1:
            break
        line_counter += 1
    
    db_file.close()
    
    return problem_list
    

def gen_edges_in_G(number_of_nodes, adj_matrix):
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

    return edge_index, edge_attr


def gen_database_in_G(path, number_of_nodes, shortest_costs, edge_index, edge_attr, max_len = None):
    problems = read_problems(path, max_len)
    
    database = []
        
    for start, end, truth_order in problems:
        wps = copy.copy(truth_order)
        random.shuffle(wps) # in place shuffle, we do not want GNN to process already ordered waypoints


        node_list = []
        for ind in range(number_of_nodes):
            if ind == start:
                node_list.append((1.0, 0.0, 0.0, 1.0, shortest_costs[ind][end] ))
                continue
            if ind == end:
                node_list.append((0.0, 1.0, 0.0, shortest_costs[start][ind], 1.0))
                continue
            if ind in wps:           
                node_list.append((0.0, 0.0, 1.0, shortest_costs[start][ind], shortest_costs[ind][end]))
                continue
            node_list.append((0.0, 0.0, 0.0, shortest_costs[start][ind], shortest_costs[ind][end]))
        
        x = torch.tensor(node_list, dtype=torch.float)

        cur_graph = Data(x = x, edge_index=edge_index, edge_attr = edge_attr)
        
        # We need truth matrix to have same data order as graph processed by GNN
        edge_matrix = torch.zeros(len(wps), len(wps), dtype = torch.float32)
        for ind1, node1 in enumerate(wps):
            for ind2, node2 in enumerate(wps):
                if truth_order.index(node1) == truth_order.index(node2) + 1:
                    edge_matrix[ind1][ind2] = 1.0
                    edge_matrix[ind2][ind1] = 1.0
        
        edge_matrix = torch.unsqueeze(edge_matrix, 0)
        
        database.append((cur_graph, wps, edge_matrix))
        
    return database

def main():
    
    print('start')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    
    completed = []
    
    for root, dirs, files in os.walk(os.path.join('Cartes', 'processed'), topdown=True):
        for name in files:
            if '.json' not in name:
                continue
            city = name.split('_')[1].split('.')[0]
            
            if city in completed:
                continue
                    
            print(city)
            create_folders(city)
            learn_a_city(city, os.path.join(root, name))
            
            
def create_folders(city):
    if not os.path.exists('W'):
        os.makedirs('W')
    if not os.path.exists('runs'):
        os.makedirs('runs')
    if not os.path.exists(os.path.join('W', city)):
        os.makedirs(os.path.join('W', city))

        
def learn_a_city(city, graph_path):
    adj_matrix, number_of_nodes, shortest_costs = loadData(graph_path)            
    
    print('city', city, 'graph order', number_of_nodes)

    edge_index, edge_attr = gen_edges_in_G(number_of_nodes, adj_matrix)
        
    database        = gen_database_in_G(os.path.join('Datasets', 'Train','DB_train_'+city+'_len_10to30.csv'), number_of_nodes, shortest_costs, edge_index, edge_attr)
    database_test7  = gen_database_in_G(os.path.join('Datasets', 'TestVarSize','DB_'+city+'_len_7.csv'), number_of_nodes, shortest_costs, edge_index, edge_attr)
    database_test32 = gen_database_in_G(os.path.join('Datasets', 'TestVarSize','DB_'+city+'_len_32.csv'), number_of_nodes, shortest_costs, edge_index, edge_attr)
    database_test62 = gen_database_in_G(os.path.join('Datasets', 'TestVarSize','DB_'+city+'_len_62.csv'), number_of_nodes, shortest_costs, edge_index, edge_attr)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    writer = SummaryWriter(comment=city)

    model = GCN_in_G(5).to(device)
    print('parameter number : ', count_parameters(model))
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # faster init from epoch 0 to 35

    log_file = 'train_log.txt'

    res_file = open(log_file, 'a')
    print('city', city, file = res_file)
    res_file.close()


    for epoch in range(0,101):
        losses = []
        model.train()

        indexes = list(range(len(database)))
        random.shuffle(indexes)

        for index in indexes:

            myGraph, wps, truth_matrix = database[index]
            myGraphGpu = myGraph.to(device)

            optimizer.zero_grad()
            out = model(myGraphGpu, torch.tensor(wps).to(device))

            ratio = len(truth_matrix[0]) / 2 - 1

            loss_val = F.binary_cross_entropy_with_logits(out, truth_matrix.to(device), pos_weight = torch.tensor(ratio))

            losses.append(loss_val.item())
            loss_val.backward()
            optimizer.step()    


        model.eval()


        losses_test7 = []

        for myGraph, wps, truth_matrix in database_test7:

            myGraphGpu = myGraph.to(device)

            out = model(myGraphGpu, torch.tensor(wps).to(device))

            ratio = len(truth_matrix[0]) / 2 - 1

            loss_val = F.binary_cross_entropy_with_logits(out, truth_matrix.to(device), pos_weight = torch.tensor(ratio))

            losses_test7.append(loss_val.item())


        losses_test32 = []

        for myGraph, wps, truth_matrix in database_test32:

            myGraphGpu = myGraph.to(device)

            out = model(myGraphGpu, torch.tensor(wps).to(device))

            ratio = len(truth_matrix[0]) / 2 - 1

            loss_val = F.binary_cross_entropy_with_logits(out, truth_matrix.to(device), pos_weight = torch.tensor(ratio))

            losses_test32.append(loss_val.item())



        losses_test62 = []

        for myGraph, wps, truth_matrix in database_test62:

            myGraphGpu = myGraph.to(device)

            out = model(myGraphGpu, torch.tensor(wps).to(device))

            ratio = len(truth_matrix[0]) / 2 - 1

            loss_val = F.binary_cross_entropy_with_logits(out, truth_matrix.to(device), pos_weight = torch.tensor(ratio))

            losses_test62.append(loss_val.item())


        print(epoch, 'TRAIN', np.mean(losses), 'TEST7', np.mean(losses_test7), 'TEST32', np.mean(losses_test32), 'TEST62', np.mean(losses_test62))    
        res_file = open(log_file, 'a')        
        print(epoch, 'TRAIN', np.mean(losses), 'TEST7', np.mean(losses_test7), 'TEST32', np.mean(losses_test32), 'TEST62', np.mean(losses_test62), file = res_file) 
        res_file.close()

        writer.add_scalar('Loss/train', np.mean(losses), epoch)
        writer.add_scalar('Loss/valid5', np.mean(losses_test7), epoch)
        writer.add_scalar('Loss/valid30', np.mean(losses_test32), epoch)
        writer.add_scalar('Loss/valid60', np.mean(losses_test62), epoch)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join('W', city, 'w_'+str(epoch)+'.bin'))
            # torch.save(optimizer.state_dict(), os.path.join('W', city, 'Opt_'+str(epoch)+'.bin'))                
        
        
main()
