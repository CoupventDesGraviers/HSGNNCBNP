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
from minizinc import Instance, Model, Solver, Status
import datetime

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
        vect = self.sigm(vect).clone()

        for ind in range(nb_waypoints):
            vect[ind][ind] = torch.tensor([0.0]).to(torch.device('cuda'))
            # vect[ind][ind] = torch.tensor([0.0])
        
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
        optimal = float(elems[0])
        if '.' in elems[1]:
            start, end = int(elems[2]),int(elems[3])
            sequence = [int(x) for x in elems[4:]]
        else:
            start, end = int(elems[1]),int(elems[2])
            sequence = [int(x) for x in elems[3:]]
        problem_list.append((start, end, sequence, optimal))
        
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
        
    for start, end, truth_order, optimal in problems:
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
        
        database.append((cur_graph, wps, edge_matrix, start, end, optimal))
        
    return database
    
def solve_with_minizinc(start, end, allpoints, shortest_costs, edge_prediction, optimal):

    # Prepare json datas
    size = len(allpoints)
    
    matrix = np.zeros((size,size), dtype = float)
    edges = np.zeros((size,size), dtype = int)
    ind_wp = []
    ind_to_wp = dict()
    wp_to_ind = dict()
    for ind1, wp1 in enumerate(allpoints):
        if wp1 == start:
            ind_start = ind1+1
            ind_to_wp[ind_start] = start
            wp_to_ind[start] = ind_start
        elif wp1 == end:
            ind_end = ind1+1
            ind_to_wp[ind_end] = end
            wp_to_ind[end] = ind_end
        else:
            ind_wp.append(ind1+1)
            ind_to_wp[ind1+1] = wp1
            wp_to_ind[wp1] = ind1+1
        
        for ind2, wp2 in enumerate(allpoints):
            matrix[ind1][ind2] = shortest_costs[wp1][wp2]


    for ind1, wp1 in enumerate(allpoints):
        for ind2, wp2 in enumerate(allpoints):
            max_edge_pred = max(edge_prediction[ind1][ind2], edge_prediction[ind2][ind1]) # make the edge matrix symetric
            edges[ind1][ind2] = 100 - int(100 * max_edge_pred)

    data = {'matrix': matrix.tolist(), 'number_of_nodes': size - 2, 'start':ind_start, 'end':ind_end, 'edge_prediction': edges.tolist()}
        
    model_minizinc = Model('find_min_th.mzn')
    gecode = Solver.lookup("gecode")

    # Create an Instance of the model for Gecode
    instance = Instance(gecode, model_minizinc)

    # Assign value to the model parameters
    for key in data:
        instance[key] = data[key] 
        
    result = instance.solve()
    minimum_TH = result['min_treshold']
    solve_th_time = result.statistics['solveTime'].total_seconds()
    print(minimum_TH, solve_th_time)
    
    get_th_file = open('finding_th.txt', 'a')
    print('foundTH:', minimum_TH, ':durarion:', solve_th_time, file = get_th_file)     
    get_th_file.close()
    
    data['minimum_TH'] = minimum_TH
        
    # For several solving timout
    for duration in range(60,241,60):

        model_minizinc = Model('solver_unguided.mzn')
        gecode = Solver.lookup("gecode")

        # Create an Instance of the model for Gecode
        instance = Instance(gecode, model_minizinc)

        # Assign value to the model parameters
        for key in data:
            instance[key] = data[key] 
            
        result = instance.solve(timeout=datetime.timedelta(seconds=duration))
        if(result.status == Status.UNKNOWN):
            unguided_path_len = float('inf')
        else:
            unguided_path_len = result['sum_path']
        
        # Load the minizinc model and the solver
        model_minizinc_edges = Model('solver_guided.mzn')

        # Create an Instance of the model for Gecode
        instance_edges = Instance(gecode, model_minizinc_edges)

        # Assign value to the model parameters
        for key in data:
            instance_edges[key] = data[key] 
            
        result_edges = instance_edges.solve(timeout=datetime.timedelta(seconds=duration))
        if(result_edges.status == Status.UNKNOWN):
            guided_path_len = float('inf')
        else:
            guided_path_len = result_edges['sum_path']
        print('duration:', duration, ':unguided:', unguided_path_len, ':guided:',   guided_path_len, ':optimal:', optimal) 
        res_file = open('inference_results_Seattle_62.txt', 'a')
        print('duration:', duration, ':unguided:', unguided_path_len, ':guided:',   guided_path_len, ':optimal:', optimal, file = res_file)     
        res_file.close() 

def main():
    print('start')
    city = 'seattle'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    adj_matrix, number_of_nodes, shortest_costs = loadData(os.path.join('graph_'+city+'.json')) 
    print('city', city, 'graph order', number_of_nodes)
    
    edge_index, edge_attr = gen_edges_in_G(number_of_nodes, adj_matrix)
    database_validation = gen_database_in_G(os.path.join('DB_'+city+'_len_62.csv'), number_of_nodes, shortest_costs, edge_index, edge_attr)

    print('data base loaded', len(database_validation))
    
    model = GCN_in_G(5).to(device)
    print(model)
    
    model.load_state_dict(torch.load(os.path.join('w_'+city+'.bin'),  map_location=torch.device('cpu')))

        
    model.eval()
        
    for myGraph, wps, truth_matrix, start, end, optimal in database_validation:
        print(start, end, wps)
    
        myGraphGpu = myGraph.to(device)
        
        out = model(myGraphGpu, torch.tensor(wps).to(device))

        edge_prediction = out[0].cpu().detach().numpy()
        
        solve_with_minizinc(start, end, wps, shortest_costs, edge_prediction, optimal)

main()