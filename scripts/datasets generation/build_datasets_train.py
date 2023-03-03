import json
import random
from ortools.sat.python import cp_model
import os


# Chargement graph data
def loadData(city):
    path = 'graph_'+city+'.json'

    file = open(path, 'rb')

    data = json.load(file)
  
    file.close()

    adj_matrix = data['adj']
    number_of_nodes = int(data['number_of_nodes'])
    shortest_costs = data['shortest_costs']
    
    print('Data loaded', path, 'with', number_of_nodes, 'nodes')
    
    return adj_matrix, number_of_nodes, shortest_costs


def generateProblem(size, number_of_nodes):
    start   = random.randint(0,number_of_nodes-1)
    # we need an end position different from start position
    while (True):
        end     = random.randint(0,number_of_nodes-1)
        if end != start:
            break
    
    wp_number = random.randint(10,30)
    
    possible_mandatories = [x for x in range(number_of_nodes) if x != start and x != end] # incremetal list of nodes except start and end
    current_choices = list(possible_mandatories)

    mandatory_list = []
    for _ in range(wp_number):
        selected_node = random.choice(current_choices)
        mandatory_list.append(selected_node)
        current_choices.remove(selected_node)   
    return start, end, mandatory_list
    
    
def remap(waypoints):
    # Remap node identifiers to continous index
    node_to_int = dict()
    int_to_node = []
    
    for ind, waypoint in enumerate(waypoints):
        node_to_int[waypoint] = ind
        int_to_node.append(waypoint)   
        
    return node_to_int, int_to_node
    
def find_optimal(start, end, wp, node_to_int, int_to_node, shortest_costs, forced = None):
    model = cp_model.CpModel()
    
    size = len(wp)
    decisions = []
    objective_items = []
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            lit = model.NewBoolVar('from ' + str(i)+' to '+str(j))
            decisions.append([i, j, lit])
            if int_to_node[i] == end and int_to_node[j] == start:
                model.Add(lit == True)
            else:
                objective_items.append(shortest_costs[int_to_node[i]][int_to_node[j]] * lit)
                if forced != None:
                    if i == node_to_int[forced[0]] and j == node_to_int[forced[1]]:
                        model.Add(lit == True)
    
    model.AddCircuit(decisions)
    
    model.Minimize(sum(objective_items))        
    
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status == cp_model.OPTIMAL:
        print('found')
        found_dist = solver.ObjectiveValue()
        solve_time = solver.UserTime()
        seq = []
        
        cur_node = node_to_int[start]
        while(True):
            seq.append(int_to_node[cur_node])
            if int_to_node[cur_node] == end:
                break
            for i,j, lit in decisions:
                if i == cur_node and solver.Value(lit):
                    next_node = j
                    break
            cur_node = next_node
        return found_dist, solve_time, seq
    else:
        print(forced, solver.StatusName())

    
def analyse(start, end, wp, shortest_costs, city):    
    print('from ', start, 'to', end)
    random.shuffle(wp)

    node_to_int, int_to_node = remap(wp)
    
    lenght, solve_time, seq = find_optimal(start, end, wp, node_to_int, int_to_node, shortest_costs)
    print('best lenght', lenght, 'solve time', solve_time ,  'solution', seq)
    
    strseq = [str(i) for i in seq]
    db_file = open('DB_train_'+city+'_len_10to30.csv', 'a')
    print(lenght,';',solve_time,';',start,';', end, ';',  ';'.join(strseq), file = db_file)
    db_file.close()
    
def main():
    print('start')
    
    completed = []
    
    for root, dirs, files in os.walk('.', topdown=True):
        for name in files:
            if '.json' not in name:
                continue
            city = name.split('_')[1].split('.')[0]
            
            if city in completed:
                continue
    
            adj_matrix, number_of_nodes, shortest_costs = loadData(city)
    
            for _ in range(10000):
                start, end, wp = generateProblem(0, number_of_nodes)
                analyse(start, end, wp+[start, end], shortest_costs, city)

main()

