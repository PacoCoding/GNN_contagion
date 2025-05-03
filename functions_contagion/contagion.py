import torch
import numpy as np
import random
import os
import sys
from utils import load_data_for_quarter

######################### 
# THESE FUNCTIONS AND IN PARTICULAR THE BIG FUNCTION get_contagion_list PRODUCE THE CONTAGION CHAIN TENSOR THAT WILL
# BE THEN USED AS ADDED FEATURES FOR PREDICTING THE CREDIT WORTHINESS USING GCN + TRANSFORMER
#########################


####################
# This is the actual full function that uses get_contagion_list_for_quarter to compute stuffs
###################
def get_contagion_list(year_from, quarter_from, year_to, quarter_to):
    time_steps = 0
    base_dir = '../../../datasets/contagion_data'
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    if (year_from > year_to or (year_from == year_to and quarter_from > quarter_to)):
        print("the input year and quarter error!")
        return
    while (True):
        time_steps += 1
        contagion_list = get_contagion_list_for_quarter(
            year_from, quarter_from)
        file_path = base_dir + "/contagion_" + \
            str(year_from) + "_" + str(quarter_from) + ".pth"
        torch.save(contagion_list, file_path)
        ###
        print('year {} quarter {} done'.format(year_from,quarter_from))
        ###
        if (year_from == year_to and quarter_from == quarter_to):
            break
        if (quarter_from == 4):
            quarter_from = 1
            year_from += 1
        else:
            quarter_from += 1
        

def get_contagion_list_for_quarter(year, quarter):
    max_path_length = 8
    max_pathlist_length = 8
    edge_attr, single_edge_index = load_data_edgeWeight_quarter(year, quarter)
    sort_edges = sort_edges_by_weight(single_edge_index, edge_attr)
    adjacency = adjacency_list(sort_edges, max_path_length)
    all_paths = get_all_paths_from_edge_index(adjacency, single_edge_index.max(
    ).item() + 1, max_path_length, max_pathlist_length)
    all_paths = torch.IntTensor(all_paths)
    feature = load_data_for_quarter(year, quarter)[1]
    # Append a new row fill with 0 
    new_row = torch.full((1, 70), fill_value=0, dtype=torch.float)
    feature_extended = torch.cat((feature, new_row), dim=0)
    result = all_paths.clone()
    result = result.to(torch.float32)
    result = torch.unsqueeze(result, dim=3)
    result = result.expand(-1, -1, -1, 70)
    all_paths = all_paths.long()
    result = feature_extended[all_paths]

    return result

def load_data_edgeWeight_quarter(year, quarter, data_dir='graph_data'):
    # Define the file path from which you want to propagate your data#
    
    # Edge = "../../../datasets/{}/edge_Q/edge_".format(data_dir) + \
        # str(year) + "Q" + str(quarter) + ".csv"

    Edge = "../../../datasets/edges/edge_" + \
    str(year) + "Q" + str(quarter) + ".csv"
    ##################
    # We read the file path line by line, the edge list is a N X 3 matrix. Where presents as columns the source nodes,
    # the target node and the weight associated to that edge.
    # It skips the first row ( header)
    ##################
    edge_attr = []
    single_edge_index = []
    with open(Edge, "r") as f:
        ################
        # Readlines gives as output this: ["0,1,4440.0\n", "0,2,3809.0\n", "0,3,16011.0\n", ...]
        ################
        edges = f.readlines()
        j = 0
        for edge in edges:
            if j == 0:
                j = 1
                continue
            #############
            # For read every line we eliminate ('\n') and separate values when find commas
            # Then we add the corresponding values to start, end and weight and append does
            # values to a list
            ############
            edge = edge.strip('\n')
            start, end, weight = edge.split(',')
            weight = float(weight)
            start = int(start)
            end = int(end)
            single_edge_index.append([start, end])
            edge_attr.append(weight)
    ########################
    # Now we transform the list we obtained in to tensors, in order to PyTorch to process them
    # In particular pytorch looks for edge shape to be: [num_edges, num_features], in our case: [num_edges, 1(weight)]
    # Single edge_index has shape : [2, num_edges]
    ########################
    edge_attr = torch.FloatTensor(edge_attr)
    single_edge_index = torch.IntTensor(single_edge_index)
    edge_attr = torch.unsqueeze(edge_attr, dim=1)
    return edge_attr, single_edge_index


##################
# This function given the edges_attr (weight) and single_edge_index (start, end)
# It first find the num_nodes that will be used to fond how many small lists [] we need (one per node)
#
##################
def sort_edges_by_weight(edge_index, edge_weights):
    num_nodes = edge_index.max().item() + 1
    edge_groups = [[] for _ in range(num_nodes)]
    # We use this syntax because edge_index is a 2D tensor so you pick i (index) and the start, end at every index ex:[4,9]
    # We create edge_group that is: edge_groups[u] == [(v1, w1), (v2, w2), …], “from node u, there’s an edge to v1 of weight w1, an edge to v2 of weight w2, etc.”  
    for i, (start, end) in enumerate(edge_index):
        weight = edge_weights[i]
        edge_groups[start].append((end, weight))
    for group in edge_groups:
        random.shuffle(group)
    return edge_groups
############
# function that orchestrate find_path and find_paths to iterate over all nodes doing DFS
###########
def get_all_paths_from_edge_index(truncated_list, num_nodes, max_path_length, max_pathlist_length):
    #all_paths will collect each node’s list of paths.
    all_paths = []
    #visited is shared across all searches, but is reset per branch by the DFS helpers.
    visited = [False] * num_nodes
    # For each bank node u
    for node in range(num_nodes):
        current_path = []
        one_node_paths = []
        find_paths(truncated_list, node, visited, current_path,
                   one_node_paths, max_path_length)
        for sublist in one_node_paths:
            #Some paths will end early (dead ends) at length < max_path_length so we increase their lenght with a dummy id so all habe the same lenght
            if len(sublist) < max_path_length:
                sublist.extend([4548] * (max_path_length - len(sublist)))
        if len(one_node_paths) < max_pathlist_length:
            missing_paths = max_pathlist_length - len(one_node_paths)
            one_node_paths.extend([[4548] * max_path_length] * missing_paths)
        all_paths.append(one_node_paths[0:max_pathlist_length])
    return all_paths
##########
# Similar to adjacency_list but we also keep the information of weights
###########
def adjacency_list_with_weights(sorted_edge_groups, max_path_length):
    # Instead of dropping weights, keep them:
    adj_list = []
    for edges in sorted_edge_groups:
        # edges is [(neighbor, weight), …]
        adj_list.append(edges[:max_path_length])
    return adj_list
###################
# We randomly sample some nodes that of the form: [[ (v₁, w₁), (v₂, w₂), … ], # node 0 [ (u₁, x₁), … ], # node 1 … ]
# We take for each one of them only the neighbours and extract only the ones that are present inside the max_path_lengh 
# computed by DFS
#################
def adjacency_list(sorted_edge_groups, max_path_length):
    adj_list = []
    for edges in sorted_edge_groups:
        adj_list.append([(int)(end) for end, _ in edges])
    truncated_list = [sublist[:max_path_length] if len(
        sublist) > max_path_length else sublist for sublist in adj_list]
    return truncated_list
########
# Need another function to compute contagion with weights
#########
#def ....

#################
# This takes the truncated_list of the nodes, given a node, 
#################
def find_path(graph, node, visited, current_path, one_node_paths, max_path_length):
    # We mark the current node so it will never pass through it again and we append it to current_path
    visited[node] = True
    current_path.append(node)
    current_length = len(current_path)
    #--------  Base case: path complete or dead‐end ----------
    # If we arrive at the limit of hops or we find a leaf we take a copy of all the list, we then return false to the node
    # such that the last node will be still looked in future iterations and use as starting point, we return True to say that
    # we visited the full path for this branch
    if current_length == max_path_length or len(graph[node]) == 0:
        one_node_paths.append(current_path.copy())
        visited[node] = False
        current_path.pop()
        return True
    #--------- Recursive case: explore each neighbor ---------
    flag = False
    for neighbor in graph[node]:
        if not visited[neighbor]:
            # Dive one step deeper
            flag = find_path(graph, neighbor, visited,
                             current_path, one_node_paths, max_path_length)
            if flag:
                # Once a full path is recorded down this branch,
                # stop exploring the other neighbors of this node
                break
    # ------------Final backtracking after exploring neighbors --------------
    visited[node] = False
    current_path.pop()
    return flag
##############
# for each v that is a neighbour of u, invoke find_path(...,v,...) to explore one entire branch starting at v
############
def find_paths(graph, node, visited, current_path, one_node_paths, max_path_length):
    current_path.append(node)
    for neighbor in graph[node]:
        find_path(graph, neighbor, visited, current_path,
                  one_node_paths, max_path_length)
    if len(graph[node]) == 0:
        one_node_paths.append(current_path)

if __name__ == '__main__':
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # _path = os.path.abspath(os.path.join(current_dir, '../../..'))
    # sys.path.append(_path)
    get_contagion_list(2016, 1, 2023, 1)
