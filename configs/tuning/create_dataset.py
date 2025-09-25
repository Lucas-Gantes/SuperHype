import warnings
import contextlib
import argparse

import numpy as np
from itertools import combinations
import hypernetx as hnx
import matplotlib.pyplot as plt
import random
import networkx as nx
import os
import torch
from torch_geometric.utils import from_networkx
from collections import Counter
import pickle
from scipy.stats import chi2
import math
import graph_tool.all as gt

import time

nb_attempts = 1000


def reindex_hypergraph(hg: hnx.Hypergraph) -> hnx.Hypergraph:
    """
    Reindex the hypergraph to have integer labels starting from 0.
    """
    original_nodes = list(hg.nodes)
    mapping = {old: new for new, old in enumerate(original_nodes)}

    edges = list(hg.edges.incidence_dict.items())
    num_nodes = len(original_nodes)
    num_edges = len(edges)
    incidence = np.zeros((num_nodes, num_edges), dtype=int)

    for j, (edge_id, members) in enumerate(edges):
        for n in members:
            incidence[mapping[n], j] = 1

    return hnx.Hypergraph.from_incidence_matrix(incidence)


def generate_erdos_renyi_hypergraphs(num_hypergraphs, min_size, max_size, probs, k, seed=0, node_condition=False):
    """Generate random Erdos-Renyi hypergraphs."""
    rng = np.random.default_rng(seed)
    hypergraphs = []

    print("Generating Erdős–Rényi random hypergraphs: ", end='')
    while len(hypergraphs) < num_hypergraphs:
        num_nodes = rng.integers(min_size, max_size, endpoint=True)
        incidence_matrix = []

        # Select the hyperedges
        for edge_order in np.arange(2, k + 1):
            for combi in map(list, combinations(np.arange(num_nodes), edge_order)):
                if rng.random() < probs[edge_order - 2]:
                    vector_edge = np.zeros(num_nodes)
                    vector_edge[combi] = 1

                    incidence_matrix.append(vector_edge)

        if (len(incidence_matrix) > 0):
            incidence_matrix = np.array(incidence_matrix).T
            H = hnx.Hypergraph.from_incidence_matrix(incidence_matrix)
            if node_condition:
                if H.is_connected() and len(H.nodes) >= min_size and len(H.nodes) <= max_size:
                    hypergraphs.append(reindex_hypergraph(H))
            else:
                if H.is_connected():
                    hypergraphs.append(reindex_hypergraph(H))

        # Display the progressbar
        progression = (len(hypergraphs) / num_hypergraphs) * 100
        bar = '#' * (len(hypergraphs) * 50 // num_hypergraphs)
        print(f'\rGenerating Erdős–Rényi random hypergraphs: [{bar:<50}] {progression:.2f}%', end='', flush=True)

    print("\nDone!")

    return hypergraphs

def generate_ego_hypergraph(num_hypergraphs, min_size, max_size, num_edges, max_edge_size, seed=0, node_condition=False):
    """Generate random Ego hypergraphs."""
    rng = np.random.default_rng(seed)
    hypergraphs = []

    while len(hypergraphs) < num_hypergraphs:
        num_nodes = rng.integers(min_size, max_size, endpoint=True)
        # Generate a random graph
        nodes = range(num_nodes)
        edges = {}
        for i in range(num_edges):
            edge_size = random.randint(2, min(max_edge_size, num_nodes))
            edge_nodes = random.sample(nodes, edge_size)
            edges[i] = edge_nodes

        H = hnx.Hypergraph(edges)

        # Generate the ego hypergraph
        ego_node = random.choice(list(H.nodes))

        ego_edges = {}
        for edge in H.edges:
            if ego_node in H.edges[edge]:
                ego_edges[edge] = list(H.edges[edge])

        if node_condition:
            if H.is_connected() and len(H.nodes) >= min_size and len(H.nodes) <= max_size:
                hypergraphs.append(reindex_hypergraph(H))
        else:
            if H.is_connected():
                hypergraphs.append(reindex_hypergraph(H))

        # Display the progressbar
        progression = (len(hypergraphs) / num_hypergraphs) * 100
        bar = '#' * (len(hypergraphs) * 50 // num_hypergraphs)
        print(f'\rGenerating Ego random hypergraphs: [{bar:<50}] {progression:.2f}%', end='', flush=True)

    print("\nDone!")

    return hypergraphs

def generate_hypertrees(num_hypergraphs, min_size, max_size, p, k, seed=0, node_condition=False):
    """Generate hypertrees by merging connected edges."""
    rng = np.random.default_rng(seed)
    hypergraphs = []

    while len(hypergraphs) < num_hypergraphs:
        # Generate a random tree
        num_nodes = rng.integers(min_size, max_size, endpoint=True)
        T = nx.random_labeled_tree(n=num_nodes, seed=rng)

        # Initialize the hyperedges list
        hyperedges = []

        # Start with all edges as potential hyperedges
        potential_edges = list(T.edges())

        while potential_edges:
            # Start with a random edge
            current_edge = potential_edges.pop(rng.integers(len(potential_edges)))
            hyperedge = set(current_edge)


            # Grow the hyperedge
            while len(hyperedge) < k and potential_edges:
                # Find edges connected to the current hyperedge
                connected_edges = [e for e in potential_edges if set(e) & hyperedge]

                if not connected_edges:
                    break

                # Randomly choose a connected edge to add
                if rng.random() < p:
                    new_edge = connected_edges[rng.integers(len(connected_edges))]
                    hyperedge.update(new_edge)
                    potential_edges.remove(new_edge)
                else:
                    break

            hyperedges.append(hyperedge)

        # Add any remaining edges as hyperedges
        hyperedges.extend([set(e) for e in potential_edges])

        # Create a Hypergraph object
        H = hnx.Hypergraph(hyperedges)
        hypergraphs.append(reindex_hypergraph(H))

        # Display the progressbar
        progression = (len(hypergraphs) / num_hypergraphs) * 100
        bar = '#' * (len(hypergraphs) * 50 // num_hypergraphs)
        print(f'\rGenerating random hypertrees: [{bar:<50}] {progression:.2f}%', end='', flush=True)

    print("\nDone!")

    return hypergraphs

def generate_sbm_hypergraphs(num_hypergraphs, min_size, max_size, p, q, k, seed=0, node_condition=False):
    """Generate SBM hypergraphs."""
    rng = np.random.default_rng(seed)
    hypergraphs = []

    while len(hypergraphs) < num_hypergraphs:
        num_nodes = rng.integers(min_size, max_size, endpoint=True)

        communities = rng.choice([-1, 1], size=num_nodes)

        incidence_matrix = []

        # Select the hyperedges
        for combi in map(list, combinations(np.arange(num_nodes), k)):
            values = communities[combi]

            same_cluster = np.all(values == values[0])
            prob = rng.random()

            if (same_cluster and prob < p) or (not same_cluster and prob < q):
                vector_edge = np.zeros(num_nodes)
                vector_edge[combi] = 1

                incidence_matrix.append(vector_edge)

        if (len(incidence_matrix) > 0):
            incidence_matrix = np.array(incidence_matrix).T
            H = hnx.Hypergraph.from_incidence_matrix(incidence_matrix)

            if node_condition:
                if H.is_connected() and len(H.nodes) >= min_size and len(H.nodes) <= max_size:
                    hypergraphs.append(reindex_hypergraph(H))
            else:
                if H.is_connected():
                    hypergraphs.append(reindex_hypergraph(H))
        # Display the progressbar
        progression = (len(hypergraphs) / num_hypergraphs) * 100
        bar = '#' * (len(hypergraphs) * 50 // num_hypergraphs)
        print(f'\rGenerating SBM random hypergraphs: [{bar:<50}] {progression:.2f}%', end='', flush=True)

    print("\nDone!")

    return hypergraphs

def clique_expansion(H: hnx.Hypergraph):
    G = nx.Graph()
    
    for edge_nodes in H.incidence_dict.values():
        nodes = list(edge_nodes)
        k = len(nodes)
        if k < 2:
            continue
        
        for u, v in combinations(nodes, 2):
            if G.has_edge(u, v):
                G[u][v]['weight'] = 1
            else:
                G.add_edge(u, v, weight=1)
    
    nodes = list(G.nodes)
    nb_nodes = len(nodes)
    # Create a mapping from node to index
    node_indices = {node: i for i, node in enumerate(nodes)}

    # Create the weighted adjacency matrix
    adj_matrix = np.zeros((nb_nodes, nb_nodes))
    for u, v in G.edges:
        adj_matrix[node_indices[u], node_indices[v]] = G[u][v]['weight']
        adj_matrix[node_indices[v], node_indices[u]] = G[u][v]['weight']  # Symmetric matrix
    
    return adj_matrix

def weighted_clique_expansion(H: hnx.Hypergraph, weight_rule='uniform'):
    G = nx.Graph()
    
    for edge_nodes in H.incidence_dict.values():
        nodes = list(edge_nodes)
        k = len(nodes)
        if k < 2:
            continue
        
        if weight_rule == 'uniform':
            weight = 1.0
        elif weight_rule == 'inverse':
            weight = 1 / (k - 1)
        else:
            raise ValueError("weight_rule must be 'uniform' or 'inverse'")
        
        for u, v in combinations(nodes, 2):
            if G.has_edge(u, v):
                G[u][v]['weight'] += weight
            else:
                G.add_edge(u, v, weight=weight)
    
    nodes = list(G.nodes)
    nb_nodes = len(nodes)
    # Create a mapping from node to index
    node_indices = {node: i for i, node in enumerate(nodes)}

    # Create the weighted adjacency matrix
    adj_matrix = np.zeros((nb_nodes, nb_nodes))
    for u, v in G.edges:
        adj_matrix[node_indices[u], node_indices[v]] = G[u][v]['weight']
        adj_matrix[node_indices[v], node_indices[u]] = G[u][v]['weight']  # Symmetric matrix
    
    return adj_matrix


def get_clique_projection(hg):
    """
    Get the clique projection of a hypergraph.
    """
    list_edges = list(hg.edges.incidence_dict.values())
    # list_edges = [[int(i) for i in edge] for edge in list_edges]
    node_list = list(hg.nodes)
    node_list = [int(node) for node in node_list]
    adj_matrix = np.zeros((len(node_list), len(node_list)))
    for i in range(len(list_edges)):
        edge_nodes = list_edges[i]
        for j in range(len(edge_nodes)):
            for k in range(j + 1, len(edge_nodes)):
                adj_matrix[node_list.index(edge_nodes[j])][node_list.index(edge_nodes[k])] = 1
    return adj_matrix


# A correct way to test it
def can_add_clique(new_graph, max_cliques, new_hyperedge):
    t1 = time.time()

    new_max_cliques = [ set(cl) for cl in nx.find_cliques(nx.from_numpy_array(new_graph)) if len(cl) >= 2]

    t2 = time.time()

    allowed = set(map(frozenset, max_cliques)) | {frozenset(new_hyperedge)}

    new_max_cliques_set = set(map(frozenset, new_max_cliques))
    
    return new_max_cliques_set == allowed

def neighbor_subgraph(graph, node_set):
    """
    Create a subgraph of the input graph with all the neighbors of the nodes in node_set
    """
    subgraph_nodes = set()
    for node in node_set:
        subgraph_nodes.update(graph.neighbors(node))

    subgraph_nodes = subgraph_nodes.union(node_set)

    return graph.subgraph(subgraph_nodes).copy()


def add_clique(nb_nodes, graph_list, max_cliques_list, clique):
    nb_graphs = len(graph_list)
    id_adding_graph = -1

    for graph_id in random.sample(list(range(nb_graphs)), k=nb_graphs):
        new_graph = graph_list[graph_id].copy() 
        for i in clique:
            for j in clique:
                if i != j:
                    new_graph[i, j] = 1

        preserving_max_cliques = can_add_clique(new_graph, max_cliques_list[graph_id], clique)

        if preserving_max_cliques:
            graph_list[graph_id] = new_graph
            max_cliques_list[graph_id].append(clique)
            id_adding_graph = graph_id
            break

    if id_adding_graph == -1:
        graph_list.append(np.zeros((nb_nodes, nb_nodes)))
        for i in clique:
            for j in clique:
                if i != j:
                    graph_list[-1][i, j] = 1
        max_cliques_list.append([clique])

        return nb_graphs
    else:
        for i in clique:
            for j in clique:
                if i != j:
                    graph_list[id_adding_graph][i, j] = 1
        max_cliques_list[id_adding_graph].append(clique)
        return id_adding_graph


def add_clique_fixed_layers(graph_list, max_cliques_list, clique, num_layers):
    id_adding_graph = -1

    for graph_id in random.sample(list(range(num_layers)), k=num_layers):
        t1 = time.time()
        new_graph = graph_list[graph_id].copy()  
        t2 = time.time()
        for i in clique:
            for j in clique:
                if i != j:
                    new_graph[i, j] = 1
        t3 = time.time()

        preserving_max_cliques = can_add_clique(new_graph, max_cliques_list[graph_id], clique)


        if preserving_max_cliques:
            graph_list[graph_id] = new_graph
            max_cliques_list[graph_id].append(clique)
            id_adding_graph = graph_id
            break

    if id_adding_graph == -1:
        return False
    else:
        for i in clique:
            for j in clique:
                if i != j:
                    graph_list[id_adding_graph][i, j] = 1
        max_cliques_list[id_adding_graph].append(clique)
        return True


def get_multiple_projections(hg):
    nb_nodes = len(hg.nodes)
    t1 = time.time()
    projected_graph = nx.from_numpy_array(get_clique_projection(hg))

    t2 = time.time()

    max_cliques = list(nx.find_cliques(projected_graph))
    max_cliques = [set(clique) for clique in max_cliques]

    t3 = time.time()

    he_list = list(hg.edges.incidence_dict.values())
    he_list = [set(he) for he in he_list]

    t4 = time.time()

    generated_graphs = []
    max_cliques_graphs = []


    for he in he_list:
        nb_init_graphs = len(generated_graphs)
        result = add_clique(nb_nodes, generated_graphs, max_cliques_graphs, he)

    t5 = time.time()

    return generated_graphs

def get_multiple_projections(hg, num_layers):
    nb_nodes = len(hg.nodes)
    t1 = time.time()

    t2 = time.time()

    t3 = time.time()

    he_list = list(hg.edges.incidence_dict.values())
    he_list = [set(he) for he in he_list]

    t4 = time.time()

    generated_graphs = []
    for i in range(num_layers):
        generated_graphs.append(np.zeros((nb_nodes, nb_nodes)))
    max_cliques_graphs = [[] for i in range(num_layers)]

    for he in he_list:
        is_success = add_clique_fixed_layers(generated_graphs, max_cliques_graphs, he, num_layers)
        if not is_success:
            return None

    t5 = time.time()

    return generated_graphs


def make_hypergraph_compact_projections(hg_list, augmentation_factor=1):
    """
    Generate the labeled graphs corresponding to the multiple graph projection of the hypergraphs
    The representation must be as compact as possible
    """
    multi_adj_matrices = []
    max_num_graphs = 0
    for k in range(augmentation_factor):
        for i, hg in enumerate(hg_list):
            with contextlib.redirect_stderr(open(os.devnull, 'w')):
                proj_graphs = get_multiple_projections(hg)
            multi_adj_matrices.append(proj_graphs)
            if len(proj_graphs) > max_num_graphs:
                max_num_graphs = len(proj_graphs)
            
            
            progression = ((i+1)*(k+1) / (len(hg_list)*augmentation_factor)) * 100
            bar = '#' * ((i+1)*(k+1) * 50 // (len(hg_list)*augmentation_factor))
            print(f'\rGenerating multiple graph projection: [{bar:<50}] {progression:.2f}%', end='', flush=True)

    print("\nProjection of hypergraphs done.")
    print("Max number of graphs required for projection: "+str(max_num_graphs))

    # Create the labeled adjacency matrices
    labeled_adj_mat_list = []
    for mat_list in multi_adj_matrices:
        labeled_adj_mat = np.zeros((mat_list[0].shape[0], mat_list[0].shape[1], max_num_graphs))
        for i, mat in enumerate(mat_list):
            labeled_adj_mat[:, :, i] = mat

        labeled_adj_mat_list.append(labeled_adj_mat)
    print("Labeled adjacency matrices created.")
    return labeled_adj_mat_list


def make_hypergraph_sparse_projections(hg_list, num_layers, augmentation_factor=1):
    """
    Generate the labeled graphs corresponding to the multiple graph projection of the hypergraphs
    The representation will have num_layers layer in the end
    """
    multi_adj_matrices = []
    for k in range(augmentation_factor):
        for i, hg in enumerate(hg_list):
            print(f"Projecting graph n°{i} for the {k}th time")
            with contextlib.redirect_stderr(open(os.devnull, 'w')):
                for j in range(nb_attempts):
                    proj_graphs = get_multiple_projections(hg, num_layers)
                    if proj_graphs is not None:
                        break
                    else:
                        print(f"Failed to create a projection for hypergraph n°{i}, trying again...")
            if proj_graphs is None:
                raise RuntimeError(f"Failed to create a projection for hypergraph n°{i} after {nb_attempts} attempts")
            multi_adj_matrices.append(proj_graphs)
            
            
            progression = ((i+1)*(k+1) / (len(hg_list)*augmentation_factor)) * 100
            bar = '#' * ((i+1)*(k+1) * 50 // (len(hg_list)*augmentation_factor))
            print(f'\rGenerating sparse multiple graph projection: [{bar:<50}] {progression:.2f}%', end='', flush=True)

    print("\nProjection of hypergraphs done.")

    # Create the labeled adjacency matrices
    labeled_adj_mat_list = []
    for mat_list in multi_adj_matrices:
        labeled_adj_mat = np.zeros((mat_list[0].shape[0], mat_list[0].shape[1], num_layers))
        for i, mat in enumerate(mat_list):
            labeled_adj_mat[:, :, i] = mat

        labeled_adj_mat_list.append(labeled_adj_mat)
    print("Labeled adjacency matrices created.")
    return labeled_adj_mat_list


def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1):
    """Split the dataset into train, validation, and test sets."""
    num_graphs = len(dataset)
    indices = np.arange(num_graphs)
    np.random.shuffle(indices)

    train_size = int(num_graphs * train_ratio)
    val_size = int(num_graphs * val_ratio)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    return train_indices, val_indices, test_indices


def save_projected_dataset(labeled_adj_mat_train, labeled_adj_mat_val, labeled_adj_mat_test, folder_path):
    torch.save(labeled_adj_mat_train, folder_path + "train.pt")
    print(f"Training projected dataset saved to {folder_path + 'train.pt'}")

    torch.save(labeled_adj_mat_val, folder_path + "val.pt")
    print(f"Validation projected dataset saved to {folder_path + 'val.pt'}")

    torch.save(labeled_adj_mat_test, folder_path + "test.pt")
    print(f"Testing projected dataset saved to {folder_path + 'test.pt'}")


def save_hg_dataset(hg_list_train, hg_list_val, hg_list_test, folder_path):
    with open(folder_path + "hg_train.pkl", "wb") as f:
        pickle.dump(hg_list_train, f)
    print(f"Training hypergraph dataset saved to {folder_path + 'hg_train.pkl'}")
    with open(folder_path + "hg_val.pkl", "wb") as f:
        pickle.dump(hg_list_val, f)
    print(f"Validation hypergraph dataset saved to {folder_path + 'hg_val.pkl'}")
    with open(folder_path + "hg_test.pkl", "wb") as f:
        pickle.dump(hg_list_test, f)
    print(f"Testing hypergraph dataset saved to {folder_path + 'hg_test.pkl'}")



def generate_dataset(dataset_type, node_condition, num_train, num_val, num_test, num_layers,
                     augmentation_factor, seed_train=0, seed_val=1, seed_test=2):
    # Generate the hypergraphs
    print(f"Generating {num_train} training hypergraphs, {num_val} validation hypergraphs, and {num_test} test hypergraphs of type {dataset_type} with seed {[seed_train, seed_val, seed_test]}")
    
    with contextlib.redirect_stderr(open(os.devnull, 'w')):
        if dataset_type == 'erdos':
            hg_list_train = generate_erdos_renyi_hypergraphs(num_train, 32, 32, [0.1, 0.005, 0.0005], 4, seed=seed_train, node_condition=node_condition)
            hg_list_val = generate_erdos_renyi_hypergraphs(num_val, 32, 32, [0.1, 0.005, 0.0005], 4, seed=seed_val, node_condition=node_condition)
            hg_list_test = generate_erdos_renyi_hypergraphs(num_test, 32, 32, [0.1, 0.005, 0.0005], 4, seed=seed_test, node_condition=node_condition)
        elif dataset_type == 'sbm_custom':
            hg_list_train = generate_sbm_hypergraphs(num_train, 32, 32, 0.05, 0.001, 3, seed=seed_train, node_condition=node_condition)
            hg_list_val = generate_sbm_hypergraphs(num_val, 32, 32, 0.05, 0.001, 3, seed=seed_val, node_condition=node_condition)
            hg_list_test = generate_sbm_hypergraphs(num_test, 32, 32, 0.05, 0.001, 3, seed=seed_test, node_condition=node_condition)
        elif dataset_type == 'hypertrees':
            hg_list_train = generate_hypertrees(num_train, 32, 32, 0.1, 5, seed=seed_train, node_condition=node_condition)
            hg_list_val = generate_hypertrees(num_val, 32, 32, 0.1, 5, seed=seed_val, node_condition=node_condition)
            hg_list_test = generate_hypertrees(num_test, 32, 32, 0.1, 5, seed=seed_test, node_condition=node_condition)
        elif dataset_type == 'ego':
            hg_list_train = generate_ego_hypergraph(num_train, 150, 200, 3000, 5, seed=seed_train, node_condition=node_condition)
            hg_list_val = generate_ego_hypergraph(num_val, 150, 200, 3000, 5, seed=seed_val, node_condition=node_condition)
            hg_list_test = generate_ego_hypergraph(num_test, 150, 200, 3000, 5, seed=seed_test, node_condition=node_condition)
        else:
            raise NotImplementedError
        

    # multiple graph projections
    if num_layers is None:
        print("No number of layers specified, trying to have the most compact representation")
        labeled_adj_mat_list_train = make_hypergraph_compact_projections(hg_list_train, augmentation_factor)
        labeled_adj_mat_list_val = make_hypergraph_compact_projections(hg_list_val)
        labeled_adj_mat_list_test = make_hypergraph_compact_projections(hg_list_test)
    else:
        print(f"The required number of layers is: {num_layers}")
        labeled_adj_mat_list_train = make_hypergraph_sparse_projections(hg_list_train, num_layers, augmentation_factor)
        labeled_adj_mat_list_val = make_hypergraph_sparse_projections(hg_list_val, num_layers)
        labeled_adj_mat_list_test = make_hypergraph_sparse_projections(hg_list_test, num_layers)
        


    # Save the dataset
    node_condtion_str = "fixed_num_nodes" if node_condition else "flexible_num_nodes"
    save_path = dataset_type+str(num_train)+'x'+str(augmentation_factor)+'_'+str(num_val)+'_'+str(num_test)+'_l'+str(num_layers)+'_'+node_condtion_str+'/raw/'
    dataset_path = dataset_type+str(num_train)+'x'+str(augmentation_factor)+'_'+str(num_val)+'_'+str(num_test)+'_l'+str(num_layers)+'_'+node_condtion_str+'/'
    processed_path = dataset_type+str(num_train)+'x'+str(augmentation_factor)+'_'+str(num_val)+'_'+str(num_test)+'_l'+str(num_layers)+'_'+node_condtion_str+'/processed/'
    if not os.path.isdir(dataset_path):
        print(f"The folder {dataset_path} does not exist and will be created")
        os.makedirs(dataset_path, exist_ok=False)
    if not os.path.isdir(save_path):
        print(f"The folder {save_path} does not exist and will be created")
        os.makedirs(save_path, exist_ok=False)
    if not os.path.isdir(processed_path):
        print(f"The folder {processed_path} does not exist and will be created")
        os.makedirs(processed_path, exist_ok=False)

    
    save_hg_dataset(hg_list_train, hg_list_val, hg_list_test, save_path)
    save_projected_dataset(labeled_adj_mat_list_train, labeled_adj_mat_list_val, labeled_adj_mat_list_test, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hypergraph generation arguments parser")
    parser.add_argument('--type', type=str, required=True, help="The type of hypergraphs to generate (from 'erdos', 'sbm_custom', 'hypertrees' and 'ego')")
    parser.add_argument('--node_condition', action='store_true', help="Indicate whether the number of nodes should be fixed")
    parser.add_argument('--num_train', type=int, required=True, help="The number of hypergraphs in the training dataset")
    parser.add_argument('--num_val', type=int, required=True, help="The number of hypergraphs in the validation dataset")
    parser.add_argument('--num_test', type=int, required=True, help="The number of hypergraphs in the test dataset")
    parser.add_argument('--layers', type=int, required=False, help="The number of layers for each hypergraph")
    parser.add_argument('--augmentation_factor', type=int, required=False, help="The number of multiple graph projections for each hypergraph")
    parser.add_argument('--seed_train', type=int, required=False, help="The seed of the random generation process of the training dataset")
    parser.add_argument('--seed_val', type=int, required=False, help="The seed of the random generation process of the validation dataset")
    parser.add_argument('--seed_test', type=int, required=False, help="The seed of the random generation process of the test dataset")

    args = parser.parse_args()
    seed_train = 0 if args.seed_train is None else args.seed_train
    seed_val = 1 if args.seed_val is None else args.seed_val
    seed_test = 2 if args.seed_test is None else args.seed_test
    augmentation_factor = 1 if args.augmentation_factor is None else args.augmentation_factor
    print("node_condition: "+str(args.node_condition))
    generate_dataset(dataset_type=args.type,
                     node_condition=args.node_condition, 
                     num_train=args.num_train,
                     num_val=args.num_val,
                     num_test=args.num_test,
                     num_layers=args.layers,
                     augmentation_factor=augmentation_factor, 
                     seed_train=seed_train,
                     seed_val=seed_val,
                     seed_test=seed_test)



