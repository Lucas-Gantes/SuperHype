import contextlib
import os
import torch
import torch.nn as nn
import networkx as nx
import hypernetx as hnx
import numpy as np
from torch_geometric.utils import to_networkx
import wandb
from scipy.stats import wasserstein_distance, chi2
from scipy.linalg import eigvalsh, toeplitz
from scipy.sparse import diags, eye
import pyemd
from collections import Counter
import math
import graph_tool.all as gt
from tqdm.auto import tqdm
import time
from networkx.algorithms import isomorphism as iso

# Import stat functions from spectre_utils
from src.analysis.spectre_utils import (
    degree_stats, clustering_stats, spectral_stats,
    motif_stats, orbit_stats_all,
    eval_acc_sbm_graph, eval_acc_planar_graph,
    eval_fraction_unique_non_isomorphic_valid,
    eval_fraction_isomorphic
)

def node_degree_wasserstein_graph(hg_train_list, hg_generated_list):
    """
    Compute the Wasserstein distance betwenn node degree distribution of training hypergraph and generated hypergraph
    """
    reference_dist = Counter()
    pred_dist = Counter()
    for hg_train in hg_train_list:
        reference_dist += Counter(hnx.reports.descriptive_stats.degree_dist(hg_train))

    for hg_generated in hg_generated_list:
        pred_dist += Counter(hnx.reports.descriptive_stats.degree_dist(hg_generated))
    
    print("node degree distribution ref (in node_degree_wasserstein_graph): "+str(reference_dist))
    print("node degree distribution pred (in node_degree_wasserstein_graph): "+str(pred_dist))

    degree_dist_ref_keys = list(reference_dist.keys())
    degree_dist_ref_values = list(reference_dist.values())
    degree_dist_pred_keys = list(pred_dist.keys())
    degree_dist_pred_values = list(pred_dist.values())
        
    # Compute the Wasserstein distance
    return wasserstein_distance(
        np.array(degree_dist_ref_keys),
        np.array(degree_dist_pred_keys),
        np.array(degree_dist_ref_values),
        np.array(degree_dist_pred_values)
    )

def node_degree_wasserstein_label(hg_train_list, node_labels_array):
    """
    Compute the Wasserstein distance between node degree distribution of training hypergraph and generated hypergraph
    """
    reference_dist = Counter()
    for hg_train in hg_train_list:
        reference_dist += Counter(hnx.reports.descriptive_stats.degree_dist(hg_train))

    node_degrees_matrix = np.sum(node_labels_array, axis=(2,3))
    batch_size, n = node_degrees_matrix.shape
    pred_dist = Counter()
    for i in range (batch_size):
        for j in range(n):
            pred_dist += Counter({round(node_degrees_matrix[i][j]): 1})

    # print("node degree distribution ref (in node_degree_wasserstein_label): "+str(reference_dist))
    # print("node degree distribution pred (in node_degree_wasserstein_label): "+str(pred_dist))

    degree_dist_ref_keys = list(reference_dist.keys())
    degree_dist_ref_values = list(reference_dist.values())
    degree_dist_pred_keys = list(pred_dist.keys())
    degree_dist_pred_values = list(pred_dist.values())
        
    # Compute the Wasserstein distance
    return wasserstein_distance(
        np.array(degree_dist_ref_keys),
        np.array(degree_dist_pred_keys),
        np.array(degree_dist_ref_values),
        np.array(degree_dist_pred_values)
    )


def edge_size_wasserstein_graph(hg_train_list, hg_generated_list):
    """
    Compute the Wasserstein distance betwenn edge size distribution of training hypergraph and generated hypergraph
    """
    reference_dist = Counter()
    pred_dist = Counter()
    for hg_train in hg_train_list:
        reference_dist += Counter(hnx.reports.descriptive_stats.edge_size_dist(hg_train))
    
    for hg_generated in hg_generated_list:
        pred_dist += Counter(hnx.reports.descriptive_stats.edge_size_dist(hg_generated))

    print("edge size distribution ref (in edge_size_wasserstein_graph): "+str(reference_dist))
    print("edge size distribution pred (in edge_size_wasserstein_graph): "+str(pred_dist))

    size_dist_ref_keys = list(reference_dist.keys())
    size_dist_ref_values = list(reference_dist.values())
    size_dist_pred_keys = list(pred_dist.keys())
    size_dist_pred_values = list(pred_dist.values())
        
    # Compute the Wasserstein distance
    return wasserstein_distance(
        np.array(size_dist_ref_keys),
        np.array(size_dist_pred_keys),
        np.array(size_dist_ref_values),
        np.array(size_dist_pred_values)
    ), reference_dist, pred_dist


def edge_size_wasserstein_label(hg_train_list, node_labels_array):
    """
    Compute the Wasserstein distance betwenn edge size distribution of training hypergraph 
    and predicted hyperedges labels on each node for the generated graphs
    """
    reference_dist = Counter()
    for hg_train in hg_train_list:
        reference_dist += Counter(hnx.reports.descriptive_stats.edge_size_dist(hg_train))

    pred_sizes = np.sum(node_labels_array, axis=(0,1,2))
    sizes = np.arange(2, pred_sizes.shape[0]+2)

    pred_sizes = pred_sizes / sizes  # A hyperedge of size k is counted k times for each of its k nodes

    pred_dist = Counter({i+2: round(pred_sizes[i]) for i in range(pred_sizes.shape[0])})

    size_dist_ref_keys = list(reference_dist.keys())
    size_dist_ref_values = list(reference_dist.values())
    size_dist_pred_keys = list(pred_dist.keys())
    size_dist_pred_values = list(pred_dist.values())
    
    # print("edge size distribution ref (in edge_size_wasserstein_label): "+str(reference_dist))
    # print("edge size distribution pred (in edge_size_wasserstein_label): "+str(pred_dist))

    # Compute the Wasserstein distance
    return wasserstein_distance(
        np.array(size_dist_ref_keys),
        np.array(size_dist_pred_keys),
        np.array(size_dist_ref_values),
        np.array(size_dist_pred_values)
    )


def gaussian_emd(x, y, sigma=1.0, distance_scaling=1.0):
    ''' Gaussian kernel with squared distance in exponential term replaced by EMD
        Args:
            x, y: 1D pmf of two distributions with the same support
            sigma: standard deviation
    '''
    support_size = max(len(x), len(y))
    d_mat = toeplitz(range(support_size)).astype(np.float64)
    distance_mat = d_mat / distance_scaling

    # convert histogram values x and y to float, and make them equal len
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    emd = pyemd.emd(x, y, distance_mat)
    return np.exp(-emd * emd / (2 * sigma * sigma))

def gaussian_tv(x, y, sigma=1.0):  
    support_size = max(len(x), len(y))
    # convert histogram values x and y to float, and make them equal len
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    dist = np.abs(x - y).sum() / 2.0
    return np.exp(-dist * dist / (2 * sigma * sigma))

def disc(samples1, samples2, kernel, is_parallel=True, *args, **kwargs):
    ''' Discrepancy between 2 samples '''
    d = 0

    for s1 in samples1:
        for s2 in samples2:
            d += kernel(s1, s2, *args, **kwargs)

    if len(samples1) * len(samples2) > 0:
        d /= len(samples1) * len(samples2)
    else:
        d = 1e+6
    return d

def compute_mmd(samples1, samples2, kernel, is_hist=True, *args, **kwargs):
    ''' MMD between two samples '''
    # normalize histograms into pmf  
    if is_hist:
        samples1 = [s1 / (np.sum(s1) + 1e-6) for s1 in samples1]
        samples2 = [s2 / (np.sum(s2) + 1e-6) for s2 in samples2]
    return disc(samples1, samples1, kernel, *args, **kwargs) + \
                    disc(samples2, samples2, kernel, *args, **kwargs) - \
                    2 * disc(samples1, samples2, kernel, *args, **kwargs)

def normalized_laplacian_matrix(H):
    # Compute the incidence matrix
    incidence_matrix = H.incidence_matrix().toarray()
    
    # Compute the degree of nodes and hyperedges
    node_degree = np.sum(incidence_matrix, axis=1)
    hyperedge_degree = np.sum(incidence_matrix, axis=0)
    
    # Compute the diagonal matrices Dv and De
    Dv = np.diag(node_degree)
    De = np.diag(hyperedge_degree)
    
    # Compute the inverse square root of Dv
    Dv_inv_sqrt = np.linalg.inv(np.sqrt(Dv))
    
    # Compute the inverse of De
    De_inv = np.linalg.inv(De)
    
    # Compute the normalized Laplacian
    normalized_laplacian = np.eye(Dv.shape[0]) - Dv_inv_sqrt @ incidence_matrix @ De_inv @ incidence_matrix.T @ Dv_inv_sqrt

    return normalized_laplacian

def normalized_laplacian_matrix_sparse(H):
    incidence_matrix = H.incidence_matrix().tocsr()
    
    node_degree = np.ravel(incidence_matrix.sum(axis=1)) 
    hyperedge_degree = np.ravel(incidence_matrix.sum(axis=0)) 

    inv_sqrt_v = np.where(node_degree>0, 1.0/np.sqrt(node_degree), 0.0)

    inv_e = np.where(hyperedge_degree>0, 1.0/hyperedge_degree, 0.0)

    Dv_inv_sqrt = diags(inv_sqrt_v) 
    De_inv = diags(inv_e)

    n = incidence_matrix.shape[0]
    I_n = eye(n, format='csr')

    L = I_n - (Dv_inv_sqrt @ incidence_matrix) @ (De_inv @ incidence_matrix.T) @ Dv_inv_sqrt
    
    return L

def prune_isolated_nodes(H: hnx.Hypergraph) -> hnx.Hypergraph:
    deg = Counter()
    for he, nodes in H.incidence_dict.items():
        deg.update(nodes)
    isolated = [v for v in H.nodes if deg[v] == 0]
    if not isolated:
        return H
    else:
        raise ValueError("There is an isolated node in the hypergraph")
    H_pruned = H.copy()  
    H_pruned.remove_nodes_from(isolated)
    return H_pruned

def spectral_worker(H, n_eigvals=-1):
    H = prune_isolated_nodes(H)
    try:
        lp = normalized_laplacian_matrix(H)
        eigs = eigvalsh(lp)  
    except:
        eigs = np.zeros(len(H))
    if n_eigvals > 0:
        eigs = eigs[1:n_eigvals+1]
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    return spectral_pmf

def spectral_stats(hypergraph_ref_list, hypergraph_pred_list, is_parallel=True, n_eigvals=-1, compute_emd=False):
    sample_ref = []
    sample_pred = []

    hypergraph_pred_list_remove_empty = [
        H for H in hypergraph_pred_list if not len(H) == 0
    ]

    for i in range(len(hypergraph_ref_list)):
        spectral_temp = spectral_worker(hypergraph_ref_list[i], n_eigvals)
        sample_ref.append(spectral_temp)
    for i in range(len(hypergraph_pred_list_remove_empty)):
        spectral_temp = spectral_worker(hypergraph_pred_list_remove_empty[i], n_eigvals)
        sample_pred.append(spectral_temp)
    if compute_emd:
        # EMD option uses the same computation as hypergraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)

    return mmd_dist

def is_hypertree(H):
    if H.is_connected():
        # Step 1: Create the line graph of the hypergraph
        # line_graph = nx.Graph()
        line_graph = nx.DiGraph()  # Use directed graph to handle cycles correctly
        dic = {}

        # Add a node for each hyperedge
        for i, hyperedge in enumerate(H.edges):
            line_graph.add_node(i)
            dic[i] = hyperedge

        # Add edges between nodes (hyperedges) that share at least one vertex
        for i, edge1 in enumerate(H.edges):
            for j, edge2 in enumerate(H.edges):
                if i != j and set(H.edges[edge1]).intersection(H.edges[edge2]):
                # if i < j and set(H.edges[edge1]).intersection(H.edges[edge2]):
                    line_graph.add_edge(i, j)

        # Step 2: Check for cycles in the line graph        
        for cycle in nx.simple_cycles(line_graph):
            involved_edges = [set(H.edges[dic[line_node]]) for line_node in cycle]
            intersection = set.intersection(*involved_edges)
            if not intersection:
                return False

        return True
    

def is_sbm_hypergraph(H, p_intra=0.05, p_inter=0.001, k=3, strict=True, refinement_steps=1000):
    """
    Check how closely a given hypernetx hypergraph matches an SBM
    by computing the mean probability of the Wald test statistic for each recovered parameter,
    comparing to the real distribution in the hypergraph.
    """
    # Convert hypernetx hypergraph to graph-tool graph
    g = gt.Graph(directed=False)
    edge_list = []
    for e in H.edges:
        nodes = list(H.edges[e])
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                edge_list.append((nodes[i], nodes[j]))
    
    # Add unique vertices and edges
    vertices = {v: g.add_vertex() for v in H.nodes}
    for e in set(edge_list):  # Remove duplicates
        g.add_edge(vertices[e[0]], vertices[e[1]])

    try:
        state = gt.minimize_blockmodel_dl(g)
    except ValueError:
        return False if strict else 0.0

    # Refine using merge-split MCMC
    for _ in range(refinement_steps): 
        state.multiflip_mcmc_sweep(beta=np.inf, niter=10)
    
    b = gt.contiguous_map(state.get_blocks())
    state = state.copy(b=b)
    e = state.get_matrix()
    n_blocks = state.get_nonempty_B()
    
    # Calculate real probabilities from the hypergraph
    est_p_intra = []
    est_p_inter = []

    if strict and n_blocks != 2:
        return False

    est_p_inter = np.zeros((n_blocks, n_blocks))
    
    for i in range(n_blocks):
        block_nodes = [v for v, block in enumerate(b) if block == i]
        intra_edges = sum(1 for edge in H.edges if len(set(H.edges[edge]) & set(block_nodes)) == len(set(H.edges[edge])))
        possible_intra_edges = math.comb(len(block_nodes), k)
        if possible_intra_edges == 0:
            possible_intra_edges = 1  # Avoid division by zero
        est_p_intra.append(intra_edges / possible_intra_edges)
        
        for j in range(i+1, n_blocks):
            other_block_nodes = [v for v, block in enumerate(b) if block == j]
            inter_edges = sum(1 for edge in H.edges() if 
                              len(set(H.edges[edge]) & set(block_nodes)) >= 1 and 
                              len(set(H.edges[edge]) & set(other_block_nodes)) >= 1)
            possible_inter_edges = math.comb(len(block_nodes) + len(other_block_nodes), k) - math.comb(len(block_nodes), k) - math.comb(len(other_block_nodes), k)
            if possible_inter_edges == 0:
                possible_inter_edges = 1  # Avoid division by zero
            est_p_inter[i, j] = est_p_inter[j, i] =  inter_edges / possible_inter_edges
    
    est_p_intra = np.array(est_p_intra)

    W_p_intra = (est_p_intra - p_intra)**2 / (est_p_intra * (1-est_p_intra) + 1e-6)
    W_p_inter = (est_p_inter - p_inter)**2 / (est_p_inter * (1-est_p_inter) + 1e-6)
    
    W = W_p_inter.copy()
    np.fill_diagonal(W, W_p_intra)
    p = 1 - chi2.cdf(abs(W), 1)
    p = p.mean()
    if strict:
        return p > 0.9 # p value < 10 %
    else:
        return p

def is_ego_hypergraph(H):
    if len(H.nodes) == 0 or len(H.edges) == 0:
        return False

    # Find the node that appears in the most hyperedges
    node_frequencies = {node: 0 for node in H.nodes}
    for edge in H.edges:
        for node in H.edges[edge]:
            node_frequencies[node] += 1
    
    potential_ego = max(node_frequencies, key=node_frequencies.get)
    
    # Check if the potential ego is in all or most hyperedges
    edges_with_ego = sum(1 for edge in H.edges if potential_ego in H.edges[edge])
    
    if edges_with_ego < len(H.edges):
        return False

    # Check if all other nodes are directly connected to the ego
    nodes_connected_to_ego = set()
    for edge in H.edges:
        if potential_ego in H.edges[edge]:
            nodes_connected_to_ego.update(H.edges[edge])
    
    if nodes_connected_to_ego != set(H.nodes):
        return False

    # Check if there are any hyperedges not including the ego or its direct connections
    for edge in H.edges:
        if potential_ego not in H.edges[edge] and not any(node in nodes_connected_to_ego for node in H.edges[edge]):
            return False

    return True

    
def uniqueness(hg_dataset):
    bipartite_dataset = [hg_dataset[i].bipartite() for i in range(len(hg_dataset))]
    global_isomorphisms = 0
    for i in range(len(hg_dataset)):
        i_isomorphisms = 0
        for j in range(i+1, len(hg_dataset)):
            if len(hg_dataset[i].nodes)==len(hg_dataset[j].nodes) and len(hg_dataset[i].edges)==len(hg_dataset[j].edges):
                node_match = iso.categorical_node_match("bipartite", None)
                GM = iso.GraphMatcher(bipartite_dataset[i], bipartite_dataset[j], node_match=node_match)
                if GM.is_isomorphic():
                    i_isomorphisms += 1
        if i_isomorphisms > 0:
            global_isomorphisms += 1
    return 1. - (global_isomorphisms / len(hg_dataset))

def novelty(hg_dataset, hg_generated):
    bipartite_dataset = [hg_dataset[i].bipartite() for i in range(len(hg_dataset))]
    bipartite_generated = [hg_generated[i].bipartite() for i in range(len(hg_generated))]
    global_isomorphisms = 0
    for i in range(len(hg_generated)):
        i_isomorphisms = 0
        for j in range(len(hg_dataset)):
            if len(hg_generated[i].nodes)==len(hg_dataset[j].nodes) and len(hg_generated[i].edges)==len(hg_dataset[j].edges):
                node_match = iso.categorical_node_match("bipartite", None)
                GM = iso.GraphMatcher(bipartite_generated[i], bipartite_dataset[j], node_match=node_match)
                if GM.is_isomorphic():
                    i_isomorphisms += 1
        if i_isomorphisms > 0:
            global_isomorphisms += 1
    
    return 1. - (global_isomorphisms/len(hg_generated))

def node_dist(hg_dataset, hg_generated):
    ref_dist = Counter()
    pred_dist = Counter()

    for hg_ref in hg_dataset:
        ref_dist += Counter([len(list(hg_ref.nodes))])
    
    for hg_pred in hg_generated:
        pred_dist += Counter([len(list(hg_pred.nodes))])

    print("Node distribution from the reference dataset: "+str(ref_dist))
    print("Node distribution from the predicted batch: "+str(pred_dist))

    nb_dist_ref_keys = list(ref_dist.keys())
    nb_dist_ref_values = list(ref_dist.values())
    nb_dist_pred_keys = list(pred_dist.keys())
    nb_dist_pred_values = list(pred_dist.values())
        
    # Compute the Wasserstein distance
    return wasserstein_distance(
        np.array(nb_dist_ref_keys),
        np.array(nb_dist_pred_keys),
        np.array(nb_dist_ref_values),
        np.array(nb_dist_pred_values)
    )

def compute_centrality_distribution(hypergraphs, type):
    all_centralities = []
    for H in hypergraphs:
        if type == 'closeness':
            centralities = hnx.algorithms.s_centrality_measures.s_closeness_centrality(H)
        elif type == 'betweenness':
            centralities = hnx.algorithms.s_centrality_measures.s_betweenness_centrality(H)
        elif type == 'harmonic':
            centralities = hnx.algorithms.s_centrality_measures.s_harmonic_centrality(H)
        else:
            raise NotImplementedError(type)
        all_centralities.extend(centralities.values())
    return np.array(all_centralities)

def centrality_closeness(hg_ref, hg_pred):
    centrality_pred = compute_centrality_distribution(hg_pred, 'closeness')
    centrality_ref = compute_centrality_distribution(hg_ref, 'closeness')
    return wasserstein_distance(centrality_pred, centrality_ref)

def centrality_betweenness(hg_ref, hg_pred):
    centrality_pred = compute_centrality_distribution(hg_pred, 'betweenness')
    centrality_ref = compute_centrality_distribution(hg_ref, 'betweenness')
    return wasserstein_distance(centrality_pred, centrality_ref)

def centrality_harmonic(hg_ref, hg_pred):
    centrality_pred = compute_centrality_distribution(hg_pred, 'harmonic')
    centrality_ref = compute_centrality_distribution(hg_ref, 'harmonic')
    return wasserstein_distance(centrality_pred, centrality_ref)

def get_weighted_graph(hg):
    weighted_graph = nx.Graph()
    weighted_graph.add_nodes_from(list(hg.nodes))
    he_list = [list(hg.edges[e].elements) for e in hg.edges]
    for he in he_list:
        size = len(he)
        if size >= 2:
            for node1 in he:
                for node2 in he:
                    if weighted_graph.has_edge(node1, node2):
                        current_weight = weighted_graph['A']['B']['weight']
                        if 1/(size-1) > current_weight:
                            weighted_graph['A']['B']['weight'] = 1/(size-1)
                    else:
                        weighted_graph.add_edge(node1, node2, weight=1/(size-1))
    return weighted_graph

def weighted_graph_clustering_coef(hg):
    weighted_graph = get_weighted_graph(hg)
    clustering_coef_list = []
    for v in list(weighted_graph.nodes):
        num = 0.
        denom = 0.
        for n1 in list(weighted_graph.neighbors(v)):
            for n2 in list(weighted_graph.neighbors(v)):
                if n1 != n2:
                    num += weighted_graph[v][n1]['weight'] * weighted_graph[n1][n2]['weight'] * \
                            weighted_graph[n2][v]['weight']
                    denom += weighted_graph[n1][v]['weight'] * weighted_graph[v][n2]['weight']
        clustering_coef_list.append(num / denom)

def weighted_graph_clustering_coef_stats(hg_ref, hg_pred):
    ref_dist = Counter()
    pred_dist = Counter()

    for hg in hg_ref:
        ref_dist += Counter(weighted_graph_clustering_coef(hg))
    
    for hg in hg_pred:
        pred_dist += Counter(weighted_graph_clustering_coef(hg))

    print("Weighted graph clustering coefficient distribution from the reference dataset: "+str(ref_dist))
    print("Weighted graph clustering coefficient distribution from the predicted batch: "+str(pred_dist))

    nb_dist_ref_keys = list(ref_dist.keys())
    nb_dist_ref_values = list(ref_dist.values())
    nb_dist_pred_keys = list(pred_dist.keys())
    nb_dist_pred_values = list(pred_dist.values())
        
    # Compute the Wasserstein distance
    return wasserstein_distance(
        np.array(nb_dist_ref_keys),
        np.array(nb_dist_pred_keys),
        np.array(nb_dist_ref_values),
        np.array(nb_dist_pred_values)
    )

class MultiLabelSamplingMetrics(nn.Module):
    """
    Sampling metrics for graphs with multilabel edge features.
    Builds reference graphs directly from datasets (no DataLoader),
    and builds MultiGraph for generated graphs by expanding each label as a separate edge.
    """
    def __init__(self, datamodule, multicat=False, ref_metrics=None, compute_emd=False, 
                 metrics_list=None, post_processing=None, data_augmentation=False):
        super().__init__()
        self.multicat = multicat
        self.ref_metrics = ref_metrics
        self.compute_emd = compute_emd
        self.metrics_list = metrics_list or ['degree', 'clustering', 'orbit', 'spectre', 'sbm']
        self.post_processing = post_processing
        if post_processing is not None:
            self.max_clique_size = post_processing["max_clique_size"]

        self.data_augmentation = data_augmentation
        if data_augmentation:
            self.train_hypergraphs = datamodule.hg_train_list
            self.val_hypergraphs = datamodule.hg_val_list
            self.test_hypergraphs = datamodule.hg_test_list
            self.nb_features = datamodule.num_layers
        else:
            # Build reference graphs from the raw datasets
            with contextlib.redirect_stderr(open(os.devnull, 'w')):
                self.train_hypergraphs = self._dataset_to_hnx(datamodule.train_dataset, name="training")
                self.val_hypergraphs   = self._dataset_to_hnx(datamodule.val_dataset, name="validation")
                self.test_hypergraphs  = self._dataset_to_hnx(datamodule.test_dataset, name="test")
    
    def decode_edge_attr_full(self, edge_index, edge_attr):
        """
        Reconvert edge_attr (one-hot) and edge_index into the original adjacency tensor [n, n, feat].

        Args:
            edge_index (Tensor): [2, num_edges]
            edge_attr (Tensor): [num_edges, 2^feat] (one-hot encoding of edge type combinations)

        Returns:
            adj (Tensor): [n, n, feat] reconstructed adjacency tensor
        """
        n = int(edge_index.max().item()) + 1

        if self.multicat:
            feat = edge_attr.size(1)
            binary_codes = edge_attr.float()
        else:
            num_combinations = edge_attr.size(1)
            feat = int(torch.log2(torch.tensor(num_combinations)).item())
            assert 2 ** feat == num_combinations, "edge_attr size must be power of 2"

            indices = edge_attr.argmax(dim=1)  # [num_edges]

            binary_codes = ((indices[:, None] & (1 << torch.arange(feat))) > 0).float()

        adj = torch.zeros((n, n, feat), dtype=torch.float)
        rows, cols = edge_index
        for idx in range(edge_index.size(1)):
            i, j = rows[idx].item(), cols[idx].item()
            adj[i, j] = binary_codes[idx]

        return adj
    
    def max_hyperedges_by_node(self, H: hnx.Hypergraph, max_size: int = 10) -> dict:
        members = H.incidence_dict 
    
        result = {}
        for i in range(1, max_size + 1):
            counts = Counter()
            for nodes in members.values():
                if len(nodes) == i:
                    counts.update(nodes)
            result[i] = counts.most_common(1)[0][1] if counts else 0
        
        return result
    
    def max_across_hypergraphs(self, hypers: list, max_size: int = 10) -> dict:
        global_max = {i: 0 for i in range(1, max_size + 1)}
        
        for H in hypers:
            local = self.max_hyperedges_by_node(H, max_size=max_size)
            for i, val in local.items():
                if val > global_max[i]:
                    global_max[i] = val
                    
        return global_max

    def _dataset_to_hnx(self, dataset, name=""):
        """
        Convert a PyTorch Geometric dataset to a list of unlabeled NetworkX graphs.
        Each unlabeled graph contains an edge if the corresponding coordinate of the label is equal to 1.
        """
        hypergraphs = []
        self.nb_features = dataset.num_edge_features

        for idx in tqdm(range(len(dataset)), desc=f"Converting {name} dataset to hypergraphs"):
            max_cliques = []
            data = dataset[idx]
            n = int(data.num_nodes) if hasattr(data, 'num_nodes') else data.x.size(0)
            
            adj = self.decode_edge_attr_full(data.edge_index, data.edge_attr)

            for label_id in range(adj.shape[-1]):
                G = nx.Graph()
                G.add_nodes_from(range(n))

                # Add edges based on edge_index and label
                src, dst = data.edge_index
                edge_labels = data.edge_attr[:, label_id]
                for u, v, label in zip(src.tolist(), dst.tolist(), edge_labels.tolist()):
                    if u != v and label == 1:
                        G.add_edge(u, v)

                # Sample maximal cliques from the graph
                cliques = list(nx.find_cliques(G))
                cliques = [clique for clique in cliques if len(clique) > 1]
                max_cliques.extend(cliques)

            # Create a hypergraph from the maximal cliques
            unique_hyperedges = list(set(frozenset(he) for he in max_cliques))

            nb_nodes = max([max(he) for he in unique_hyperedges]) + 1
            incidence_matrix = np.zeros((nb_nodes, len(unique_hyperedges)), dtype=int)
            for i, he in enumerate(unique_hyperedges):
                for node in he:
                    incidence_matrix[node, i] = 1
            he_dict = {i: set(e) for i, e in enumerate(unique_hyperedges)}
            hg = hnx.Hypergraph.from_incidence_matrix(incidence_matrix, he_dict=he_dict)
            hypergraphs.append(hg)

        return hypergraphs
    
    def graph_superpositions_to_hypergraphs(self, generated_graphs):
        pred_hnx = []
        for hypergraph_id, data in enumerate(generated_graphs):
            (node_types, edge_types) = data
            nb_nodes = node_types.shape[0]
            assert nb_nodes == edge_types.shape[0]
            assert edge_types.shape[1] == nb_nodes
            edge_types_list = edge_types.to(torch.long).tolist()
            max_cliques = []
            for k in range(self.nb_features):
                G = nx.Graph()
                G.add_nodes_from(range(nb_nodes))
                if self.multicat:
                    for u in range(nb_nodes):
                        for v in range(u, nb_nodes):
                            if edge_types_list[u][v][k] == 1:
                                G.add_edge(u, v)
                else:
                    exp = 2**k
                    for u in range(nb_nodes):
                        for v in range(u, nb_nodes):
                            if edge_types_list[u][v] % exp == 1:
                                G.add_edge(u, v)
                cliques = list(nx.find_cliques(G))
                cliques = [clique for clique in cliques if len(clique) > 1]
                max_cliques.extend(cliques)

            unique_hyperedges = list(set(frozenset(he) for he in max_cliques))

            incidence_matrix = np.zeros((nb_nodes, len(unique_hyperedges)), dtype=int)
            for i, he in enumerate(unique_hyperedges):
                for node in he:
                    incidence_matrix[node, i] = 1
            he_dict = {i: set(e) for i, e in enumerate(unique_hyperedges)}
            hg = hnx.Hypergraph.from_incidence_matrix(incidence_matrix, he_dict=he_dict)
            pred_hnx.append(hg)
        return pred_hnx

    
    

    def forward(self, generated_graphs: list, name: str,
                current_epoch: int, val_counter: int,
                local_rank: int, test: bool = False):
        torch.cuda.synchronize()
        start = time.time()
        # Select reference graphs
        torch.set_printoptions(threshold=float('inf'))
        
        reference = self.test_hypergraphs if test else self.val_hypergraphs
        if local_rank == 0:
            print(f"Computing multilabel sampling metrics between {len(generated_graphs)} generated and {len(reference)} reference labeled graphs (EMD={self.compute_emd})")
        
        if self.multicat:
            nb_graphs = generated_graphs[0][1].shape[-1]
            print("Number of graphs to represent one hypergraph: "+str(nb_graphs))
        
        else:
            nb_graphs = int(math.log2(generated_graphs[0][1].shape[-1])) + 1
            print("Number of graphs to represent one hypergraph: "+str(nb_graphs))

        nb_nodes = generated_graphs[0][0].shape[0]
        if self.post_processing is not None:
            node_hyperedge_matrices = np.zeros((len(generated_graphs), nb_nodes, nb_graphs, self.max_clique_size-1))

        pred_hnx = self.graph_superpositions_to_hypergraphs(generated_graphs)
        empty_hypergraphs_indices = [i for i in range(len(pred_hnx)) if (len(pred_hnx[i].edges) == 0 or len(pred_hnx[i].nodes) == 0)]
        for i in empty_hypergraphs_indices:
            print(f"Empty hypergraph at index {i}, removing it from the list.")
            print(f"Corresponding graph superposition: {generated_graphs[i]}")
        
        pred_hnx = [H for H in pred_hnx if not (len(H.edges) == 0 or len(H.nodes) == 0)]

        to_log = {}

        # Node degree
        if 'node_degree' in self.metrics_list:
            if local_rank == 0: print("Computing node degree stats...")
            if local_rank == 0: print("Computing node degree Wasserstein distance...")
            wasserstein_dist_ref = node_degree_wasserstein_graph(reference, pred_hnx)
            wasserstein_dist_train = node_degree_wasserstein_graph(self.train_hypergraphs, pred_hnx)
            print("Node degree Wasserstein distance (w.r.t reference dataset):", wasserstein_dist_ref)
            print("Node degree Wasserstein distance (w.r.t training dataset):", wasserstein_dist_train)

            if self.post_processing is not None:
                wasserstein_dist_label = node_degree_wasserstein_label(reference, node_hyperedge_matrices)
                print("Node degree Wasserstein distance (with the labels):", wasserstein_dist_label)
                to_log['node_degree_wasserstein_label'] = wasserstein_dist_label
            
            to_log['node_degree_wasserstein_ref'] = wasserstein_dist_ref
            to_log['node_degree_wasserstein_train'] = wasserstein_dist_train
            # wandb.log({'node_degree_wasserstein': wasserstein_dist})
        
        # Edge size
        if 'edge_size' in self.metrics_list:
            if local_rank == 0: print("Computing edge size stats...")
            wasserstein_dist_ref, _, _ = edge_size_wasserstein_graph(reference, pred_hnx)
            wasserstein_dist_train, ref_dist_graph, pred_dist_graph = edge_size_wasserstein_graph(self.train_hypergraphs, pred_hnx)
            print("Edge size Wasserstein distance (w.r.t. reference dataset):", wasserstein_dist_ref)
            print("Edge size Wasserstein distance (w.r.t. training dataset):", wasserstein_dist_train)

            if self.post_processing is not None:
                wasserstein_dist_label, ref_dist_label, pred_dist_label = edge_size_wasserstein_label(reference, node_hyperedge_matrices)
                print("Edge size Wasserstein distance (with the labels):", wasserstein_dist_label)
                to_log['edge_size_wasserstein_label'] = wasserstein_dist_label

            to_log['edge_size_wasserstein_ref'] = wasserstein_dist_ref
            to_log['edge_size_wasserstein_train'] = wasserstein_dist_train
            # wandb.log({'edge_size_wasserstein_graph': wasserstein_dist})

            if 'clique_size' in self.metrics_list:
                if local_rank == 0: print("Computing clique stats...")
                total_cliques_ref = sum(ref_dist_graph.values())
                total_cliques_pred = sum(pred_dist_graph.values())
                pred_clique_metric = {f"prop_{k}_pred": v / total_cliques_pred for k, v in pred_dist_graph.items()}
                ref_clique_metric = {f"prop_{k}_ref": v / total_cliques_ref for k, v in ref_dist_graph.items()}
                to_log.update(pred_clique_metric)
                to_log.update(ref_clique_metric)

        # Spectral
        if 'spectral' in self.metrics_list:
            if local_rank == 0: print("Computing spectral stats...")
            s_ref = spectral_stats(reference, pred_hnx, is_parallel=False, n_eigvals=-1, compute_emd=False)
            s_train = spectral_stats(self.train_hypergraphs, pred_hnx, is_parallel=False, n_eigvals=-1, compute_emd=False)
            print("Distance between normalized laplacians eigenvalues distributions (with reference dataset):", s_ref)
            print("Distance between normalized laplacians eigenvalues distributions (with training dataset):", s_train)
            to_log['spectral_ref'] = s_ref
            to_log['spectral_train'] = s_train
            # wandb.log({'spectral': s})

        if 'is_sbm' in self.metrics_list:
            if local_rank == 0: print("Computing SBM stats...")
            nb_hg = len(pred_hnx)
            true_sbm = 0
            for i in range(nb_hg):
                with contextlib.redirect_stderr(open(os.devnull, 'w')):
                    if is_sbm_hypergraph(pred_hnx[i], p_intra=0.05, p_inter=0.001, k=3, strict=True, refinement_steps=100):
                        true_sbm += 1
            sbm_acc = true_sbm / nb_hg

            true_sbm_train = 0.
            for i in range(len(self.test_hypergraphs)):
                with contextlib.redirect_stderr(open(os.devnull, 'w')):
                    if is_sbm_hypergraph(self.test_hypergraphs[i], p_intra=0.05, p_inter=0.001, k=3, strict=True, refinement_steps=100):
                        true_sbm_train += 1
            sbm_acc_train = true_sbm_train / len(self.test_hypergraphs)
            print("SBM accuracy:", sbm_acc)
            print("SBM accuracy for the training dataset:", sbm_acc_train)
            to_log['is_sbm'] = sbm_acc
            # wandb.log({'is_sbm': sbm_acc})
        
        if 'is_hypertree' in self.metrics_list:
            if local_rank == 0: print("Computing hypertree stats...")
            nb_hg = len(pred_hnx)
            true_hypertree = 0
            for i in range(nb_hg):
                with contextlib.redirect_stderr(open(os.devnull, 'w')):
                    if is_hypertree(pred_hnx[i]):
                        true_hypertree += 1
            hypertree_acc = true_hypertree / nb_hg
            print("Hypertree accuracy:", hypertree_acc)
            to_log['is_hypertree'] = hypertree_acc
        
        if 'is_ego' in self.metrics_list:
            if local_rank == 0: print("Computing ego hypergraph stats...")
            nb_hg = len(pred_hnx)
            true_ego = 0
            for i in range(nb_hg):
                with contextlib.redirect_stderr(open(os.devnull, 'w')):
                    if is_ego_hypergraph(pred_hnx[i]):
                        true_ego += 1
            ego_acc = true_ego / nb_hg
            print("Ego hypergraph accuracy:", ego_acc)
            to_log['is_ego'] = ego_acc

            nb_ref_hg = len(reference)
            true_ego_ref = 0
            for i in range(nb_ref_hg):
                with contextlib.redirect_stderr(open(os.devnull, 'w')):
                    if is_ego_hypergraph(reference[i]):
                        true_ego_ref += 1
            ego_acc_ref = true_ego_ref / nb_ref_hg
            print("Ego hypergraph accuracy (with the reference dataset):", ego_acc_ref)
        
        if 'uniqueness' in self.metrics_list:
            if local_rank == 0: print("Computing uniqueness stats...")
            uniqueness_value = uniqueness(pred_hnx)
            print("Proportion of unique hypergraphs:", uniqueness_value)
            to_log['uniqueness'] = uniqueness_value
        
        if 'novelty' in self.metrics_list:
            if local_rank == 0: print("Computing novelty stats...")
            with contextlib.redirect_stderr(open(os.devnull, 'w')):
                novelty_value = novelty(self.train_hypergraphs, pred_hnx)
            print("Proportion of new hypergraphs:", novelty_value)
            to_log['novelty'] = novelty_value
        
        if 'nb_nodes' in self.metrics_list:
            if local_rank == 0: print("Computing stats on the number of nodes...")
            nb_nodes_dist = node_dist(self.train_hypergraphs, pred_hnx)
            print("Distance between node number distributions: ", nb_nodes_dist)
            to_log['nb_nodes_wasserstein'] = nb_nodes_dist
        
        if 'centrality_closeness' in self.metrics_list:
            if local_rank == 0: print("Computing stats on centrality closeness...")
            with contextlib.redirect_stderr(open(os.devnull, 'w')):
                centr_closeness_train = centrality_closeness(self.train_hypergraphs, pred_hnx)
            print("Centrality closeness metric (with the training dataset): ", centr_closeness_train)
            to_log['centrality_closeness_train'] = centr_closeness_train
            with contextlib.redirect_stderr(open(os.devnull, 'w')):
                centr_closeness_ref = centrality_closeness(reference, pred_hnx)
            print("Centrality closeness metric (with the reference dataset): ", centr_closeness_ref)
            to_log['centrality_closeness_ref'] = centr_closeness_ref

        if 'centrality_betweenness' in self.metrics_list:
            if local_rank == 0: print("Computing stats on centrality betweenness...")
            with contextlib.redirect_stderr(open(os.devnull, 'w')):
                centr_betweenness_train = centrality_betweenness(self.train_hypergraphs, pred_hnx)
            print("Centrality betweenness metric (with the training dataset): ", centr_betweenness_train)
            to_log['centrality_betweenness_train'] = centr_betweenness_train
            with contextlib.redirect_stderr(open(os.devnull, 'w')):
                centr_betweenness_ref = centrality_betweenness(reference, pred_hnx)
            print("Centrality betweenness metric (with the reference dataset): ", centr_betweenness_ref)
            to_log['centrality_betweenness_ref'] = centr_betweenness_ref

        if 'centrality_harmonic' in self.metrics_list:
            if local_rank == 0: print("Computing stats on centrality harmonic...")
            with contextlib.redirect_stderr(open(os.devnull, 'w')):
                centr_harmonic_train = centrality_harmonic(self.train_hypergraphs, pred_hnx)
            print("Centrality harmonic metric (with the training dataset): ", centr_harmonic_train)
            to_log['centrality_harmonic_train'] = centr_harmonic_train
            with contextlib.redirect_stderr(open(os.devnull, 'w')):
                centr_harmonic_ref = centrality_harmonic(reference, pred_hnx)
            print("Centrality harmonic metric (with the reference dataset): ", centr_harmonic_ref)
            to_log['centrality_harmonic_ref'] = centr_harmonic_ref
        
        if 'wg_clustering_coef' in self.metrics_list:
            if local_rank == 0: print("Computing stats on the weighted graph clustering coefficient...")
            wg_clustering_coef_train = weighted_graph_clustering_coef_stats(self.train_hypergraphs, pred_hnx)
            print("Weighted graph clustering coefficient metric (with the training dataset): ", wg_clustering_coef_train)
            to_log['wg_clustering_coef_train'] = wg_clustering_coef_train
            wg_clustering_coef_ref = weighted_graph_clustering_coef_stats(reference, pred_hnx)
            print("Weighted graph clustering coefficient metric (with the reference dataset): ", wg_clustering_coef_ref)
            to_log['wg_clustering_graph_ref'] = wg_clustering_coef_ref
        
        if 'sbm_score' in self.metrics_list:
            if local_rank == 0: print("Computing SBM score...")
            sbm_score_train = 0.
            sbm_score_ref = 0.
            for hg in self.train_hypergraphs:
                with contextlib.redirect_stderr(open(os.devnull, 'w')):
                    sbm_score_train += is_sbm_hypergraph(hg, p_intra=0.05, p_inter=0.001, k=3, strict=False)
            sbm_score_train /= len(self.train_hypergraphs)
            print("SBM score (with the training dataset): ", sbm_score_train)
            to_log['sbm_score_train'] = sbm_score_train

            for hg in reference:
                with contextlib.redirect_stderr(open(os.devnull, 'w')):
                    sbm_score_ref += is_sbm_hypergraph(hg, p_intra=0.05, p_inter=0.001, k=3, strict=False)
            sbm_score_ref /= len(reference)
            print("SBM score (with the reference dataset): ", sbm_score_ref)
            to_log['sbm_score_ref'] = sbm_score_ref

            sbm_score_pred = 0.
            for hg in pred_hnx:
                with contextlib.redirect_stderr(open(os.devnull, 'w')):
                    sbm_score_pred += is_sbm_hypergraph(hg, p_intra=0.05, p_inter=0.001, k=3, strict=False)
            sbm_score_pred /= len(pred_hnx)
            print("SBM score (with the predicted hypergraphs): ", sbm_score_pred)
            to_log['sbm_score_pred'] = sbm_score_pred
        
        if 'hypertree_score' in self.metrics_list:
            if local_rank == 0: print("Computing hypertree score...")
            hypertree_score_train = 0
            hypertree_score_ref = 0
            for hg in self.train_hypergraphs:
                with contextlib.redirect_stderr(open(os.devnull, 'w')):
                    hypertree_score_train += is_hypertree(hg)
            hypertree_score_train /= len(self.train_hypergraphs)
            print("Hypertree score (with the training dataset): ", hypertree_score_train)
            to_log['hypertree_score_train'] = hypertree_score_train

            for hg in reference:
                with contextlib.redirect_stderr(open(os.devnull, 'w')):
                    hypertree_score_ref += is_hypertree(hg)
            hypertree_score_ref /= len(reference)
            print("Hypertree score (with the reference dataset): ", hypertree_score_ref)
            to_log['hypertree_score_ref'] = hypertree_score_ref

            for hg in pred_hnx:
                with contextlib.redirect_stderr(open(os.devnull, 'w')):
                    hypertree_score_pred += is_hypertree(hg)
            hypertree_score_pred /= len(pred_hnx)
            print("Hypertree score (with the predicted hypergraphs): ", hypertree_score_pred)
            to_log['hypertree_score_pred'] = hypertree_score_pred

        if wandb.run:
            # Log the metrics to wandb
            baseline_log = {}
            for metric in self.ref_metrics:
                if self.ref_metrics is not None and metric in self.ref_metrics:
                    for key, ref_value in self.ref_metrics[metric].items():
                        baseline_log[f"{metric}_{key}"] = ref_value
                else:
                    print(f"Warning: {metric} not in ref_metrics")
                    print(f"self.ref_metrics: {self.ref_metrics}")
            
            to_log.update(baseline_log)
            print("val_counter: "+str(val_counter))
            wandb.log(to_log, commit=True)
            wandb.run.summary[name] = to_log
            wandb.run.summary[name + '_epoch'] = current_epoch
            print(f"Logged metrics to wandb: {to_log}")
        
        torch.cuda.synchronize()
        end = time.time()
        print(f"Metrics computation took {end - start:.2f} seconds")

        return to_log

    def reset(self):
        """
        Reset the metrics. This is called at the beginning of each epoch.
        """
        pass