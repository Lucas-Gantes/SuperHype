import numpy as np
import graph_tool.all as gt
from itertools import combinations
import hypernetx as hnx
import math
from scipy.stats import chi2
from tqdm import tqdm


def reindex_hypergraph_from_incidence(
    hg: hnx.Hypergraph
) -> hnx.Hypergraph:
    """
    Reindex nodes of a Hypergraph to contiguous integers 0..n-1 by reconstructing
    un hypergraph à partir de sa matrice d'incidence.

    Args:
        hg: Hypergraph original avec labels arbitraires.

    Returns:
        Un nouvel Hypergraph dont les nœuds sont remis à 0..N-1, construit via
        Hypergraph.from_incidence_matrix, garantissant des labels contigus.
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



def reindex_hypergraph(hg: hnx.Hypergraph) -> hnx.Hypergraph:
    """
    Reindex the hypergraph to have integer labels starting from 0.
    """
    original_nodes = list(hg.nodes)
    label_to_idx = { label: idx for idx, label in enumerate(original_nodes) }

    new_edges = {}
    for edge_id, members in hg.edges.incidence_dict.items():
        new_members = [ label_to_idx[node] for node in members ]
        new_edges[edge_id] = new_members

    return hnx.Hypergraph(new_edges)


def reindex_hypergraph_inplace(
    hg: hnx.Hypergraph
) -> hnx.Hypergraph:
    """
    Réindexe les nœuds d'un Hypergraph en place pour qu'ils soient contigus 0..n-1,
    sans reconstruire l'objet Hypergraph, afin de préserver toute structure interne
    (attributs, métadonnées) utilisée par is_sbm_hypergraphs.

    Args:
        hg: Hypergraph original

    Returns:
        Le même Hypergraph avec :
          - _node_dict mis à jour avec les nouveaux indices,
          - _incidence_dict mis à jour avec les nouveaux indices,
          - aucune perte d'attribut ou de métadonnée interne.
    """
    original_nodes = list(hg.nodes)
    mapping = {old: new for new, old in enumerate(original_nodes)}

    if hasattr(hg, '_node_dict'):
        new_node_dict = {mapping[old]: attrs for old, attrs in hg._node_dict.items()}
        hg._node_dict.clear()
        hg._node_dict.update(new_node_dict)

    if hasattr(hg.edges, 'incidence_dict'):
        new_incidence = {}
        for edge_id, members in hg.edges.incidence_dict.items():
            new_incidence[edge_id] = [mapping[n] for n in members]
        hg.edges.incidence_dict.clear()
        hg.edges.incidence_dict.update(new_incidence)

    return hg



def generate_sbm_hypergraphs(num_hypergraphs, min_size, max_size, p, q, k, seed=0):
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

            if H.is_connected():
                hypergraphs.append(reindex_hypergraph_from_incidence(H))

    return hypergraphs


def generate_sbm_hypergraphs2(num_hypergraphs, min_size, max_size, p, q, k, seed=0, node_condition=False):
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
                    hypergraphs.append(reindex_hypergraph_from_incidence(H))
            else:
                if H.is_connected():
                    hypergraphs.append(reindex_hypergraph_from_incidence(H))
        # Display the progressbar
        progression = (len(hypergraphs) / num_hypergraphs) * 100
        bar = '#' * (len(hypergraphs) * 50 // num_hypergraphs)
        print(f'\rGenerating SBM random hypergraphs: [{bar:<50}] {progression:.2f}%', end='', flush=True)

    print("\nDone!")

    return hypergraphs




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
        est_p_intra.append(intra_edges / possible_intra_edges)
        
        for j in range(i+1, n_blocks):
            other_block_nodes = [v for v, block in enumerate(b) if block == j]
            inter_edges = sum(1 for edge in H.edges() if 
                              len(set(H.edges[edge]) & set(block_nodes)) >= 1 and 
                              len(set(H.edges[edge]) & set(other_block_nodes)) >= 1)
            possible_inter_edges = math.comb(len(block_nodes) + len(other_block_nodes), k) - math.comb(len(block_nodes), k) - math.comb(len(other_block_nodes), k)
            
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


sbm_list = generate_sbm_hypergraphs2(128, 32, 32, 0.05, 0.001, 3, seed=0)

print(f"Hyperedges in the first hypergraph: {list(sbm_list[0].edges)}")
print(f"Nodes in the first hypergraph: {list(sbm_list[0].nodes)}")
sbm_score = 0
for H in tqdm(sbm_list):
    if len(list(H.nodes)) < 32:
        print(f"Nodes in hypergraph n° {sbm_list.index(H)}: {list(H.nodes)}")
    if is_sbm_hypergraph(H, p_intra=0.05, p_inter=0.001, k=3, strict=True, refinement_steps=1000):
        sbm_score += 1
        print(f"Hypergraph {H} is a valid SBM hypergraph.")
    else:
        print(f"Hypergraph {H} is NOT a valid SBM hypergraph.")
  
print("Number of correct SBM hypergraphs: "+str(sbm_score))