import os
import pathlib
import torch
import numpy as np
from torch.utils.data import IterableDataset
from torch_geometric.data import Data
import torch_geometric.utils as pyg_utils
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
import src.utils as utils
from tqdm import tqdm
import networkx as nx
import hypernetx as hnx
import random
import contextlib
import pickle

nb_attempts = 1000


def get_node_info(adj_matrices, k):
    """
    Computes the number of maximal cliques of size i containing each node represented by the graph
    (for 2 <= i <= k)
    """
    G_list = [nx.from_numpy_array(np.array(adj_matrices[...,i])) for i in range(adj_matrices.shape[-1])]

    n = adj_matrices.size(0)
    assert adj_matrices.shape[0] == adj_matrices.shape[1]

    nb_graphs = len(G_list)

    cliques_counter = torch.zeros((n, nb_graphs*(k-1)+1), dtype = torch.float)
    # First column: discrete diffusion node labels (i.e. 1)
    for i in range(n):
        cliques_counter[i, 0] = 1.

    for i in range(len(G_list)):
        G = G_list[i]
        
        cliques = list(nx.find_cliques(G))
        for clique in cliques:
            size = len(clique)
            if 2 <= size <= k:
                for node in clique:
                    cliques_counter[node, i*(k-1) + size-1] += 1
    
    return cliques_counter

def recreate_hypergraphs(dataset, name=""):
    """
    Convert a multiple graph projection dataset into a raw hypergraph dataset
    """
    hypergraphs = []

    for idx in tqdm(range(len(dataset)), desc=f"Converting {name} dataset to hypergraphs"):
        max_cliques = []
        data = dataset[idx]
        n = int(data.num_nodes) if hasattr(data, 'num_nodes') else data.x.size(0)
        
        adj = dataset[idx]

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
        # he_dict = {i: set(e) for i, e in enumerate(unique_hyperedges)}
        he_dict = {
            f"edge_{i}": set(nodes)
            for i, nodes in enumerate(unique_hyperedges)
        }
        hg = hnx.Hypergraph(he_dict)
        hypergraphs.append(hg)
        
    return hypergraphs


def can_add_clique(new_graph, max_cliques, new_hyperedge):
    new_max_cliques = [ set(cl) for cl in nx.find_cliques(nx.from_numpy_array(new_graph)) if len(cl) >= 2]

    allowed = set(map(frozenset, max_cliques)) | {frozenset(new_hyperedge)}

    new_max_cliques_set = set(map(frozenset, new_max_cliques))
    
    return new_max_cliques_set == allowed


def add_clique_fixed_layers(graph_list, max_cliques_list, clique, num_layers):
    id_adding_graph = -1

    for graph_id in random.sample(list(range(num_layers)), k=num_layers):
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
        return False
    else:
        for i in clique:
            for j in clique:
                if i != j:
                    graph_list[id_adding_graph][i, j] = 1
        max_cliques_list[id_adding_graph].append(clique)
        return True


def get_multiple_projections(hg, num_layers):
    nb_nodes = len(hg.nodes)

    he_list = list(hg.edges.incidence_dict.values())
    he_list = [set(he) for he in he_list]

    max_cliques_graphs = [[] for i in range(num_layers)]
    generated_graphs = np.zeros((num_layers, nb_nodes, nb_nodes))

    for he in he_list:
        is_success = add_clique_fixed_layers(generated_graphs, max_cliques_graphs, he, num_layers)
        if not is_success:
            return None

    return generated_graphs


def make_hypergraph_sparse_projection(hg, num_layers):
    """
    Generate the labeled graphs corresponding to the multiple graph projection of the hypergraphs
    The representation will have num_layers layer in the end
    """
    with contextlib.redirect_stderr(open(os.devnull, 'w')):
        for j in range(nb_attempts):
            proj_graphs = get_multiple_projections(hg, num_layers)
            if proj_graphs is not None:
                break
    if proj_graphs is None:
        raise RuntimeError(f"Failed to create a projection for hypergraph n°{i} after {nb_attempts} attempts")
        
    # Create the labeled adjacency matrices
    labeled_adj_mat = np.zeros((proj_graphs[0].shape[0], proj_graphs[0].shape[1], num_layers))
    for i, mat in enumerate(proj_graphs):
        labeled_adj_mat[:, :, i] = mat

    return labeled_adj_mat


class CustomGraphAugmentedDataset(IterableDataset):
    def __init__(self, hg_list, samples_per_epoch: int, num_layers: int, post_processing: dict, multilabel: bool, change_rate: float):
        self.samples_per_epoch = samples_per_epoch
        self.hg_list = hg_list
        self.num_layers = num_layers
        self.post_processing = post_processing
        self.multilabel = multilabel
        self.change_rate = change_rate
        if change_rate is not None:
            self.already_projected = []
            for hg in self.hg_list:
                adj = make_hypergraph_sparse_projection(hg, self.num_layers)
                adj = torch.from_numpy(adj).float()
                adj = adj.float()

                n, _, feat = adj.shape

                if self.post_processing is not None:
                    x = get_node_info(adj, self.post_processing["max_clique_size"])
                else:
                    x = torch.ones((n, 1), dtype=torch.float)

                mask_any = (adj != 0).any(dim=-1)
                rows, cols = mask_any.nonzero(as_tuple=True)
                keep = rows != cols
                rows, cols = rows[keep], cols[keep]
                edge_index = torch.stack([rows, cols], dim=0)

                if self.multilabel:
                    mask_any = (adj != 0).any(dim=-1)
                    rows, cols = mask_any.nonzero(as_tuple=True)
                    keep = rows != cols
                    rows, cols = rows[keep], cols[keep]
                    edge_index = torch.stack([rows, cols], dim=0)
                    edge_attr_category = (adj[rows, cols, :] != 0).float()
                    edge_attr = edge_attr_category

                else:
                    mask_any = (adj != 0).any(dim=-1)
                    rows, cols = mask_any.nonzero(as_tuple=True)
                    keep = rows != cols
                    rows, cols = rows[keep], cols[keep]
                    edge_index = torch.stack([rows, cols], dim=0)
                    edge_attr_category = (adj[rows, cols, :] != 0).float()

                    num_categories = edge_attr_category.size(1)

                    num_combinations = 2 ** num_categories
                    powers = 2 ** torch.arange(num_categories)
                    indices = (edge_attr_category * powers).sum(dim=1)
                    indices = indices.to(torch.long)

                    edge_attr = torch.nn.functional.one_hot(indices, num_classes=num_combinations)
                    edge_attr = edge_attr.to(torch.float)

                self.already_projected.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                        y=torch.zeros(1, 0), num_nodes=n))

            self.indices_to_change = list(range(len(hg_list)))
            random.shuffle(self.indices_to_change)

    def __iter__(self):
        projections_epoch = []
        if self.change_rate is None:
            hg_list_epoch = random.sample(self.hg_list, self.samples_per_epoch)
            for hg in hg_list_epoch:
                adj = make_hypergraph_sparse_projection(hg, self.num_layers)
                adj = torch.from_numpy(adj).float()
                adj = adj.float()

            n, _, feat = adj.shape

            if self.post_processing is not None:
                x = get_node_info(adj, self.post_processing["max_clique_size"])
            else:
                x = torch.ones((n, 1), dtype=torch.float)

            mask_any = (adj != 0).any(dim=-1)
            rows, cols = mask_any.nonzero(as_tuple=True)
            keep = rows != cols
            rows, cols = rows[keep], cols[keep]
            edge_index = torch.stack([rows, cols], dim=0)

            if self.multilabel:
                mask_any = (adj != 0).any(dim=-1)
                rows, cols = mask_any.nonzero(as_tuple=True)
                keep = rows != cols
                rows, cols = rows[keep], cols[keep]
                edge_index = torch.stack([rows, cols], dim=0)
                edge_attr_category = (adj[rows, cols, :] != 0).float()
                edge_attr = edge_attr_category

            else:
                mask_any = (adj != 0).any(dim=-1)
                rows, cols = mask_any.nonzero(as_tuple=True)
                keep = rows != cols
                rows, cols = rows[keep], cols[keep]
                edge_index = torch.stack([rows, cols], dim=0)
                edge_attr_category = (adj[rows, cols, :] != 0).float()

                num_categories = edge_attr_category.size(1)

                num_combinations = 2 ** num_categories
                powers = 2 ** torch.arange(num_categories)
                indices = (edge_attr_category * powers).sum(dim=1)
                indices = indices.to(torch.long)

                edge_attr = torch.nn.functional.one_hot(indices, num_classes=num_combinations)
                edge_attr = edge_attr.to(torch.float)

            projections_epoch.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                       y=torch.zeros(1, 0), num_nodes=n))
            
        else:
            nb_to_change = round(len(self.hg_list)*self.change_rate)
            for i in range(nb_to_change):
                if len(self.indices_to_change)==0:
                    self.indices_to_change = list(range(len(self.hg_list)))
                    random.shuffle(self.indices_to_change)
                index_to_change = self.indices_to_change.pop()
                adj = make_hypergraph_sparse_projection(self.hg_list[index_to_change], self.num_layers)
                adj = torch.from_numpy(adj).float()
                adj = adj.float()
            
            n, _, feat = adj.shape

            if self.post_processing is not None:
                x = get_node_info(adj, self.post_processing["max_clique_size"])
            else:
                x = torch.ones((n, 1), dtype=torch.float)

            mask_any = (adj != 0).any(dim=-1)
            rows, cols = mask_any.nonzero(as_tuple=True)
            keep = rows != cols
            rows, cols = rows[keep], cols[keep]
            edge_index = torch.stack([rows, cols], dim=0)

            if self.multilabel:
                mask_any = (adj != 0).any(dim=-1)
                rows, cols = mask_any.nonzero(as_tuple=True)
                keep = rows != cols
                rows, cols = rows[keep], cols[keep]
                edge_index = torch.stack([rows, cols], dim=0)
                edge_attr_category = (adj[rows, cols, :] != 0).float()
                edge_attr = edge_attr_category

            else:
                mask_any = (adj != 0).any(dim=-1)
                rows, cols = mask_any.nonzero(as_tuple=True)
                keep = rows != cols
                rows, cols = rows[keep], cols[keep]
                edge_index = torch.stack([rows, cols], dim=0)
                edge_attr_category = (adj[rows, cols, :] != 0).float()

                num_categories = edge_attr_category.size(1)

                num_combinations = 2 ** num_categories
                powers = 2 ** torch.arange(num_categories)
                indices = (edge_attr_category * powers).sum(dim=1)
                indices = indices.to(torch.long)

                edge_attr = torch.nn.functional.one_hot(indices, num_classes=num_combinations)
                edge_attr = edge_attr.to(torch.float)

            self.already_projected[index_to_change] = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                       y=torch.zeros(1, 0), num_nodes=n)

            projections_epoch = random.sample(self.already_projected, self.samples_per_epoch)

        for proj in projections_epoch:
            yield proj

    def __len__(self):
        return self.samples_per_epoch


class GraphAugmentedDataModule(pl.LightningDataModule):
    def __init__(self, cfg, post_processing=None, change_rate=None):
        super().__init__()
        self.samples_per_epoch = cfg.dataset.samples_per_epoch
        self.num_layers = cfg.dataset.num_layers
        self.batch_size = cfg.train.batch_size
        self.num_workers = os.cpu_count() or 1
        self.pin_memory = getattr(cfg.dataset, 'pin_memory', False)
        self.multilabel = cfg.dataset.multilabel
        self.post_processing = post_processing

        base_path = os.path.join(os.path.dirname(__file__), '..', '..', cfg.dataset.datadir)
        raw_dir = os.path.join(base_path, 'raw')

        self.hg_train_file = os.path.join(raw_dir, 'hg_train.pkl')
        self.hg_val_file   = os.path.join(raw_dir, 'hg_val.pkl')
        self.hg_test_file  = os.path.join(raw_dir, 'hg_test.pkl')
        with open(self.hg_train_file, 'rb') as f:
            self.hg_train_list = pickle.load(f)
        with open(self.hg_val_file, 'rb') as f:
            self.hg_val_list = pickle.load(f)
        with open(self.hg_test_file, 'rb') as f:
            self.hg_test_list = pickle.load(f)
        
        self.change_rate = change_rate
        if change_rate is not None:
            print("The following change rate will be applied to the dataset: "+str(change_rate))
        else:
            print("All the dataset will be re-projected at every epoch")
    
    def _get_dl(self, hg_list, split, change_rate):
        if split == 'train':
            samples_per_epoch = self.samples_per_epoch
        elif split == 'val':
            samples_per_epoch = len(self.hg_val_list)
        elif split == 'test':
            samples_per_epoch = len(self.hg_test_list)
        else:
            raise NotImplementedError(f"{split} must be in ['train', 'val', 'test']")
        ds = CustomGraphAugmentedDataset(
            hg_list=hg_list,
            samples_per_epoch=samples_per_epoch,
            num_layers=self.num_layers,
            post_processing=self.post_processing,
            multilabel=self.multilabel,
            change_rate=change_rate
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            prefetch_factor=2
        )

    def train_dataloader(self):
        return self._get_dl(self.hg_train_list, 'train', change_rate=self.change_rate)

    def val_dataloader(self):
        return self._get_dl(self.hg_val_list, 'val', change_rate=None)

    def test_dataloader(self):
        return self._get_dl(self.hg_test_list, 'test', change_rate=None)


class CustomAugmentedDatasetInfos:
    """
    DatasetInfos for CustomGraphDataset with multilabel edges.
    Extracts feature dims, node distribution, and maximum number of nodes.
    """
    def __init__(self, datamodule):
        from src.diffusion.distributions import DistributionNodes
        self.datamodule = datamodule
        self.name = 'nx_graphs'
        self.post_processing = datamodule.post_processing

        # Sample first graph to get feature dims
        loader = datamodule._get_dl(datamodule.hg_train_list, 'train', change_rate=None)
        batch = next(iter(loader))
        self.node_feature_dim = batch.x.size(1)
        self.edge_feature_dim = batch.edge_attr.size(1)

        # No node types
        self.node_types = torch.tensor([1])
        # Uniform distribution over edge labels
        self.edge_types = torch.ones(self.edge_feature_dim) / self.edge_feature_dim

        # Compute maximum number of nodes and node count distribution
        node_counts = [len(hg.nodes) for hg in datamodule.hg_train_list]
        self.max_n_nodes = max(node_counts)
        counts = torch.bincount(torch.tensor(node_counts), minlength=self.max_n_nodes+1).float()
        self.nodes_dist = counts / counts.sum()

        counts = counts / counts.sum()
        self.nodes_dist = DistributionNodes(counts)
        example_batch = next(iter(self.datamodule.train_dataloader()))
        input_dims = {
            'X': example_batch['x'].size(1),
            'E': example_batch['edge_attr'].size(1),
            'y': example_batch['y'].size(1) + 1  # +1 for time conditioning
        }
        if self.post_processing is not None:
            self.nb_node_labels = (self.post_processing['max_clique_size'] - 1)*example_batch['edge_attr'].size(1)
            input_dims['X'] -= self.nb_node_labels
        else: self.nb_node_labels = None

    def compute_input_output_dims(self, extra_features, domain_features):
        # Compute input and output dimensions based on node and edge counts
        loader = self.datamodule._get_dl(self.datamodule.hg_train_list, 'train', change_rate=None)
        example_batch = next(iter(loader))
        ex_dense, node_mask = utils.to_dense(example_batch.x, example_batch.edge_index, example_batch.edge_attr,
                                             example_batch.batch)
        example_data = {'X_t': ex_dense.X, 'E_t': ex_dense.E, 'y_t': example_batch['y'], 'node_mask': node_mask}

        input_dims = {
            'X': example_batch['x'].size(1),
            'nb_layers': example_batch['edge_attr'].size(1),
            'Y': example_batch['y'].size(1) + 1,  # +1 for time conditioning
            'e': 1  # Only a binary to indicate the presence of edges
        }

        # Add extra features
        ex_extra_feat = extra_features(example_data)
        input_dims['X'] += ex_extra_feat.X.size(-1)
        input_dims['x'] = ex_extra_feat.x.size(-1)
        input_dims['e'] += ex_extra_feat.e.size(-1)
        input_dims['Y'] += ex_extra_feat.Y.size(-1)
        input_dims['y'] = ex_extra_feat.y.size(-1)

        # Add domain-specific features
        ex_extra_molecular_feat = domain_features(example_data)
        input_dims['X'] += ex_extra_molecular_feat.X.size(-1)
        input_dims['x'] += ex_extra_molecular_feat.x.size(-1)
        input_dims['e'] += ex_extra_molecular_feat.e.size(-1)
        input_dims['Y'] += ex_extra_molecular_feat.Y.size(-1)
        input_dims['y'] += ex_extra_molecular_feat.y.size(-1)

        if self.post_processing is not None:
            # Add node labels for maximal cliques
            input_dims['X'] -= (self.post_processing['max_clique_size'] - 1) * example_batch['edge_attr'].size(1)

        output_dims = {
            'X': 1,
            'x': 0,
            'nb_layers': example_batch['edge_attr'].size(1),
            'e': 1,
            'Y': 0,
            'y': 0
        }

        self.input_dims = input_dims
        self.output_dims = output_dims
