import os
import pathlib
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
import torch_geometric.utils as pyg_utils
import pytorch_lightning as pl
import src.utils as utils
from tqdm import tqdm
import networkx as nx


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


class CustomGraphDataset(InMemoryDataset):
    """
    Dataset for custom graphs stored as .pt files containing lists of numpy arrays or Tensors with shape [n, n, feat].
    Converts each adjacency tensor to a PyG Data object with multilabel edges.
    """
    def __init__(self, dataset_name, split, root, multilabel=False,
                 transform=None, pre_transform=None, pre_filter=None, post_processing=None):
        self.dataset_name = dataset_name
        self.multilabel = multilabel
        self.post_processing = post_processing
        self.split = split  # 'train', 'val', or 'test'
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)


    @property
    def raw_file_names(self):
        # Raw files under raw_dir: train.pt, val.pt, test.pt
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self):
        # Single processed file for this split
        return [f'{self.split}.pt']

    @property
    def raw_dir(self):
        # Raw files are located in the 'raw' subfolder of root
        return os.path.join(self.root, 'raw')

    def process(self):
        split_to_idx = {'train': 0, 'val': 1, 'test': 2}
        raw_path = self.raw_paths[split_to_idx[self.split]]
        raw_list = torch.load(raw_path, weights_only=False)
        print(f"Processing {self.split} dataset with {len(raw_list)} graphs.")

        data_list = []
        x_list = []
        print(f"Processing graphs from {self.split} dataset: {len(raw_list)} hypergraphs in the datsets", end=' ')
        for i, adj in tqdm(enumerate(raw_list)):
            # Convert to Tensor
            if isinstance(adj, np.ndarray):
                adj = torch.from_numpy(adj).float()
            else:
                adj = adj.float()
            assert adj.dim() == 3 and adj.shape[0] == adj.shape[1], \
                f"Adj[{i}].shape invalid: {adj.shape}"
            n, _, feat = adj.shape

            if self.post_processing is not None:
                # Node features: number of maximal cliques of each size
                x = get_node_info(adj, self.post_processing["max_clique_size"])

            else:
                # Node features: simple ones
                x = torch.ones((n, 1), dtype=torch.float)

            # Build multilabel edge_index and edge_attr: keep full feat dim
            # Flatten each feature channel separately: edge_index and edge_attr

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

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.zeros(1, 0),
                num_nodes=n
            )

            x_list.append(x)


            if self.pre_filter and not self.pre_filter(data):
                continue
            if self.pre_transform:
                data = self.pre_transform(data)

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class CustomGraphDataModule(pl.LightningDataModule):
    """
    LightningDataModule for the CustomGraphDataset.
    Provides train/val/test dataloaders using PyG DataLoader.
    """
    def __init__(self, cfg, post_processing=None):
        super().__init__()
        self.cfg = cfg
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root = os.path.join(base_path, cfg.dataset.datadir)

        # Instantiate each split
        self.train_dataset = CustomGraphDataset(
            dataset_name=cfg.dataset.name,
            split='train',
            root=root,
            pre_transform=None,
            multilabel=cfg.dataset.multilabel,
            post_processing=post_processing
        )
        self.val_dataset = CustomGraphDataset(
            dataset_name=cfg.dataset.name,
            split='val',
            root=root,
            pre_transform=None,
            multilabel=cfg.dataset.multilabel,
            post_processing=post_processing
        )
        self.test_dataset = CustomGraphDataset(
            dataset_name=cfg.dataset.name,
            split='test',
            root=root,
            pre_transform=None,
            multilabel=cfg.dataset.multilabel,
            post_processing=post_processing
        )

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            pin_memory=getattr(self.cfg.dataset, 'pin_memory', False),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            pin_memory=getattr(self.cfg.dataset, 'pin_memory', False),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            pin_memory=getattr(self.cfg.dataset, 'pin_memory', False),
        )


class CustomDatasetInfos:
    """
    DatasetInfos for CustomGraphDataset with multilabel edges.
    Extracts feature dims, node distribution, and maximum number of nodes.
    """
    def __init__(self, datamodule, dataset_config):
        from src.diffusion.distributions import DistributionNodes
        self.datamodule = datamodule
        self.name = 'nx_graphs'
        self.post_processing = dataset_config['post_processing']


        # Sample first graph to get feature dims
        sample = datamodule.train_dataset[0]
        self.node_feature_dim = sample.x.size(1)
        self.edge_feature_dim = sample.edge_attr.size(1)

        # No node types
        self.node_types = torch.tensor([1])
        # Uniform distribution over edge labels
        self.edge_types = torch.ones(self.edge_feature_dim) / self.edge_feature_dim

        # Compute maximum number of nodes and node count distribution
        node_counts = []
        nb_edges = 0
        nb_potential_edges = 0
        self.max_n_nodes = 0
        for data in datamodule.train_dataset:
            n = int(data.num_nodes) if hasattr(data, 'num_nodes') else data.x.size(0)
            node_counts.append(n)
            nb_edges += (data.edge_attr != 0).sum().item()
            nb_potential_edges += n * (n - 1) * data.edge_attr.size(-1) / 2
            if n > self.max_n_nodes:
                self.max_n_nodes = n

        for data in datamodule.val_dataset:
            n = int(data.num_nodes) if hasattr(data, 'num_nodes') else data.x.size(0)
            if n > self.max_n_nodes:
                self.max_n_nodes = n
        
        for data in datamodule.test_dataset:
            n = int(data.num_nodes) if hasattr(data, 'num_nodes') else data.x.size(0)
            if n > self.max_n_nodes:
                self.max_n_nodes = n

        self.prop_edges = nb_edges / (nb_potential_edges + 1e-8)
        print("Proportion of edges: ", self.prop_edges)

        counts = torch.zeros(self.max_n_nodes + 1, dtype=torch.float)
        for n in node_counts:
            counts[n] += 1
        counts = counts / counts.sum()
        self.nodes_dist = DistributionNodes(counts)
        print(f"Node distribution: {self.nodes_dist.prob}")
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
        example_batch = next(iter(self.datamodule.train_dataloader()))
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
