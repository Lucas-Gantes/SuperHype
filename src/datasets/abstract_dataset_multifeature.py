import torch
from torch_geometric.data.lightning import LightningDataset
from src.datasets.abstract_dataset import AbstractDataModule
from src.diffusion.distributions import DistributionNodes

class MultiFeatureDataModule(AbstractDataModule):
    """
    DataModule for graphs with multidimensional edge features.
    Overrides compute_input_output_dims to avoid converting to dense adjacency,
    and directly infers feature dimensions from a single sample.
    """
    def __init__(self, cfg, train_dataset, val_dataset, test_dataset):
        datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
        super().__init__(cfg, datasets)

    def compute_input_output_dims(self, **kwargs):
        sample = self.train_dataset[0]
        x_dim = sample.x.size(1)
        e_dim = sample.edge_attr.size(1)
        y_dim = sample.y.size(1)
        self.input_dims = {'X': x_dim, 'E': e_dim, 'y': y_dim + 1}
        self.output_dims = {'X': x_dim, 'E': e_dim, 'y': 0}
        return self.input_dims, self.output_dims

class MultiFeatureDatasetInfos:
    """
    DatasetInfos for graphs with multidimensional edge features.
    Collects basic dataset statistics directly from datasets.
    """
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = dataset_config.name
        
        # Feature dimensions from a single sample
        sample = datamodule.train_dataset[0]
        self.node_feature_dim = sample.x.size(1)
        self.edge_feature_dim = sample.edge_attr.size(1)
        
        # Node count distribution
        counts = self._compute_node_counts_distribution(datamodule.train_dataset)
        self.nodes_dist = DistributionNodes(counts)
        self.max_n_nodes = counts.size(0) - 1

    def _compute_node_counts_distribution(self, dataset):
        node_counts = [
            int(data.num_nodes) if hasattr(data, 'num_nodes') else data.x.size(0)
            for data in dataset
        ]
        max_n = max(node_counts)
        counts = torch.zeros(max_n + 1)
        for n in node_counts:
            counts[n] += 1
        return counts / counts.sum()

    def compute_input_output_dims(self, **kwargs):
        sample = self.datamodule.train_dataset[0]
        x_dim = sample.x.size(1)
        e_dim = sample.edge_attr.size(1)
        y_dim = sample.y.size(1)
        self.input_dims = {'X': x_dim, 'E': e_dim, 'y': y_dim + 1}
        self.output_dims = {'X': x_dim, 'E': e_dim, 'y': 0}
        return self.input_dims, self.output_dims
