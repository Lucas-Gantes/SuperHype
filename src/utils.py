import os
import torch_geometric.utils
from omegaconf import OmegaConf, open_dict
from torch_geometric.utils import to_dense_adj, to_dense_batch
import torch
import omegaconf
import wandb
import numpy as np
import networkx as nx
from pytorch_lightning.callbacks import Callback


def create_folders(args):
    try:
        # os.makedirs('checkpoints')
        os.makedirs('graphs')
        os.makedirs('chains')
    except OSError:
        pass

    try:
        # os.makedirs('checkpoints/' + args.general.name)
        os.makedirs('graphs/' + args.general.name)
        os.makedirs('chains/' + args.general.name)
    except OSError:
        pass


def normalize(X, E, y, norm_values, norm_biases, node_mask):
    X = (X - norm_biases[0]) / norm_values[0]
    E = (E - norm_biases[1]) / norm_values[1]
    y = (y - norm_biases[2]) / norm_values[2]

    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask)


def unnormalize(X, E, y, norm_values, norm_biases, node_mask, collapse=False):
    """
    X : node features
    E : edge features
    y : global features`
    norm_values : [norm value X, norm value E, norm value y]
    norm_biases : same order
    node_mask
    """
    X = (X * norm_values[0] + norm_biases[0])
    E = (E * norm_values[1] + norm_biases[1])
    y = y * norm_values[2] + norm_biases[2]

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse)


def to_dense(x, edge_index, edge_attr, batch, multilabel=False):
    X, node_mask = to_dense_batch(x=x, batch=batch)
    # node_mask = node_mask.float()
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)
    # TODO: carefully check if setting node_mask as a bool breaks the continuous case
    max_num_nodes = X.size(1)
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    if not multilabel:
        E = encode_no_edge(E)
    return PlaceHolder(X=X, E=E, y=None), node_mask


def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E


def update_config_with_new_keys(cfg, saved_cfg):
    saved_general = saved_cfg.general
    saved_train = saved_cfg.train
    saved_model = saved_cfg.model

    for key, val in saved_general.items():
        OmegaConf.set_struct(cfg.general, True)
        with open_dict(cfg.general):
            if key not in cfg.general.keys():
                setattr(cfg.general, key, val)

    OmegaConf.set_struct(cfg.train, True)
    with open_dict(cfg.train):
        for key, val in saved_train.items():
            if key not in cfg.train.keys():
                setattr(cfg.train, key, val)

    OmegaConf.set_struct(cfg.model, True)
    with open_dict(cfg.model):
        for key, val in saved_model.items():
            if key not in cfg.model.keys():
                setattr(cfg.model, key, val)
    return cfg


class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False, multicat=False, layer_labels=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1
        if layer_labels:
            x_mask = x_mask.unsqueeze(-1)  # bs, n, 1, 1
            e_mask1 = e_mask1.unsqueeze(-1)  # bs, n, 1, 1, 1
            e_mask2 = e_mask2.unsqueeze(-1)  # bs, 1, n, 1, 1

        if collapse:
            if multicat:
                self.X = torch.argmax(self.X, dim=-1)        # (bs, n)
                self.X[~node_mask] = -1

                self.E = (self.E >= 0.5).to(torch.int64)     # (bs, n, n, d_e)
                invalid = (e_mask1 * e_mask2).squeeze(-1) == 0  # (bs, n, n)

                self.E[invalid] = 0
            else:
                self.X = torch.argmax(self.X, dim=-1)
                self.E = torch.argmax(self.E, dim=-1)

                self.X[node_mask == 0] = - 1
                self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * (e_mask1 * e_mask2)
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self

class PlaceHolderMultilayer:
    def __init__(self, X, x, e, Y, y):
        """ 
        A placeholder class for handling multi-layer graph data.
        Args:
        X: Node features common to all layers (tensor of shape [batch_size, num_nodes, feature_dim])
        x: Node features for each layer (tensor of shape [batch_size, num_nodes, num_layers, feature_dim])
        e: Edge features for each layer (tensor of shape [batch_size, num_nodes, num_nodes, num_layers, feature_dim])
        Y: Global features common to all layers (tensor of shape [batch_size, feature_dim])
        x: Global features for each layer (tensor of shape [batch_size, num_layers, feature_dim])
        """
        self.X = X
        self.x = x
        self.e = e
        self.Y = Y
        self.y = y
    
    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.x = self.x.type_as(x)
        self.e = self.e.type_as(x)
        self.Y = self.Y.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask):
        X_mask = node_mask.unsqueeze(-1)                 # bs, n, 1
        x_mask = node_mask.unsqueeze(-1).unsqueeze(-1)   # bs, n, 1, 1
        e_mask1 = x_mask.unsqueeze(2)                    # bs, n, 1, 1, 1
        e_mask2 = x_mask.unsqueeze(1)                    # bs, 1, n, 1, 1

        self.X = self.X * X_mask
        self.x = self.x * x_mask
        self.e = self.e * (e_mask1 * e_mask2)
        assert torch.allclose(self.e, torch.transpose(self.e, 1, 2))
        return self


def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'name': cfg.general.name, 'project': f'graph_ddm_{cfg.dataset.name}', 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': cfg.general.wandb}
    wandb.init(**kwargs)
    dataset_config = cfg['dataset']
    if dataset_config.get('baseline') is not None:
        for metric, values in dataset_config.baseline.items():
            for model, value in values.items():
                print("Logging metric: ", f'{metric}_{model}')
                wandb.define_metric(f'{metric}_{model}', step_metric=None)
    wandb.save('*.txt')


def compute_kernel(adj_mat, kernel_coef):
    if kernel_coef is None:
        return None
    else:
        kernel = np.zeros_like(adj_mat)
        for i in range(adj_mat.shape[0]):
            for j in range(adj_mat.shape[3]):
                kernel[i,:,:,j] = kernel_coef * nx.floyd_warshall_numpy(nx.from_numpy_array(adj_mat[i,:,:,j].numpy()))
            
        # print("adjacency matrix: "+str(adj_mat[0,:,:,0]))
        # print("kernel: "+str(kernel[0,:,:,0]))
        return torch.from_numpy(kernel)


class EMA(Callback):
    def __init__(self, decay: float):
        """
        Initializes the EMA object.

        Args:
            decay (float): The decay rate for the exponential moving average. Should be between 0 and 1.
        """
        if not (0 < decay < 1):
            raise ValueError("Decay must be a float between 0 and 1.")
        self.decay = decay
        self.shadow = {}

    def register(self, name: str, value: float):
        """
        Registers a new value to track with EMA.

        Args:
            name (str): The name of the value to track.
            value (float): The initial value.
        """
        self.shadow[name] = value

    def update(self, name: str, value: float):
        """
        Updates the EMA value for a given name.

        Args:
            name (str): The name of the value to update.
            value (float): The new value.
        """
        if name not in self.shadow:
            raise KeyError(f"Value '{name}' is not registered. Use 'register' to add it first.")
        self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * value

    def get(self, name: str) -> float:
        """
        Retrieves the current EMA value for a given name.

        Args:
            name (str): The name of the value to retrieve.

        Returns:
            float: The current EMA value.
        """
        if name not in self.shadow:
            raise KeyError(f"Value '{name}' is not registered.")
        return self.shadow[name]

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        """
        Hook to update EMA values at the end of each training batch.

        Args:
            trainer: The PyTorch Lightning trainer.
            pl_module: The LightningModule being trained.
            outputs: The outputs from the model for the batch.
            batch: The input batch.
            batch_idx: The index of the batch.
            dataloader_idx: The index of the dataloader.
        """
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                if name not in self.shadow:
                    self.register(name, param.data.clone())
                else:
                    self.update(name, param.data.clone())