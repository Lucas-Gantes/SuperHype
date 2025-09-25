import torch
from torch import Tensor
import torch.nn as nn
from torchmetrics import Metric, MeanSquaredError, MetricCollection, MeanMetric
import torch.nn.functional as F
import time
import wandb
import shutil
from src.metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchMSE, SumExceptBatchKL, CrossEntropyMetric, \
    ProbabilityMetric, NLL

eps = 1e-8


class NodeMSE(MeanSquaredError):
    def __init__(self, *args):
        super().__init__(*args)


class EdgeMSE(MeanSquaredError):
    def __init__(self, *args):
        super().__init__(*args)


class TrainLoss(nn.Module):
    def __init__(self):
        super(TrainLoss, self).__init__()
        self.train_node_mse = NodeMSE()
        self.train_edge_mse = EdgeMSE()
        self.train_y_mse = MeanSquaredError()

    def forward(self, masked_pred_epsX, masked_pred_epsE, pred_y, true_epsX, true_epsE, true_y, log: bool):
        mse_X = self.train_node_mse(masked_pred_epsX, true_epsX) if true_epsX.numel() > 0 else 0.0
        mse_E = self.train_edge_mse(masked_pred_epsE, true_epsE) if true_epsE.numel() > 0 else 0.0
        mse_y = self.train_y_mse(pred_y, true_y) if true_y.numel() > 0 else 0.0
        mse = mse_X + mse_E + mse_y

        if log:
            to_log = {'train_loss/batch_mse': mse.detach(),
                      'train_loss/node_MSE': self.train_node_mse.compute(),
                      'train_loss/edge_MSE': self.train_edge_mse.compute(),
                      'train_loss/y_mse': self.train_y_mse.compute()}
            if wandb.run:
                wandb.log(to_log, commit=True)

        return mse

    def reset(self):
        for metric in (self.train_node_mse, self.train_edge_mse, self.train_y_mse):
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_mse = self.train_node_mse.compute() if self.train_node_mse.total > 0 else -1
        epoch_edge_mse = self.train_edge_mse.compute() if self.train_edge_mse.total > 0 else -1
        epoch_y_mse = self.train_y_mse.compute() if self.train_y_mse.total > 0 else -1

        to_log = {"train_epoch/epoch_X_mse": epoch_node_mse,
                  "train_epoch/epoch_E_mse": epoch_edge_mse,
                  "train_epoch/epoch_y_mse": epoch_y_mse}
        if wandb.run:
            wandb.log(to_log)
        return to_log



class TrainLossDiscrete(nn.Module):
    """ Train with Cross entropy"""
    def __init__(self, lambda_train):
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.y_loss = CrossEntropyMetric()
        self.lambda_train = lambda_train

    def forward(self, masked_pred_X, masked_pred_E, pred_y, true_X, true_E, true_y, log: bool):
        """ Compute train metrics
        masked_pred_X : tensor -- (bs, n, dx)
        masked_pred_E : tensor -- (bs, n, n, de)
        pred_y : tensor -- (bs, )
        true_X : tensor -- (bs, n, dx)
        true_E : tensor -- (bs, n, n, de)
        true_y : tensor -- (bs, )
        log : boolean. """
        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n, dx)
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_X = torch.reshape(masked_pred_X, (-1, masked_pred_X.size(-1)))  # (bs * n, dx)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))   # (bs * n * n, de)

        # Remove masked rows
        mask_X = (true_X != 0.).any(dim=-1)
        mask_E = (true_E != 0.).any(dim=-1)

        flat_true_X = true_X[mask_X, :]
        flat_pred_X = masked_pred_X[mask_X, :]

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        loss_X = self.node_loss(flat_pred_X, flat_true_X) if true_X.numel() > 0 else 0.0


        term_width = shutil.get_terminal_size().columns
        torch.set_printoptions(threshold=float('inf'), linewidth=term_width)
        torch.set_printoptions(threshold=float('inf'), linewidth=term_width)

        loss_E = self.edge_loss(flat_pred_E, flat_true_E) if true_E.numel() > 0 else 0.0
        loss_y = self.y_loss(pred_y, true_y) if true_y.numel() > 0 else 0.0

        if log:
            to_log = {"train_loss/batch_CE": (loss_X + loss_E + loss_y).detach(),
                      "train_loss/X_CE": self.node_loss.compute() if true_X.numel() > 0 else -1,
                      "train_loss/E_CE": self.edge_loss.compute() if true_E.numel() > 0 else -1,
                      "train_loss/y_CE": self.y_loss.compute() if true_y.numel() > 0 else -1}
            if wandb.run:
                wandb.log(to_log, commit=True)
        return loss_X + self.lambda_train[0] * loss_E + self.lambda_train[1] * loss_y

    def reset(self):
        for metric in [self.node_loss, self.edge_loss, self.y_loss]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_node_loss = self.node_loss.compute() if self.node_loss.total_samples > 0 else -1
        epoch_edge_loss = self.edge_loss.compute() if self.edge_loss.total_samples > 0 else -1
        epoch_y_loss = self.train_y_loss.compute() if self.y_loss.total_samples > 0 else -1

        to_log = {"train_epoch/x_CE": epoch_node_loss,
                  "train_epoch/E_CE": epoch_edge_loss,
                  "train_epoch/y_CE": epoch_y_loss}
        if wandb.run:
            wandb.log(to_log, commit=False)

        return to_log



class MultiLabelTrainLoss(TrainLossDiscrete):
    """
    Entraînement multilabel pour les arêtes :
    - Hérite de TrainLossDiscrete pour X et y (CrossEntropy)
    - BCEWithLogitsLoss sur chaque bit d'arête
    """
    def __init__(self, lambda_train, weight_BCE=None):
        super().__init__(lambda_train)
        self.edge_loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight_BCE) if weight_BCE is not None else nn.BCEWithLogitsLoss()
        self.edge_bce_metric = MeanMetric()

        self.node_label_mse_fn = nn.MSELoss()

        self.sigmoid = nn.Sigmoid()

    def forward(self, masked_pred_X, masked_pred_E, pred_y,
                true_X, true_E, true_y, log: bool, dataset_node_labels=None):
        
        if dataset_node_labels is not None:
            bs, n, dx = masked_pred_X.shape
            _, nb_nodes, nb_features = dataset_node_labels.shape
            assert n == nb_nodes
            assert dx >= nb_features
            predicted_node_labels = masked_pred_X[:, :, (dx-nb_features):]
            masked_pred_X = masked_pred_X[:, :, :(dx-nb_features)]

        bs, n, dx = masked_pred_X.shape
        flat_true_X = true_X.reshape(-1, dx)
        flat_pred_X = masked_pred_X.reshape(-1, dx)
        mask_X = (flat_true_X != 0.).any(dim=-1)
        flat_true_X = flat_true_X[mask_X]
        flat_pred_X = flat_pred_X[mask_X]
        loss_X = self.node_loss(flat_pred_X, flat_true_X)

        bs, n, _, e_bits = true_E.shape
        flat_true_E = true_E.view(bs * n * n, e_bits).float()
        flat_pred_E = masked_pred_E.view(bs * n * n, e_bits)

        diag_mask = torch.eye(n, device=flat_true_E.device, dtype=torch.bool).view(-1)
        valid = (~diag_mask).repeat(bs)
        flat_true_E = flat_true_E[valid]
        flat_pred_E = flat_pred_E[valid]

        flat_true_E = flat_true_E.view(flat_true_E.size(0) * e_bits, 1).float()
        flat_pred_E = flat_pred_E.view(flat_pred_E.size(0) * e_bits, 1)
        
        loss_E = self.edge_loss_fn(flat_pred_E, flat_true_E)
        self.edge_bce_metric.update(loss_E.detach())
        self._edge_bce_updated = True

        loss_y = self.y_loss(pred_y, true_y) if true_y.numel() > 0 else 0.0

        if dataset_node_labels is not None:
            assert dataset_node_labels.shape == predicted_node_labels.shape
            node_label_mse = self.node_label_mse_fn(
                predicted_node_labels.float(),
                dataset_node_labels.float()
            )
            print("Node label MSE: ", node_label_mse.item())
            node_label_var = torch.var(predicted_node_labels, dim=0).mean()
            print("Node label variance: ", node_label_var.item())
            node_label_var_loss = (node_label_var - 0.61) ** 2
            print("Node label variance loss: ", node_label_var_loss.item())

            total = (
                loss_X
                + self.lambda_train[0] * loss_E
                + self.lambda_train[1] * loss_y
                + self.lambda_train[2] * node_label_mse
                + self.lambda_train[3] *  node_label_var_loss
            )
        else:
            total = (
                loss_X
                + self.lambda_train[0] * loss_E
                + self.lambda_train[1] * loss_y
            )
        
        # Logging
        if log and hasattr(self, 'log'):
            self.log('train_loss/batch_X_CE', loss_X.detach())
            self.log('train_loss/batch_E_BCE', loss_E.detach())
            self.log('train_loss/batch_y_CE', loss_y.detach())
            if dataset_node_labels is not None:
                self.log('train_loss/batch_node_label_MSE', node_label_mse.detach())
                self.log('train_loss/batch_node_label_var', node_label_var.detach())

        return total

    def log_epoch_metrics(self):
        epoch_node_loss = self.node_loss.compute() if self.node_loss.total_samples > 0 else -1
        epoch_edge_loss = (self.edge_bce_metric.compute() if self._edge_bce_updated else -1)
        epoch_y_loss   = self.y_loss.compute()     if self.y_loss.total_samples  > 0 else -1

        to_log = {"train_epoch/x_CE":  epoch_node_loss,
                "train_epoch/E_CE": epoch_edge_loss,
                "train_epoch/y_CE":  epoch_y_loss}
        if wandb.run:
            wandb.log(to_log, commit=False)
        return to_log

    def reset(self):
        super().reset()
        self.edge_bce_metric.reset()
        self._edge_bce_updated = False


class CliqueTrainLoss(nn.Module):
    def __init__(self, clique_loss_coef):
        super(CliqueTrainLoss, self).__init__()
        self.loss3cliques = nn.MSELoss()
        self.loss4cliques = nn.MSELoss()
        self.loss5cliques = nn.MSELoss()
        self.clique_loss_coef = clique_loss_coef

        self.sigmoid = nn.Sigmoid()

        self.tri_loss = 0.
        self.quad_loss = 0.

    def forward(self, pred_adj_logits, true_adj, log: bool):
        pred_adj = self.sigmoid(pred_adj_logits)
        
        tri_pred = torch.einsum('bikd,bkjd,bijd->bijkd', pred_adj, pred_adj, pred_adj)
        tri_true = torch.einsum('bikd,bkjd,bijd->bijkd', true_adj, true_adj, true_adj)
        tri_true = (tri_true > 0.1).float()
        
        quad_pred = torch.einsum('bijd,bikd,bild,bjkd,bjld,bkld->bijkld', pred_adj, pred_adj, pred_adj, pred_adj, pred_adj, pred_adj)
        quad_true = torch.einsum('bijd,bikd,bild,bjkd,bjld,bkld->bijkld', true_adj, true_adj, true_adj, true_adj, true_adj, true_adj)
        quad_true = (quad_true > 0.1).float()
        
        den_tri = torch.clamp(tri_true.sum(), min=1e-8)
        tri_loss = F.mse_loss(tri_pred.sum().unsqueeze(0), den_tri.unsqueeze(0)) * self.clique_loss_coef[0] / (den_tri ** 2)

        den_quad = torch.clamp(quad_true.sum(), min=1e-8)
        quad_loss = F.mse_loss(quad_pred.sum().unsqueeze(0), den_quad.unsqueeze(0)) * self.clique_loss_coef[1] / (den_quad ** 2)

        self.tri_loss += tri_loss.detach()
        self.quad_loss += quad_loss.detach()

        return tri_loss + quad_loss
    
    def log_epoch_metrics(self):
        to_log = {"train_epoch/3-cliques_loss": self.tri_loss,
                  "train_epoch/4-cliques_loss": self.quad_loss}
        if wandb.run:
            wandb.log(to_log, commit=False)
        return to_log
    
    def reset(self):
        self.tri_loss = 0.
        self.quad_loss = 0.
