import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import wandb
import os
from tqdm.auto import tqdm
import shutil
import sys
from torch_ema import ExponentialMovingAverage

from src.models.transformer_model import GraphTransformer
from src.models.multi_layer_transformer import MultiLayerGraphTransformer, GraphSuperpositionTransformer
from src.diffusion.noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete,\
    MarginalUniformTransition, MultiHotUniformTransition
from src.diffusion import diffusion_utils
from src.metrics.train_metrics import TrainLossDiscrete, MultiLabelTrainLoss, CliqueTrainLoss
from src.metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL
from src import utils


class DiscreteDenoisingDiffusion(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, extra_features,
                 domain_features, multifactor=False, num_node_labels=None, clique_loss_coef=None, 
                 model_type='XEyTransformer', kernel_coef=None, triplet_interactions=False, parallel_model=False, ema_decay=None,
                 single_layer=False):
        super().__init__()
        print("triplet_interactions: "+str(triplet_interactions))

        input_dims = dataset_infos.input_dims.copy()
        output_dims = dataset_infos.output_dims.copy()
        nodes_dist = dataset_infos.nodes_dist

        self.lr = cfg.train.lr

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps

        self.num_node_labels = num_node_labels

        self.Xdim = input_dims['X']
        self.xdim = input_dims['x']
        self.edim = input_dims['e']  # = nb_layers (=/= num_edge_types)
        self.Ydim = input_dims['Y']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.xdim_output = output_dims['x']
        self.edim_output = output_dims['e']
        self.num_layers = input_dims['nb_layers']
        self.Ydim_output = output_dims['Y']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        if multifactor:
            if cfg.train.weight_BCE:
                print("Using MultiLabelTrainLoss with weighted BCE")
                w = cfg.train.weight_BCE
                print("weight for BCE: ", w)
                self.train_loss = MultiLabelTrainLoss(self.cfg.model.lambda_train, weight_BCE=torch.tensor(w))
            else:
                print("Using MultiLabelTrainLoss without weighted BCE")
                self.train_loss = MultiLabelTrainLoss(self.cfg.model.lambda_train)
        else:
            self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train)
        
        self.is_clique_loss = clique_loss_coef is not None

        if clique_loss_coef is not None:
            self.clique_loss = CliqueTrainLoss(clique_loss_coef, clip_values=cfg.model.clip_values)


        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()
        self.val_node_labels = nn.MSELoss() if num_node_labels is not None else None

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        # self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.kernel_coef = kernel_coef

        if multifactor:
            if model_type == 'XEyTransformer':
                print("Former input_dims: ", input_dims)
                input_dims = {
                    'X': input_dims['X'] + input_dims['x']*input_dims['nb_layers'],
                    'E': input_dims['nb_layers'],
                    'y': input_dims['Y'] + input_dims['y']*input_dims['nb_layers'],
                }
                output_dims = {
                    'X': output_dims['X'] + output_dims['x']*output_dims['nb_layers'],
                    'E': output_dims['nb_layers'],
                    'y': output_dims['Y'] + output_dims['y']*output_dims['nb_layers'],
                }
                print("Before the creation of the model, input_dims: ", input_dims)
                self.model = GraphTransformer(n_layers=cfg.model.n_layers,
                                    input_dims=input_dims,
                                    hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                    hidden_dims=cfg.model.hidden_dims,
                                    output_dims=output_dims,
                                    act_fn_in=nn.ReLU(),
                                    act_fn_out=nn.ReLU(),
                                    nb_labels=num_node_labels)
                self.model_type = model_type
            elif model_type == 'MultiLayerXEyTransformer':
                print("Former input_dims: ", input_dims)
                input_dims = {
                    'x': input_dims['x'] + input_dims['X'],
                    'e': input_dims['e'],
                    'y': input_dims['y'] + input_dims['Y'],
                }
                output_dims = {
                    'x': output_dims['x'] + output_dims['X'],
                    'e': output_dims['e'],
                    'y': output_dims['y'] + output_dims['Y'],
                }
                print("Before the creation of the model, input_dims: ", input_dims)
                self.model = MultiLayerGraphTransformer(n_layers=cfg.model.n_layers,
                                    input_dims=input_dims,
                                    hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                    hidden_dims=cfg.model.hidden_dims,
                                    output_dims=output_dims,
                                    act_fn_in=nn.ReLU(),
                                    act_fn_out=nn.ReLU(),
                                    nb_labels=num_node_labels)
                self.model_type = model_type
            elif model_type == "GraphSuperpositionTransformerv1" or model_type == "GraphSuperpositionTransformerv2":
                if model_type == "GraphSuperpositionTransformerv1":
                    self.cross_attention = False
                else:
                    self.cross_attention = True
                input_dims['x'] = input_dims['x'] + input_dims['X']
                input_dims['y'] = input_dims['y'] + input_dims['Y']
                self.model = GraphSuperpositionTransformer(n_layers=cfg.model.n_layers,
                                    input_dims=input_dims,
                                    hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                    hidden_dims=cfg.model.hidden_dims,
                                    output_dims=output_dims,
                                    act_fn_in=nn.ReLU(),
                                    act_fn_out=nn.ReLU(),
                                    cross_attention=self.cross_attention,
                                    triplet_interactions=triplet_interactions,
                                    parallel=parallel_model,
                                    single_layer=single_layer)
                self.model_type = "GraphSuperpositionTransformer"

            else:
                raise ValueError(f"Unknown model type: {model_type}. Supported types are 'XEyTransformer' and 'MultiLayerXEyTransformer'.")
        else:
            self.model = GraphTransformer(n_layers=cfg.model.n_layers,
                                    input_dims=input_dims,
                                    hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                    hidden_dims=cfg.model.hidden_dims,
                                    output_dims=output_dims,
                                    act_fn_in=nn.ReLU(),
                                    act_fn_out=nn.ReLU(),
                                    nb_labels=num_node_labels)
        
        if ema_decay is not None:
            self.use_ema = True
            self.ema_module = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.use_ema = False

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps)

        if self.num_node_labels is not None:
            self.x_diff_dim = self.Xdim_output - self.num_node_labels
        else:
            self.x_diff_dim = self.Xdim_output
    
        if multifactor:
            # print("MultiHotUniformTransition created")
            self.transition_model = MultiHotUniformTransition(x_classes=self.x_diff_dim, e_bits=self.num_layers,
                                                              y_classes=self.Ydim_output)
            
            x_limit = torch.ones(self.x_diff_dim) / self.x_diff_dim
            e_limit = torch.ones(self.num_layers) / 2.
            # print(f"e_limit={e_limit}")
            y_limit = torch.ones(self.Ydim_output) / self.Ydim_output
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)
        elif cfg.model.transition == 'uniform':
            self.transition_model = DiscreteUniformTransition(x_classes=self.x_diff_dim, e_classes=self.edim_output,
                                                              y_classes=self.Ydim_output)
            x_limit = torch.ones(self.x_diff_dim) / self.x_diff_dim
            e_limit = torch.ones(self.edim_output) / self.edim_output
            y_limit = torch.ones(self.Ydim_output) / self.Ydim_output
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)
        elif cfg.model.transition == 'marginal':
            node_types = self.dataset_info.node_types.float()
            x_marginals = node_types / torch.sum(node_types)

            edge_types = self.dataset_info.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)
            # print(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")
            self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                              y_classes=self.Ydim_output)
            self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                                y=torch.ones(self.Ydim_output) / self.Ydim_output)
        

        self.save_hyperparameters(ignore=['train_metrics', 'sampling_metrics'])
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0

        self.multifactor = multifactor
    
    def extract_node_labels(self, X):
        """ Extracts node labels from the input tensor X. """
        return X[:, :, self.x_diff_dim:], X[:, :, :self.x_diff_dim]

    def training_step(self, data, i):
        if data.edge_index.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return
        
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, 
                                               data.batch, multilabel=self.multifactor)
        dense_data = dense_data.mask(node_mask)

        X, E = dense_data.X, dense_data.E

        if self.num_node_labels is not None:
            # Extract node labels from the input tensor X
            node_labels, X = self.extract_node_labels(X)
        else:
            node_labels = None

        noisy_data = self.apply_noise(X, E, data.y, node_mask)

        extra_data = self.compute_extra_data(noisy_data)

        pred = self.forward(noisy_data, extra_data, node_mask)        

        if self.is_clique_loss:
            clique_loss = self.clique_loss(
                pred.E,
                E,
                log=(i % self.log_every_steps == 0)
                )
        else: clique_loss = 0.0
        

        graph_loss = self.train_loss(
            pred.X,     # logits node
            pred.E,     # logits edges (bs,n,n,e_bits)
            pred.y,     # logits global
            X,          # one-hot true_X
            E,          # one-hot true_E ou multi-hot selon multifactor
            data.y,     # true_y
            log=(i % self.log_every_steps == 0),
            dataset_node_labels=node_labels
            )
        

        self.train_metrics(masked_pred_X=pred.X, masked_pred_E=pred.E, true_X=X, true_E=E,
                           log=i % self.log_every_steps == 0)

        return {'loss': graph_loss + clique_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, amsgrad=True,
                                 weight_decay=self.cfg.train.weight_decay)
        if self.cfg.train.scheduler == "OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0005, total_steps=20000, 
                                                    anneal_strategy='cos', cycle_momentum=False, div_factor=10)
        elif self.cfg.train.scheduler == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.train.n_epochs, eta_min=1e-6, interval='epoch')
        elif self.cfg.train.scheduler == "ExponentialLR":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
        else: 
            return optimizer
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        self.print("Size of the input features", self.Xdim, self.edim, self.Ydim)
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)
        
    def on_train_start(self):
        if self.use_ema:
            self.ema_module.to(self.device)
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.use_ema:
            self.ema_module.update()

    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

        if self.is_clique_loss:
            self.clique_loss.reset()

    def on_train_epoch_end(self) -> None:
        to_log = self.train_loss.log_epoch_metrics()
        opt = self.trainer.optimizers[0]
        current_lr = opt.param_groups[0]['lr']
        # self.log('lr', current_lr, prog_bar=True, logger=True)
        if self.is_clique_loss:
            clique_logs = self.clique_loss.log_epoch_metrics()
            to_log.update(clique_logs)
        to_log['lr'] = current_lr
        self.print(f"Epoch {self.current_epoch}: X_CE: {to_log['train_epoch/x_CE'] :.3f}"
                      f" -- E_CE: {to_log['train_epoch/E_CE'] :.3f} --"
                      f" y_CE: {to_log['train_epoch/y_CE'] :.3f}"
                      f" -- 3-clique loss: {to_log['train_epoch/3-cliques_loss'] if 'train_epoch/3-cliques_loss'in to_log else -1:.3f}"
                      f" -- 4-clique loss: {to_log['train_epoch/4-cliques_loss'] if 'train_epoch/4-cliques_loss'in to_log else -1:.3f}"
                      f" -- learning rate: {to_log['lr']}"
                      f" -- {time.time() - self.start_epoch_time:.1f}s ")
        epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: {epoch_at_metrics} -- {epoch_bond_metrics}")
        if self.cfg.train.display_memory_summary:
            if torch.cuda.is_available():
                print(torch.cuda.memory_summary())
            else:
                print("CUDA is not available. Skipping memory summary.")

    def on_validation_epoch_start(self) -> None:
        if self.use_ema:          
            self.ema_module.store()
            self.ema_module.copy_to() 
            print("EMA weights loaded")
        self.val_nll.reset()
        self.val_X_kl.reset()
        self.val_E_kl.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()
        self.sampling_metrics.reset()

    def validation_step(self, data, i):

        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
              

        if self.num_node_labels is not None:
            # Extract node labels from the input tensor X
            node_labels, X = self.extract_node_labels(X)
        else:
            node_labels = None

        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y,  node_mask, test=False)
        return {'loss': nll}

    def on_validation_epoch_end(self) -> None:
        metrics = [self.val_nll.compute(), self.val_X_kl.compute() * self.T, self.val_E_kl.compute() * self.T,
                   self.val_X_logp.compute(), self.val_E_logp.compute()]
        if wandb.run:
            wandb.log({"val/epoch_NLL": metrics[0],
                       "val/X_kl": metrics[1],
                       "val/E_kl": metrics[2],
                       "val/X_logp": metrics[3],
                       "val/E_logp": metrics[4]}, commit=True)

        self.print(f"Epoch {self.current_epoch}: Val NLL {metrics[0] :.2f} -- Val Atom type KL {metrics[1] :.2f} -- ",
                   f"Val Edge type KL: {metrics[2] :.2f}")

        # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        val_nll = metrics[0]
        self.log("val/epoch_NLL", val_nll, sync_dist=True)

        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
        self.print('Val loss: %.4f \t Best val loss:  %.4f\n' % (val_nll, self.best_val_nll))

        self.val_counter += 1
        if self.val_counter % self.cfg.general.sample_every_val == 0:
            start = time.time()
            samples_left_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_save = self.cfg.general.samples_to_save
            chains_left_to_save = self.cfg.general.chains_to_save

            samples = []

            ident = 0
            while samples_left_to_generate > 0:
                bs = 2 * self.cfg.train.batch_size
                to_generate = min(samples_left_to_generate, bs)
                to_save = min(samples_left_to_save, bs)
                chains_save = min(chains_left_to_save, bs)
                samples.extend(self.sample_batch(batch_id=ident, batch_size=to_generate, num_nodes=None,
                                                 save_final=to_save,
                                                 keep_chain=chains_save,
                                                 number_chain_steps=self.number_chain_steps))
                ident += to_generate

                samples_left_to_save -= to_save
                samples_left_to_generate -= to_generate
                chains_left_to_save -= chains_save
            self.print("Computing sampling metrics...")
            to_log = self.sampling_metrics.forward(samples, self.name, self.current_epoch, val_counter=self.val_counter, test=False,
                                          local_rank=self.local_rank)
            
            for metric_name in to_log.keys():
                self.log(metric_name, to_log[metric_name], sync_dist=True)
            
            self.print(f'Done. Sampling took {time.time() - start:.2f} seconds\n')
            print("Validation epoch end ends...")
        if self.use_ema:    
            self.ema_module.restore()
            print("Training weights restored")

    def on_test_epoch_start(self) -> None:
        self.print("Starting test...")
        if self.use_ema:          
            self.ema_module.store()
            self.ema_module.copy_to()  
            print("EMA weights loaded")
        self.test_nll.reset()
        self.test_X_kl.reset()
        self.test_E_kl.reset()
        self.test_X_logp.reset()
        self.test_E_logp.reset()
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def test_step(self, data, i):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y, node_mask, test=True)
        return {'loss': nll}

    def on_test_epoch_end(self) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        metrics = [self.test_nll.compute(), self.test_X_kl.compute(), self.test_E_kl.compute(),
                   self.test_X_logp.compute(), self.test_E_logp.compute()]
        if wandb.run:
            wandb.log({"test/epoch_NLL": metrics[0],
                       "test/X_kl": metrics[1],
                       "test/E_kl": metrics[2],
                       "test/X_logp": metrics[3],
                       "test/E_logp": metrics[4]}, commit=True)

        self.print(f"Epoch {self.current_epoch}: Test NLL {metrics[0] :.2f} -- Test Atom type KL {metrics[1] :.2f} -- ",
                   f"Test Edge type KL: {metrics[2] :.2f}")

        test_nll = metrics[0]
        if wandb.run:
            wandb.log({"test/epoch_NLL": test_nll}, commit=True)

        self.print(f'Test loss: {test_nll :.4f}')

        samples_left_to_generate = self.cfg.general.final_model_samples_to_generate
        samples_left_to_save = self.cfg.general.final_model_samples_to_save
        chains_left_to_save = self.cfg.general.final_model_chains_to_save

        samples = []
        id = 0
        while samples_left_to_generate > 0:
            self.print(f'Samples left to generate: {samples_left_to_generate}/'
                       f'{self.cfg.general.final_model_samples_to_generate}', end='', flush=True)
            bs = 2 * self.cfg.train.batch_size
            to_generate = min(samples_left_to_generate, bs)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            samples.extend(self.sample_batch(id, to_generate, num_nodes=None, save_final=to_save,
                                             keep_chain=chains_save, number_chain_steps=self.number_chain_steps))
            id += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save
        self.print("Saving the generated graphs")
        filename = f'generated_samples1.txt'
        for i in range(2, 10):
            if os.path.exists(filename):
                filename = f'generated_samples{i}.txt'
            else:
                break
        with open(filename, 'w') as f:
            for item in samples:
                f.write(f"N={item[0].shape[0]}\n")
                atoms = item[0].tolist()
                f.write("X: \n")
                for at in atoms:
                    f.write(f"{at} ")
                f.write("\n")
                f.write("E: \n")
                for bond_list in item[1]:
                    for bond in bond_list:
                        f.write(f"{bond} ")
                    f.write("\n")
                f.write("\n")
        self.print("Generated graphs Saved. Computing sampling metrics...")
        self.sampling_metrics(samples, self.name, self.current_epoch, self.val_counter, test=True, local_rank=self.local_rank)
        if self.use_ema:  
            self.ema_module.restore()
            print("Training weights restored")
        self.print("Done testing.")

    def kl_prior(self, X, E, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        if self.multifactor:
            return self.kl_prior_multifactor(X, E, node_mask)
        else:
            return self.kl_prior_one_hot(X, E, node_mask)

    def kl_prior_one_hot(self, X, E, node_mask):
        """
        kl_prior for one-hot model
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)       

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)
        assert probX.shape == X.shape

        bs, n, _ = probX.shape

        limit_X = self.limit_dist.X[None, None, :].expand(bs, n, -1).type_as(probX)
        limit_E = self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)

        # Make sure that masked rows do not contribute to the loss
        limit_dist_X, limit_dist_E, probX, probE = diffusion_utils.mask_distributions(true_X=limit_X.clone(),
                                                                                      true_E=limit_E.clone(),
                                                                                      pred_X=probX,
                                                                                      pred_E=probE,
                                                                                      node_mask=node_mask)

        kl_distance_X = F.kl_div(input=probX.log(), target=limit_dist_X, reduction='none')
        kl_distance_E = F.kl_div(input=probE.log(), target=limit_dist_E, reduction='none')

        return diffusion_utils.sum_except_batch(kl_distance_X) + \
               diffusion_utils.sum_except_batch(kl_distance_E)
    
    def kl_prior_multifactor(self, X, E, node_mask):
        """
        kl_prior for multifactor model
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        
        E_tr = Qtb.E.unsqueeze(1).unsqueeze(1)  # (bs, 1, 1, e_bits, 2,2)

        E_one_hot = F.one_hot(E.long(), num_classes=2).unsqueeze(-2).float()  # (bs, n, n, e_bits, 2, 1)

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE_one_hot = E_one_hot @ E_tr  # (bs, n, n, e_bits, 2, 1)
        probE = probE_one_hot[:,:,:,:,0,1]  # (bs, n, n, e_bits)
        assert probX.shape == X.shape

        bs, n, _ = probX.shape

        limit_X = self.limit_dist.X[None, None, :].expand(bs, n, -1).type_as(probX)
        limit_E = self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)

        # Make sure that masked rows do not contribute to the loss
        limit_dist_X, limit_dist_E, probX, probE = diffusion_utils.mask_distributions(true_X=limit_X.clone(),
                                                                                      true_E=limit_E.clone(),
                                                                                      pred_X=probX,
                                                                                      pred_E=probE,
                                                                                      node_mask=node_mask)

        kl_distance_X = F.kl_div(input=probX.log(), target=limit_dist_X, reduction='none')
        kl_distance_E = F.kl_div(input=probE.log(), target=limit_dist_E, reduction='none')

        return diffusion_utils.sum_except_batch(kl_distance_X) + \
               diffusion_utils.sum_except_batch(kl_distance_E)

    def compute_Lt(self, X, E, y, pred, noisy_data, node_mask, test):       
        pred_probs_X = F.softmax(pred.X, dim=-1)
        if self.multifactor:
            pred_probs_E = F.sigmoid(pred.E)
        else:
            pred_probs_E = F.softmax(pred.E, dim=-1)
        pred_probs_y = F.softmax(pred.y, dim=-1)

        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], self.device)
        Qt = self.transition_model.get_Qt(noisy_data['beta_t'], self.device)

        if self.multifactor:
            # Compute distributions to compare with KL
            bs, n, d = X.shape
            E_list = diffusion_utils.split_edge_label_tensor(E)
            E_t_list = diffusion_utils.split_edge_label_tensor(noisy_data['E_t'])
            Qsb_E_list = diffusion_utils.split_edge_transition_tensor(Qsb.E)
            Qt_E_list = diffusion_utils.split_edge_transition_tensor(Qt.E)
            Qtb_E_list = diffusion_utils.split_edge_transition_tensor(Qtb.E)
            pred_probs_E_list = diffusion_utils.split_edge_label_tensor(pred_probs_E)
            
            assert len(E_list) == len(E_t_list)
            assert len(E_t_list) == len(Qsb_E_list)
            assert len(Qsb_E_list) == len(Qt_E_list)
            assert len(Qt_E_list) == len(Qtb_E_list)

            prob_true_X = diffusion_utils.compute_posterior_distribution(M=X, M_t=noisy_data['X_t'],
                                                                   Qt_M=Qt.X, Qsb_M=Qsb.X, Qtb_M=Qtb.X)
            prob_pred_X = diffusion_utils.compute_posterior_distribution(M=pred_probs_X, M_t=noisy_data['X_t'],
                                                                   Qt_M=Qt.X, Qsb_M=Qsb.X, Qtb_M=Qtb.X)
            prob_true_X, prob_pred_X = diffusion_utils.mask_distributions_X(prob_true_X, prob_pred_X, node_mask)
            kl_x = (self.test_X_kl if test else self.val_X_kl)(prob_true_X, torch.log(prob_pred_X))

            kl_e = torch.tensor(0.).to(kl_x.device)

            for i in range(len(E_list)):
                prob_true_E = diffusion_utils.compute_posterior_distribution(M=E_list[i], M_t=E_t_list[i],
                                                                   Qt_M=Qt_E_list[i], Qsb_M=Qsb_E_list[i], Qtb_M=Qtb_E_list[i])
                prob_pred_E = diffusion_utils.compute_posterior_distribution(M=pred_probs_E_list[i], M_t=E_t_list[i],
                                                                   Qt_M=Qt_E_list[i], Qsb_M=Qsb_E_list[i], Qtb_M=Qtb_E_list[i])
                
                prob_true_E = prob_true_E.reshape((bs, n, n, -1))
                prob_pred_E = prob_pred_E.reshape((bs, n, n, -1))
                prob_true_E, prob_pred_E = diffusion_utils.mask_distributions_E(prob_true_E, prob_pred_E, node_mask)
                kl_e += (self.test_E_kl if test else self.val_E_kl)(prob_true_E, torch.log(prob_pred_E))
            

        else:
            # Compute distributions to compare with KL
            bs, n, d = X.shape
            prob_true = diffusion_utils.posterior_distributions(X=X, E=E, y=y, X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                                y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
            prob_true.E = prob_true.E.reshape((bs, n, n, -1))

            prob_pred = diffusion_utils.posterior_distributions(X=pred_probs_X, E=pred_probs_E, y=pred_probs_y,
                                                                X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                                y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
            prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))

            # Reshape and filter masked rows
            prob_true_X, prob_true_E, prob_pred.X, prob_pred.E = diffusion_utils.mask_distributions(true_X=prob_true.X,
                                                                                                    true_E=prob_true.E,
                                                                                                    pred_X=prob_pred.X,
                                                                                                    pred_E=prob_pred.E,
                                                                                                    node_mask=node_mask)
            kl_x = (self.test_X_kl if test else self.val_X_kl)(prob_true.X, torch.log(prob_pred.X))
            kl_e = (self.test_E_kl if test else self.val_E_kl)(prob_true.E, torch.log(prob_pred.E))
        
        return self.T * (kl_x + kl_e)

    def reconstruction_logp(self, t, X, E, node_mask):
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)

        probX0 = X @ Q0.X  # (bs, n, dx_out)

        if self.multifactor:
            probE0 = diffusion_utils.apply_multilabel_transition(E, Q0.E)
            sampled0 = diffusion_utils.sample_multicat_features(probX=probX0, probE=probE0, node_mask=node_mask)
            X0 = F.one_hot(sampled0.X, num_classes=self.x_diff_dim).float()
            # One hot encoding is not necessary for E because it is still binary encoded
            E0 = sampled0.E
            y0 = sampled0.y
            assert (X.shape == X0.shape) and (E.shape == E0.shape)

        else:
            probE0 = E @ Q0.E.unsqueeze(1)  # (bs, n, n, de_out)
            sampled0 = diffusion_utils.sample_discrete_features(probX=probX0, probE=probE0, node_mask=node_mask)
            X0 = F.one_hot(sampled0.X, num_classes=self.x_diff_dim).float()
            E0 = F.one_hot(sampled0.E, num_classes=self.edim_output).float()
            y0 = sampled0.y
            assert (X.shape == X0.shape) and (E.shape == E0.shape)

        sampled_0 = utils.PlaceHolder(X=X0, E=E0, y=y0).mask(node_mask)

        # Predictions
        noisy_data = {'X_t': sampled_0.X, 'E_t': sampled_0.E, 'y_t': sampled_0.y, 'node_mask': node_mask,
                      't': torch.zeros(X0.shape[0], 1).type_as(y0)}
        extra_data = self.compute_extra_data(noisy_data)
        pred0 = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        probX0 = F.softmax(pred0.X, dim=-1)
        if self.multifactor:
            probE0 = F.sigmoid(pred0.E)
        else:
            probE0 = F.softmax(pred0.E, dim=-1)
        proby0 = F.softmax(pred0.y, dim=-1)

        # Set masked rows to arbitrary values that don't contribute to loss
        probX0[~node_mask] = torch.ones(self.x_diff_dim).type_as(probX0)
        probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = torch.ones(self.edim_output).type_as(probE0)

        diag_mask = torch.eye(probE0.size(1)).type_as(probE0).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(probE0.size(0), -1, -1)
        probE0[diag_mask] = torch.ones(self.edim_output).type_as(probE0)

        return utils.PlaceHolder(X=probX0, E=probE0, y=proby0)
    
    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        if self.multifactor:
            return self.apply_noise_multifactor(X, E, y, node_mask)
        else:
            return self.apply_noise_one_hot(X, E, y, node_mask)
    
    def apply_noise_multifactor(self, X, E, y, node_mask):
        """ For multifactor model, sample noise and apply it to the data. """
        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=-1) - 1.) < 1e-4).all(), Qtb.E.sum(dim=-1) - 1

        E_tr = Qtb.E.unsqueeze(1).unsqueeze(1)  # (bs, 1, 1, e_bits, 2,2)

        # Convert E to a one-hot representation for each graph
        E_one_hot = F.one_hot(E.long(), num_classes=2).unsqueeze(-2).float()  # (bs, n, n, e_bits, 2, 1)

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE_one_hot = E_one_hot @ E_tr  # (bs, n, n, e_bits, 2, 1)
        probE = probE_one_hot[:,:,:,:,0,1]  # (bs, n, n, e_bits)

        sampled_t = diffusion_utils.sample_multicat_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.x_diff_dim)
        E_t = sampled_t.E
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        # For the moment, no mask is applied in multifactor mode
        # z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)
        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data

    def apply_noise_one_hot(self, X, E, y, node_mask):
        """ For one-hot model, sample noise and apply it to the data. """

        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.x_diff_dim)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Eeim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data

    def compute_val_loss(self, pred, noisy_data, X, E, y, node_mask, test=False):
        """Computes an estimator for the variational lower bound.
           pred: (batch_size, n, total_features)
           noisy_data: dict
           X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
           node_mask : (bs, n)
           Output: nll (size 1)
       """
        t = noisy_data['t']
        
        to_log = {}

        # 1.
        N = node_mask.sum(1).long()
        log_pN = self.node_dist.log_prob(N)
        to_log["log_pn"] = log_pN.mean()

        if self.num_node_labels is not None:
            dataset_node_labels, X = self.extract_node_labels(X)
            pred_node_labels, pred.X = self.extract_node_labels(pred.X)

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        kl_prior = self.kl_prior(X, E, node_mask)
        to_log["kl prior"] = kl_prior.mean()

        # 3. Diffusion loss
        loss_all_t = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test)
        to_log["Estimator loss terms"] = loss_all_t.mean()

        # 4. Reconstruction loss
        # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        prob0 = self.reconstruction_logp(t, X, E, node_mask)

        loss_term_0 = self.val_X_logp(X * prob0.X.log()) + self.val_E_logp(E * prob0.E.log())
        to_log["loss_term_0"] = loss_term_0.mean()
        # print("loss_term_0: "+str(loss_term_0))

        # If node labels were extracted, add the loss for them
        if self.num_node_labels is not None:
            node_labels_loss = self.val_node_labels(pred_node_labels, dataset_node_labels)
            to_log["node_labels_loss"] = node_labels_loss.mean()

        # Combine terms
        nlls = - log_pN + kl_prior + loss_all_t - loss_term_0
        if self.num_node_labels is not None:
            nlls += node_labels_loss

        assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'

        nll = (self.test_nll if test else self.val_nll)(nlls)        # Average over the batch
        if test:
            to_log["batch_test_nll"] = nll
        else:
            to_log["val_nll"] = nll
        # Update NLL metric object and return batch nll
        

        if wandb.run:
            wandb.log({"kl prior": kl_prior.mean(),
                       "Estimator loss terms": loss_all_t.mean(),
                       "log_pn": log_pN.mean(),
                       "loss_term_0": loss_term_0,
                       'batch_test_nll' if test else 'val_nll': nll}, commit=True)
        return nll

    def forward(self, noisy_data, extra_data, node_mask):
        kernel = utils.compute_kernel(noisy_data['E_t'].cpu(), self.kernel_coef)
        kernel = None if kernel is None else kernel.to(noisy_data['E_t'].device)

        if self.model_type == 'XEyTransformer':
            extra_data_X = torch.cat((extra_data.X, extra_data.x.reshape(extra_data.x.shape[:-2]+(-1,))), dim=2).float()
            X = torch.cat((noisy_data['X_t'], extra_data_X), dim=2).float()
            E = torch.cat((noisy_data['E_t'], extra_data.e), dim=3).float()
            extra_data_y = torch.cat((extra_data.Y, extra_data.y.reshape(extra_data.y.shape[:-2]+(-1,))), dim=-1)
            y = torch.hstack((noisy_data['y_t'], extra_data_y)).float()
            output = self.model(X, E, y, node_mask)

        elif self.model_type == 'MultiLayerXEyTransformer':
            X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
            X_tiled = X.unsqueeze(2).expand(-1, -1, extra_data.x.size(2), -1)
            x = torch.cat([X_tiled, extra_data.x], dim=-1).float()
            e = torch.cat((noisy_data['E_t'].unsqueeze(-1), extra_data.e), dim=-1).float()
            Y = torch.hstack((noisy_data['y_t'], extra_data.Y)).float()
            Y_tiled = Y.unsqueeze(1).expand(-1, extra_data.y.size(1), -1)
            y = torch.cat([Y_tiled, extra_data.y], dim=-1).float()
            

            output = self.model(x, e, y, node_mask, kernel)

        elif self.model_type == "GraphSuperpositionTransformer":
            X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
            X_tiled = X.unsqueeze(2).expand(-1, -1, extra_data.x.size(2), -1)
            x = torch.cat([X_tiled, extra_data.x], dim=-1).float()
            e = torch.cat((noisy_data['E_t'].unsqueeze(-1), extra_data.e), dim=-1).float()
            Y = torch.hstack((noisy_data['y_t'], extra_data.Y)).float()
            Y_tiled = Y.unsqueeze(1).expand(-1, extra_data.y.size(1), -1)
            y = torch.cat([Y_tiled, extra_data.y], dim=-1).float()

            temp_output = self.model(X, Y, e, x, y, node_mask, kernel)
            assert temp_output.x.shape[-1] == 0
            assert temp_output.X.shape[-1] == 1
            assert temp_output.e.shape[-1] == 1
            new_e = temp_output.e.squeeze(-1)
            output = utils.PlaceHolder(X=temp_output.X, E=new_e, y=temp_output.Y)

        else:
            raise NotImplementedError(f"Model type {self.model_type} is not implemented.")
        return output

    @torch.no_grad()
    def sample_batch(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int,
                     save_final: int, num_nodes=None):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)

        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask, multicat=self.multifactor)
        X, E, y = z_T.X, z_T.E, z_T.y
        
        term_width = shutil.get_terminal_size().columns
        torch.set_printoptions(threshold=float('inf'), linewidth=term_width)

        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps < self.T
        chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
        if self.multifactor:
            chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2), E.size(3)))
        else:
            chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2)))

        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        print("Sampling batch...")
        for s_int in tqdm(reversed(range(0, self.T)), desc="Sampling"):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            if self.num_node_labels is not None:
                sampled_s, discrete_sampled_s, node_labels = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask)
            else:
                sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            # Save the first keep_chain graphs
            write_index = (s_int * number_chain_steps) // self.T
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            if self.multifactor:
                chain_E[write_index] = sampled_s.E[:keep_chain] # ()
            else:
                chain_E[write_index] = discrete_sampled_s.E[:keep_chain]
        
        print("Batch sampled")

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True, multicat=self.multifactor)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        # Prepare the chain for saving
        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain                  # Overwrite last frame with the resulting X, E
            chain_E[0] = final_E_chain

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            # Repeat last frame to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            if self.multifactor:
                chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1, 1)], dim=0)
            else:
                chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.size(0) == (number_chain_steps + 10)


        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            if self.num_node_labels is not None:
                molecule_list.append([atom_types, edge_types, node_labels[i]])
            else:
                molecule_list.append([atom_types, edge_types])

        # Visualize chains
        # if self.visualization_tools is not None and (not self.multifactor):
        #     self.print('Visualizing chains...')
        #     current_path = os.getcwd()
        #     num_molecules = chain_X.size(1)       # number of molecules
        #     for i in range(num_molecules):
        #         result_path = os.path.join(current_path, f'chains/{self.cfg.general.name}/'
        #                                                  f'epoch{self.current_epoch}/'
        #                                                  f'chains/molecule_{batch_id + i}')
        #         if not os.path.exists(result_path):
        #             os.makedirs(result_path)
        #             _ = self.visualization_tools.visualize_chain(result_path,
        #                                                          chain_X[:, i, :].numpy(),
        #                                                          chain_E[:, i, :].numpy())
        #         self.print('\r{}/{} complete'.format(i+1, num_molecules), end='', flush=True)
        #     self.print('\nVisualizing molecules...')

        #     # Visualize the final molecules
        #     current_path = os.getcwd()
        #     result_path = os.path.join(current_path,
        #                                f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
        #     self.visualization_tools.visualize(result_path, molecule_list, save_final)
        #     self.print("Done.")

        return molecule_list

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        if self.num_node_labels is not None:
            # If node labels were extracted, add the loss for them
            node_labels, pred.X = self.extract_node_labels(pred.X)
        
        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
        if self.multifactor:
            pred_E = F.sigmoid(pred.E)                   # bs, n, n, d0
        else:
            pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0
        
        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=X_t,
                                                                                           Qt=Qt.X,
                                                                                           Qsb=Qsb.X,
                                                                                           Qtb=Qtb.X)

        if self.multifactor:
            E_list = diffusion_utils.split_edge_label_tensor(E_t)
            pred_E_list = diffusion_utils.split_edge_label_tensor(pred_E)
            Qt_E_list = diffusion_utils.split_edge_transition_tensor(Qt.E)
            Qsb_E_list = diffusion_utils.split_edge_transition_tensor(Qsb.E)
            Qtb_E_list = diffusion_utils.split_edge_transition_tensor(Qtb.E)
            assert len(E_list)==len(Qt_E_list)
            assert len(Qt_E_list)==len(Qsb_E_list)
            assert len(Qsb_E_list)==len(Qtb_E_list)

            p_s_and_t_given_0_E_list = []

            for i in range(len(E_list)):
                p_s_and_t_given_0_E_list.append(diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_list[i],
                                                                                           Qt=Qt_E_list[i],
                                                                                           Qsb=Qsb_E_list[i],
                                                                                           Qtb=Qtb_E_list[i]))

            # Dim of these two tensors: bs, N, d0, d_t-1
            weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
            unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
            unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
            prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1
            assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()

            prob_E_list = []
            E_s_list = []

            for i in range(len(E_list)):
                pred_E_list[i] = pred_E_list[i].reshape((bs, -1, pred_E_list[i].shape[-1]))
                weighted_E = pred_E_list[i].unsqueeze(-1) * p_s_and_t_given_0_E_list[i]        # bs, N, d0, d_t-1
                unnormalized_prob_E = weighted_E.sum(dim=-2)
                unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
                prob_E_list.append(unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True))
                prob_E_list[i] = prob_E_list[i].reshape(bs, n, n, pred_E_list[-1].shape[-1])
                assert ((prob_E_list[i].sum(dim=-1) - 1).abs() < 1e-4).all()
                sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E_list[i], node_mask=node_mask)
                X_s = F.one_hot(sampled_s.X, num_classes=self.x_diff_dim).float()
                E_s_list.append(F.one_hot(sampled_s.E, num_classes=2).float())
                assert (E_s_list[-1] == torch.transpose(E_s_list[-1], 1, 2)).all()
                assert (X_t.shape == X_s.shape) and (E_list[-1].shape == E_s_list[-1].shape)
            
            new_E_s_list = [E_s_list[i][..., 1] for i in range(len(E_s_list))]
            E_s = torch.stack(new_E_s_list, dim=-1)
            assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)
            out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
            out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))

            if self.num_node_labels is None:
                return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)
            else:
                return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t), node_labels

        else:
            p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)
            # Dim of these two tensors: bs, N, d0, d_t-1
            weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
            unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
            unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
            prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

            pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
            weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
            unnormalized_prob_E = weighted_E.sum(dim=-2)
            unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
            prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
            prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

            assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
            assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

            sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

            X_s = F.one_hot(sampled_s.X, num_classes=self.x_diff_dim).float()
            E_s = F.one_hot(sampled_s.E, num_classes=self.edim_output).float()

            assert (E_s == torch.transpose(E_s, 1, 2)).all()
            assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

            out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
            out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))

            if self.num_node_labels is None:
                return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)
            else:
                return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t), node_labels


    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """
        extra_features = self.extra_features(noisy_data)    

        t = noisy_data['t']
        extra_Y = torch.cat((extra_features.Y, t), dim=1)

        return utils.PlaceHolderMultilayer(X=extra_features.X,
                                           x=extra_features.x,
                                           e=extra_features.e,
                                           Y=extra_Y,
                                           y=extra_features.y)
