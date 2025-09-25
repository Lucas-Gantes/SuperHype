#########################################################################

# This code is adapted from DiGress (https://github.com/cvignac/DiGress). 
# The calculation of the metrics and the generation of the datasets has been mostly taken from HYGENE (https://github.com/DorianGailhard/HYGENE).

#########################################################################


import graph_tool as gt
import os
import pathlib
import warnings

import torch
torch.cuda.empty_cache()
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from pytorch_lightning.tuner import Tuner


from torch_ema import ExponentialMovingAverage

from src import utils
from src.metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics

from src.diffusion_model import LiftedDenoisingDiffusion
from src.diffusion_model_discrete import DiscreteDenoisingDiffusion
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures, CliqueComputation


from ray import tune
from src.tune_train import train_tune

from copy import deepcopy


warnings.filterwarnings("ignore", category=PossibleUserWarning)


def get_resume(cfg, model_kwargs):
    """ Resumes a run. It loads previous config without allowing to update keys (used for testing). """
    saved_cfg = cfg.copy()
    name = cfg.general.name + '_resume'
    resume = cfg.general.test_only
    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """ Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    saved_cfg = cfg.copy()
    # Fetch path to this file to get base path
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split('outputs')[0]

    resume_path = os.path.join(root_dir, cfg.general.resume)

    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
    new_cfg = model.cfg

    for category in cfg:
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path
    new_cfg.general.name = new_cfg.general.name + '_resume'

    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model


def freeze_layers(model, augmentation_cfg):
    gs_transformer = model.model
    if "xeyTrLayers" in augmentation_cfg.freeze_layers:
        print(f"Freezing the {augmentation_cfg.freeze_layers['xeyTrLayers']} first xey layers")
        for layer in gs_transformer.xeyTrLayers[:augmentation_cfg.freeze_layers["xeyTrLayers"]]:
            for p in layer.parameters():
                p.requires_grad = False
            
    if "XxTrLayers" in augmentation_cfg.freeze_layers:
        print(f"Freezing the {augmentation_cfg.freeze_layers['XxTrLayers']} first Xx layers")
        for layer in gs_transformer.XxTrLayers[:augmentation_cfg.freeze_layers["XxTrLayers"]]:
            for p in layer.parameters():
                p.requires_grad = False
    
    if "YyTrLayers" in augmentation_cfg.freeze_layers:
        print(f"Freezing the {augmentation_cfg.freeze_layers['YyTrLayers']} first Yy layers")
        for layer in gs_transformer.YyTrLayers[:augmentation_cfg.freeze_layers["YyTrLayers"]]:
            for p in layer.parameters():
                p.requires_grad = False
    
    if "YXTrLayers" in augmentation_cfg.freeze_layers:
        print(f"Freezing the {augmentation_cfg.freeze_layers['YXTrLayers']} first YX layers")
        for layer in gs_transformer.YXTrLayers[:augmentation_cfg.freeze_layers["YXTrLayers"]]:
            for p in layer.parameters():
                p.requires_grad = False

    n_tot = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters after freezing: {n_train}/{n_tot}")


@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    main_dir = get_original_cwd()
    dataset_config = cfg["dataset"]
    custom_hgs = ["erdos", "sbm_custom", "ego", "hypertrees", "meshPiano", "meshBookshelf", "meshPlant"]

    if dataset_config["name"] in ['sbm', 'comm20', 'planar', 'custom_dataset'] or dataset_config["name"] in custom_hgs:
        from src.datasets.spectre_dataset import SpectreGraphDataModule, SpectreDatasetInfos
        from src.datasets.custom_dataset import CustomGraphDataModule, CustomDatasetInfos
        from src.datasets.custom_augmented_dataset import GraphAugmentedDataModule, CustomAugmentedDatasetInfos
        from src.analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics
        from src.analysis.multi_label_sampling_metrics import MultiLabelSamplingMetrics
        from src.analysis.visualization import NonMolecularVisualization

        if dataset_config["name"] == 'custom_dataset' or dataset_config["name"] in custom_hgs:
            if dataset_config["data_augmentation"]:
                print("Data augmentation will be used to train the model")
                datamodule = GraphAugmentedDataModule(cfg, post_processing=dataset_config["post_processing"], change_rate=dataset_config["change_rate"])
                dataset_infos = CustomAugmentedDatasetInfos(datamodule)
            else:
                print("The model will be trained without data augmentation")
                datamodule = CustomGraphDataModule(cfg, post_processing=dataset_config["post_processing"])
                dataset_infos = CustomDatasetInfos(datamodule, dataset_config)
            multifactor = dataset_config["multilabel"]
            num_node_labels = dataset_infos.nb_node_labels
        else:
            print("Creating datamodule for spectre dataset")
            datamodule = SpectreGraphDataModule(cfg)
            dataset_infos = SpectreDatasetInfos(datamodule, dataset_config)
            multifactor = False
            num_node_labels = None

        if 'metrics' in cfg.general:
            metrics = cfg.general.metrics
        else:
            metrics = ['node_degree', 'edge_size', 'spectral', 'clique_size', 'uniqueness', 'novelty', 'nb_nodes', 'centrality_closeness', 'centrality_betweenness', 'centrality_harmonic']
        
        if dataset_config['name'] == 'sbm':
            print("Using SBM dataset")
            sampling_metrics = SBMSamplingMetrics(datamodule)
        elif dataset_config['name'] == 'comm20':
            sampling_metrics = Comm20SamplingMetrics(datamodule)
        elif dataset_config['name'] == 'custom_dataset' or dataset_config['name'] in custom_hgs:
            sampling_metrics = MultiLabelSamplingMetrics(
                datamodule,
                ref_metrics=dataset_config['baseline'],
                multicat=dataset_config["multilabel"],
                compute_emd=False,
                metrics_list=metrics,
                post_processing=dataset_config["post_processing"],
                data_augmentation=dataset_config["data_augmentation"]
            )
        else:
            sampling_metrics = PlanarSamplingMetrics(datamodule)

        train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == 'discrete' else TrainAbstractMetrics()
        visualization_tools = NonMolecularVisualization()

        if cfg.model.type == 'discrete' and cfg.model.extra_features== 'cliques':
            extra_features = CliqueComputation(dataset_info=dataset_infos,  clique_sizes=cfg.model.clique_sizes, algorithm=cfg.model.extra_features_algorithm)
        elif cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

        print("Type of extra features:", type(extra_features))

        if dataset_config["name"] == 'custom_dataset' or dataset_config["name"] in custom_hgs:
            dataset_infos.compute_input_output_dims(extra_features=extra_features,
                                                domain_features=domain_features)
        else:
            dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)
        
        if 'clique_loss_coef' in dataset_config:
            clique_loss_coef = dataset_config['clique_loss_coef']
        else:
            clique_loss_coef = None
        
        if 'model_type' in cfg["model"]:
            model_type = cfg["model"]["model_type"]
        else:
            model_type = "XEyTransformer"
        
        if 'kernel_coef' in cfg["model"]:
            kernel_coef = cfg["model"]["kernel_coef"]
        else:
            kernel_coef = None
        
        if 'triplet_interactions' in cfg["model"]:
            triplet_interactions = cfg["model"]["triplet_interactions"]
        else:
            triplet_interactions = None
        
        if 'parallel_model' in cfg["model"]:
            parallel_model = cfg["model"]["parallel_model"]
        else:
            print("No information found about parallelization of the model, default=False")
            parallel_model = False
        
        if 'ema_decay' in cfg.train:
            ema_decay = cfg.train["ema_decay"]
        else:
            print("No inforamtion on ema decay - no ema will be applied")
            ema_decay = None
        
        if 'single_layer' in cfg["model"]:
            single_layer = cfg["model"]["single_layer"]
        else:
            print("Parameter single_layer not found, default=False")
            single_layer = False


        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features, 'multifactor': multifactor, 
                        'num_node_labels':num_node_labels, 'clique_loss_coef': clique_loss_coef, 'model_type': model_type, 
                        'kernel_coef': kernel_coef, 'triplet_interactions': triplet_interactions, 'parallel_model': parallel_model,
                        'ema_decay': ema_decay, 'single_layer': single_layer}

    elif dataset_config["name"] in ['qm9', 'guacamol', 'moses']:
        from src.metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
        from src.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
        from src.diffusion.extra_features_molecular import ExtraMolecularFeatures
        from src.analysis.visualization import MolecularVisualization

        if dataset_config["name"] == 'qm9':
            from datasets import qm9_dataset
            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
            train_smiles = qm9_dataset.get_train_smiles(cfg=cfg, train_dataloader=datamodule.train_dataloader(),
                                                        dataset_infos=dataset_infos, evaluate_dataset=False)
        elif dataset_config['name'] == 'guacamol':
            from datasets import guacamol_dataset
            datamodule = guacamol_dataset.GuacamolDataModule(cfg)
            dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)
            train_smiles = None

        elif dataset_config.name == 'moses':
            from datasets import moses_dataset
            datamodule = moses_dataset.MosesDataModule(cfg)
            dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
            train_smiles = None
        else:
            raise ValueError("Dataset not implemented")

        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
            domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()
            domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        if cfg.model.type == 'discrete':
            train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
        else:
            train_metrics = TrainMolecularMetrics(dataset_infos)

        # We do not evaluate novelty during training
        sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
        visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    if cfg.general.test_only:
        # When testing, previous configuration is fully loaded
        cfg, _ = get_resume(cfg, model_kwargs)
        os.chdir(cfg.general.test_only.split('checkpoints')[0])
    elif cfg.general.resume is not None:
        # When resuming, we can override some parts of previous configuration
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)
        os.chdir(cfg.general.resume.split('checkpoints')[0])

    utils.create_folders(cfg)

       ### A TEST TO TUNE THE HYPERPARAMETERS ###

    if "tuning" in cfg and cfg["tuning"]["find_optimal_params"]:
        search_space = {
            "n_layers": tune.grid_search([3, 4, 5, 6, 7, 8, 9, 10]),
        }

        storage_uri = pathlib.Path("ray_results").absolute().as_uri()

        analysis = tune.run(
            tune.with_parameters(
                train_tune,
                dataset_config=cfg,
                model_kwargs=model_kwargs
            ),
            resources_per_trial={"cpu": 4, "gpu": 0.125},
            config=search_space,
            scheduler=tune.schedulers.ASHAScheduler(
                metric="is_hypertree",
                mode="max",
                max_t=cfg.train.n_epochs,
                grace_period=20,
                reduction_factor=2
            ),
            progress_reporter=tune.CLIReporter(
                metric_columns=["val/epoch_NLL", "training_iteration", "centrality_closeness", "is_hypertree"]
            ),
            storage_path=storage_uri,
            name="graph_diffusion_tuning"
        )
        print("Best hyperparameters found were: ", analysis.best_config)

    #-----------------------------------------------------------------------


    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion(cfg=cfg, **model_kwargs)

    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val/epoch_NLL',
                                              save_top_k=5,
                                              mode='min',
                                              every_n_epochs=1,
                                              verbose=True)
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='last', every_n_epochs=1, verbose=True)
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)


    name = cfg.general.name
    if name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()

    print("Clip gradient value:", cfg.train.clip_grad)

    class Ping(Callback):  # For debug only
        def on_train_epoch_end(self, trainer, *_):
            if trainer.is_global_zero:
                print(">>> rank 0 writes in", os.path.abspath(trainer.default_root_dir))


    print("Creating the trainer...")
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints
                      accelerator='gpu' if use_gpu else 'cpu',
                      devices=cfg.general.gpus if use_gpu else 1,
                      detect_anomaly=cfg.general.detect_anomaly,  # Needed to debug NaN issues
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      enable_progress_bar=False,
                      # callbacks=callbacks,
                      callbacks = callbacks + [Ping()],
                      log_every_n_steps=50 if name != 'debug' else 1,
                      logger = [])

    print("Trainer created")

    print("trainer.callbacks: "+str(trainer.callbacks))

    if "tuning" in cfg and cfg["tuning"]["find_optimal_lr"]:
        tuner = Tuner(trainer)

        lr_finder = tuner.lr_find(model, datamodule=datamodule, max_lr=cfg.tuning.max_lr)


    if not cfg.general.test_only:
        if not "augmentation_train" in cfg:
            trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
            if cfg.general.name not in ['debug', 'test']:
                print("Testing the model after the pretraining phase...")
                trainer.test(model, datamodule=datamodule)
            
            if "augmentation_train" in cfg:
                pass
        elif "augmentation_train" in cfg and cfg.augmentation_train["only_augmentation"]:
            print("The model will only be trained in augmentation mode")
            if cfg.augmentation_train["load_checkpoint"] is not None:
                
                augmentation_kwargs = model_kwargs.copy()
                new_cfg = deepcopy(cfg)

                for param in cfg.augmentation_train["overriden_train"]:
                    new_cfg.train[param] = cfg.augmentation_train[param]

                for param in cfg.augmentation_train["overriden_dataset"]:
                    new_cfg.dataset[param] = cfg.augmentation_train[param]

                if cfg.augmentation_train["data_augmentation"]:
                    print("The dataset for the second training is in augmentation mode")
                    new_datamodule = GraphAugmentedDataModule(new_cfg, post_processing=new_cfg.dataset["post_processing"], 
                                                              change_rate=cfg.augmentation_train["change_rate"])
                    new_dataset_infos = CustomAugmentedDatasetInfos(new_datamodule)
                    new_dataset_infos.compute_input_output_dims(extra_features=augmentation_kwargs["extra_features"],
                                                domain_features=augmentation_kwargs["domain_features"])
                else:
                    print("The dataset for the second training is in static mode")
                    new_datamodule = CustomGraphDataModule(new_cfg, post_processing=new_cfg.dataset["post_processing"])
                    new_dataset_infos = CustomDatasetInfos(new_datamodule, new_cfg.dataset)
                    new_dataset_infos.compute_input_output_dims(extra_features=augmentation_kwargs["extra_features"],
                                                domain_features=augmentation_kwargs["domain_features"])
                
                augmentation_kwargs["dataset_infos"] = new_dataset_infos
                augmentation_kwargs["ema_decay"] = new_cfg.train["ema_decay"]



                ckpt_dir = os.path.join(main_dir, "checkpoints")
                print("ckpt_dir: "+str(ckpt_dir))
                ckpt_path = os.path.join(ckpt_dir, cfg.augmentation_train["load_checkpoint"])
                print("ckpt_path: "+str(ckpt_path))
                new_model = DiscreteDenoisingDiffusion.load_from_checkpoint(ckpt_path, cfg=new_cfg, **augmentation_kwargs)

                freeze_layers(new_model, cfg.augmentation_train)

                new_trainer = Trainer(gradient_clip_val=new_cfg.train.clip_grad,
                                        strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints
                                        accelerator='gpu' if use_gpu else 'cpu',
                                        devices=new_cfg.general.gpus if use_gpu else 1,
                                        detect_anomaly=new_cfg.general.detect_anomaly,  # Needed to debug NaN issues
                                        max_epochs=new_cfg.train.n_epochs,
                                        check_val_every_n_epoch=new_cfg.general.check_val_every_n_epochs,
                                        fast_dev_run=new_cfg.general.name == 'debug',
                                        enable_progress_bar=False,
                                        callbacks = callbacks + [Ping()],
                                        log_every_n_steps=50 if name != 'debug' else 1,
                                        logger = [])
                
                if "tuning" in cfg and cfg["tuning"]["find_optimal_lr_augmentation"]:
                    print("Finding the best learning rate for the second training phase")
                    tuner = Tuner(new_trainer)

                    lr_finder = tuner.lr_find(new_model, datamodule=new_datamodule)
                
                new_trainer.fit(new_model, datamodule=new_datamodule, ckpt_path=new_cfg.general.resume)

            else:
                raise ValueError("The model is supposed to be in augmentation only mode, but no checkpoint has been specified")
        else:
            raise ValueError("No training task specified in the congiguration file")
    else:
        # Start by evaluating test_only_path
        print("Testing the model only")
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            print("Directory:", directory)
            files_list = os.listdir(directory)
            for file in files_list:
                if '.ckpt' in file:
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == cfg.general.test_only:
                        continue
                    print("Loading checkpoint", ckpt_path)
                    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
