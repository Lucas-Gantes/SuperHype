import os
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from pytorch_lightning import Trainer

from src import utils
from src.metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics

from src.diffusion_model import LiftedDenoisingDiffusion
from src.diffusion_model_discrete import DiscreteDenoisingDiffusion
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures, CliqueComputation

import torch
torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from pytorch_lightning.tuner import Tuner

import copy

from src.datasets.custom_dataset import CustomGraphDataModule, CustomDatasetInfos
from src.datasets.custom_augmented_dataset import GraphAugmentedDataModule, CustomAugmentedDatasetInfos


def train_tune(config, model_kwargs=None, checkpoint_dir=None, dataset_config=None):
    cfg = copy.deepcopy(dataset_config) 
    cfg.model.n_layers = config["n_layers"]

    if checkpoint_dir:
        ckpt_path = os.path.join(checkpoint_dir, "checkpoint.ckpt")  
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(ckpt_path, **model_kwargs)
    else:
        model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)

    if dataset_config.dataset["data_augmentation"]:
        datamodule = GraphAugmentedDataModule(cfg, post_processing=dataset_config.dataset["post_processing"], change_rate=dataset_config["change_rate"])
    else:
        datamodule = CustomGraphDataModule(cfg, post_processing=dataset_config.dataset["post_processing"])

    dataset_infos = CustomDatasetInfos(datamodule, cfg.dataset)
    dataset_infos.compute_input_output_dims(
        extra_features=model_kwargs["extra_features"],
        domain_features=model_kwargs["domain_features"]
    )
    model_kwargs["dataset_infos"] = dataset_infos

    metrics = {"centrality_closeness_train": "centrality_closeness_train", "is_hypertree":"is_hypertree"}
    tune_reporter = TuneReportCallback(metrics, on="validation_end")

    tune_checkpoint = TuneReportCheckpointCallback(
        metrics,
        filename="checkpoint.ckpt", on="validation_end"
    )

    trial_dir = os.getcwd()

    ckpt_cb = ModelCheckpoint(
        monitor="is_hypertree",
        mode="min",
        save_top_k=1,
        dirpath=os.path.join(trial_dir, "checkpoints"),
        filename="best"
    )

    trainer = Trainer(
        max_epochs=cfg.train.n_epochs,
        callbacks=[tune_reporter, tune_checkpoint, ckpt_cb],
        accelerator="gpu" if cfg.general.gpus>0 else "cpu",
        devices=cfg.general.gpus or 1,
        enable_progress_bar=False,
        logger=False,
    )
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints
                      accelerator="gpu" if cfg.general.gpus>0 else "cpu",
                      devices=cfg.general.gpus or 1,
                      detect_anomaly=True,  # Needed to debug NaN issues
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      enable_progress_bar=False,
                      callbacks=[tune_reporter, tune_checkpoint, ckpt_cb],
                      log_every_n_steps=50,
                      logger = [])

    trainer.fit(model, datamodule=datamodule)



