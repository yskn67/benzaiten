"""
These codes are licensed under MIT License.
Copyright (C) 2023 yskn67
https://github.com/yskn67/benzaiten/blob/2nd/LICENSE
"""

import glob
import os
import shutil
import sys
from datetime import datetime

import torch
import lightning.pytorch as pl
import hydra
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from dataset import MelodyFixerDataset
from model import MelodyFixerModel


@hydra.main(version_base=None, config_path="../../conf/melodyfixer", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.remove()
    logger.add(sys.stderr, level=cfg.logger.level)
    artifact_path = os.path.join(cfg.generate.output_dir, cfg.name)
    os.makedirs(artifact_path, exist_ok=True)

    lakh_matched = list(glob.glob(os.path.join(cfg.train.input_dir, "lakh_matched/*/*.json")))
    logger.info(f"lahk_matched: {len(lakh_matched)}")
    maestro = list(glob.glob(os.path.join(cfg.train.input_dir, "maestro/*/*.json")))
    logger.info(f"maestro: {len(maestro)}")
    infinite_bach = list(glob.glob(os.path.join(cfg.train.input_dir, "infinite_bach/*.json")))
    logger.info(f"infinite_bach: {len(infinite_bach)}")
    weimar_midi = list(glob.glob(os.path.join(cfg.train.input_dir, "weimar_midi/*.json")))
    logger.info(f"weimar_midi: {len(weimar_midi)}")
    flist = maestro + lakh_matched + infinite_bach + weimar_midi
    logger.info(f"total: {len(flist)}")

    ds = MelodyFixerDataset(files=flist)
    dl = torch.utils.data.DataLoader(dataset=ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=max(os.cpu_count(), 1))
    model = MelodyFixerModel(
        mode=cfg.model.mode,
        hidden_dim=cfg.model.hidden_dim,
        output_dim=129,
        n_measures=cfg.data.n_measures + 2,
        n_steps_per_measure=cfg.data.n_beats * cfg.data.n_parts_of_beat,
    )

    # lightning_logger = CSVLogger("logs/musicvae/")
    lightning_logger = WandbLogger(
        project="benzaiten-melodyfixer",
        name=f"{cfg.name}-{datetime.now().isoformat()}",
        save_dir="logs/melodyfixer",
        log_model=True,
    )
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="train_loss_epoch")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # earlystopping_callback = EarlyStopping(monitor="train_loss_epoch", patience=cfg.train.patience, verbose=True, mode="min")
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        logger=lightning_logger,
        callbacks=[
            checkpoint_callback,
            lr_monitor,
            # earlystopping_callback,
        ],
    )

    trainer.fit(model=model, train_dataloaders=dl)
    logger.info(f"best model: {checkpoint_callback.best_model_path}")
    shutil.copy(checkpoint_callback.best_model_path, os.path.join(artifact_path, "best_model.ckpt"))
    OmegaConf.save(config=cfg, f=os.path.join(artifact_path, "config.yaml"))


if __name__ == '__main__':
    main()