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

from dataset import make_finetune_dataset
from model import MelodyFixerModel


@hydra.main(version_base=None, config_path="../../conf/melodyfixer_finetune", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.remove()
    logger.add(sys.stderr, level=cfg.logger.level)
    artifact_path = os.path.join(cfg.generate.output_dir, cfg.name)
    os.makedirs(artifact_path, exist_ok=True)

    pretrain_artifact_path = os.path.join(cfg.train.pretrain_output_dir, cfg.pretrain_name)
    pretrain_checkpoint_path = os.path.join(pretrain_artifact_path, "best_model.ckpt")
    pretrain_cfg = OmegaConf.load(os.path.join(pretrain_artifact_path, "config.yaml"))

    omnibook = list(glob.glob("data/input/omnibook/Omnibook xml/*.xml"))
    logger.info(f"omnibook: {len(omnibook)}")
    openewld = list(glob.glob("data/input/openewld/OpenEWLD-0.1/dataset/**/*.mxl", recursive=True))
    logger.info(f"openewld: {len(openewld)}")
    flist = omnibook + openewld
    logger.info(f"total: {len(flist)}")

    ds = make_finetune_dataset(
        flist,
        n_bar=pretrain_cfg.data.n_measures,
        n_beats=pretrain_cfg.data.n_beats,
        n_parts_of_beat=pretrain_cfg.data.n_parts_of_beat,
    )
    dl = torch.utils.data.DataLoader(dataset=ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=max(os.cpu_count(), 1))

    model = MelodyFixerModel.load_from_checkpoint(
        pretrain_checkpoint_path,
        mode=cfg.model.mode,
        hidden_dim=pretrain_cfg.model.hidden_dim,
        output_dim=129,
        n_measures=pretrain_cfg.data.n_measures + 2,
        n_steps_per_measure=pretrain_cfg.data.n_beats * pretrain_cfg.data.n_parts_of_beat,
    )

    # lightning_logger = CSVLogger("logs/musicvae_finetune/")
    lightning_logger = WandbLogger(
        project="benzaiten-melodyfixer-finetune",
        name=f"{cfg.name}-{datetime.now().isoformat()}",
        save_dir="logs/melodyfixer_finetune",
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
    OmegaConf.save(config=pretrain_cfg, f=os.path.join(artifact_path, "config.pretrain.yaml"))
    OmegaConf.save(config=cfg, f=os.path.join(artifact_path, "config.finetune.yaml"))


if __name__ == '__main__':
    main()