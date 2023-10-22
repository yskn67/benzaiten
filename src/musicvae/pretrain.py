import glob
import os
import shutil
import sys

import torch
import lightning.pytorch as pl
import hydra
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from dataset import MusicVaeDataset
from model import MusicVaeModel


@hydra.main(version_base=None, config_path="../../conf/musicvae", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.remove()
    logger.add(sys.stderr, level=cfg.logger.level)
    artifact_path = os.path.join(cfg.generate.output_dir, cfg.name)
    os.makedirs(artifact_path, exist_ok=True)

    lakh_matched = list(glob.glob("data/preprocess/lakh_matched/*/*.json"))
    logger.info(f"lahk_matched: {len(lakh_matched)}")
    maestro = list(glob.glob("data/preprocess/maestro/*/*.json"))
    logger.info(f"maestro: {len(maestro)}")
    infinite_bach = list(glob.glob("data/preprocess/infinite_bach/*.json"))
    logger.info(f"infinite_bach: {len(infinite_bach)}")
    flist = maestro + lakh_matched + infinite_bach
    logger.info(f"total: {len(flist)}")

    ds = MusicVaeDataset(files=flist)
    dl = torch.utils.data.DataLoader(dataset=ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=max(os.cpu_count() - 2, 1))
    model = MusicVaeModel(
        mode=cfg.model.mode,
        encoder_hidden_dim=cfg.model.encoder_hidden_dim,
        decoder_hidden_dim=cfg.model.decoder_hidden_dim,
        latent_dim=cfg.model.latent_dim,
        n_steps_per_measure=cfg.data.n_beats * cfg.data.n_parts_of_beat,
    )

    lightning_logger = CSVLogger("logs/musicvae/")
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="train_loss_epoch")
    earlystopping_callback = EarlyStopping(monitor="train_loss_epoch", patience=5, verbose=True, mode="min")
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        logger=lightning_logger,
        callbacks=[
            checkpoint_callback,
            earlystopping_callback,
        ],
    )

    trainer.fit(model=model, train_dataloaders=dl)
    logger.info(f"best model: {checkpoint_callback.best_model_path}")
    shutil.copy(checkpoint_callback.best_model_path, os.path.join(artifact_path, "best_model.ckpt"))
    OmegaConf.save(config=cfg, f=os.path.join(artifact_path, "config.yaml"))


if __name__ == '__main__':
    main()