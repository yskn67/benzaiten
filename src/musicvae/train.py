import glob
import os

import torch
import lightning.pytorch as pl
import hydra
from omegaconf import DictConfig

from dataset import MusicVaeDataset
from model import MusicVaeModel


@hydra.main(version_base=None, config_path="../../conf/musicvae", config_name="config")
def main(cfg: DictConfig) -> None:
    flist = list(glob.glob("data/preprocess/maestro/*/*.json"))
    ds = MusicVaeDataset(files=flist)
    dl = torch.utils.data.DataLoader(dataset=ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=6)
    model = MusicVaeModel(
        hidden_dim=cfg.model.hidden_dim,
        latent_dim=cfg.model.latent_dim,
        n_steps_per_measure=cfg.data.n_beats * cfg.data.n_parts_of_beat,
    )
    trainer = pl.Trainer(max_epochs=cfg.train.max_epochs)

    trainer.fit(model=model, train_dataloaders=dl)


if __name__ == '__main__':
    main()