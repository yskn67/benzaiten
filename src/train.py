import glob

import torch
import lightning.pytorch as pl
import hydra
from omegaconf import DictConfig, OmegaConf

from dataset import make_onehot_dataset
from model import Seq2SeqMelodyGenerationModel


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    flist = list(glob.glob("data/input/omnibook/Omnibook xml/*.xml"))
    d = make_onehot_dataset(
        flist,
        n_bar=cfg.data.n_bar,
        min_note_number=cfg.data.min_note_number,
        max_note_number=cfg.data.max_note_number,
    )
    dl = torch.utils.data.DataLoader(dataset=d, batch_size=cfg.train.batch_size, shuffle=True, num_workers=6)
    model = Seq2SeqMelodyGenerationModel(
        melody_dim=cfg.model.melody_dim,
        base_dim=cfg.model.base_dim,
        hidden_dim=cfg.model.hidden_dim,
        latent_dim=cfg.model.latent_dim,
    )
    trainer = pl.Trainer(max_epochs=cfg.train.max_epochs)

    trainer.fit(model=model, train_dataloaders=dl)


if __name__ == '__main__':
    main()