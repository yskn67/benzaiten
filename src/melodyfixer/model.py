from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import lightning.pytorch as pl
from loguru import logger
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


MODE = Literal["pretrain", "finetune", "inference"]


class PositionalEncoding(nn.Module):

    def __init__(self, hidden_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(-np.log(10000.0) * torch.arange(0, hidden_dim, 2) / hidden_dim)
        positional_encoding = torch.zeros(1, max_len, hidden_dim)
        positional_encoding[0, :, 0::2] = torch.sin(position * div_term)
        positional_encoding[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.positional_encoding[:, :x.size(1), :]
        return x


class MelodyFixerModel(pl.LightningModule):

    def __init__(
            self,
            hidden_dim: int,
            output_dim: int,
            n_measures: int = 10,
            n_steps_per_measure: int = 16,
            n_mask: int = 4,
            mode: MODE = "pretrain",
            mask_measures: list[int] | None = None,
        ) -> None:
        super().__init__()
        self.mode = mode
        self.n_measures = n_measures
        self.n_steps_per_measure = n_steps_per_measure
        self.n_steps = n_measures * n_steps_per_measure
        self.n_mask = n_mask
        self.mask_measures = mask_measures
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                batch_first=True
            ),
            num_layers=4,
        )
        self.melody_embedding = nn.Embedding(129, hidden_dim)
        self.chord_embedding = nn.Embedding(12, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim=hidden_dim, max_len=self.n_steps)
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = batch["notes"].size(dim=0)
        device = batch["notes"].device

        melody_embedding = self.melody_embedding(batch["notes"])
        if self.n_mask > 0:
            masks= []
            for _ in range(batch_size):
                mask = torch.ones(self.n_steps, self.hidden_dim, dtype=torch.float, device=device)
                if self.mode == "inference" and self.mask_measures is not None:
                    mask_idx = self.mask_measures
                else:
                    mask_idx = np.random.permutation(range(self.n_measures))[:self.n_mask]
                for idx in mask_idx:
                    mask[idx * self.n_steps_per_measure:(idx + 1) * self.n_steps_per_measure, :] = 0.
                masks.append(mask.unsqueeze(0))
            masks = torch.cat(masks, dim=0)
            masked_melody_embedding = melody_embedding * masks
        else:
            masked_melody_embedding = melody_embedding

        if "chords" in batch:
            nonzero = (batch["chords"].reshape(batch_size * self.n_steps, 12) == 1).nonzero()
            chord_note_embedding = self.chord_embedding(nonzero[:, 1])
            expand_index = nonzero[:, 0].repeat(self.hidden_dim, 1).transpose(0, 1)
            chord_embedding = torch.zeros(batch_size * self.n_steps, self.hidden_dim, dtype=torch.float, device=device).scatter_reduce(0, expand_index, chord_note_embedding, reduce="sum")
            chord_embedding = chord_embedding.reshape(batch_size, self.n_steps, self.hidden_dim)
        else:
            chord_embedding = torch.zeros(batch_size, self.n_steps, self.hidden_dim, dtype=torch.float, device=device)

        embedding = self.positional_encoding(masked_melody_embedding + chord_embedding)
        embedding = self.dropout(embedding)
        out = self.transformer(embedding)
        return self.linear(out)

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = self(batch)
        loss = self.criterion(out.transpose(1, 2), batch["notes"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {
            "loss": loss
        }

    def on_train_epoch_start(self) -> None:
        if self.mode == "finetune":
            if self.current_epoch < 5:
                logger.info(f"freeze pretrained")
                for param in self.melody_embedding.parameters():
                    param.requires_grad = False
                for param in self.transformer.parameters():
                    param.requires_grad = False
                for param in self.linear.parameters():
                    param.requires_grad = False
            else:
                for param in self.melody_embedding.parameters():
                    param.requires_grad = True
                for param in self.transformer.parameters():
                    param.requires_grad = True
                for param in self.linear.parameters():
                    param.requires_grad = True

    def configure_optimizers(self):
        if self.mode == "pretrain":
            warmup_epochs=20
            max_epochs=200
            warmup_start_lr=1e-5
            eta_min=1e-5
            lr=1e-3
        else:
            warmup_epochs=10
            max_epochs=100
            warmup_start_lr=1e-6
            eta_min=1e-6
            lr=3e-4

        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=warmup_epochs,
                    max_epochs=max_epochs,
                    warmup_start_lr=warmup_start_lr,
                    eta_min=eta_min
                ),
                "interval": "epoch",
            },
        }


if __name__ == '__main__':
    hidden_dim = 128
    n_measures = 10
    n_steps_per_measure = 16
    model = MelodyFixerModel(hidden_dim=hidden_dim, output_dim=129, n_measures=n_measures, n_steps_per_measure=n_steps_per_measure, n_mask=4)
    notes = torch.randint(129, (n_measures * n_steps_per_measure,)).unsqueeze(0)
    chords = F.one_hot(torch.randint(12, (n_measures * n_steps_per_measure,)), num_classes=12).unsqueeze(0)

    out = model.forward({"notes": notes})
    print(out.size())

    out = model.forward({"notes": notes, "chord": chords})
    print(out.size())