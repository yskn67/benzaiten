import math
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import lightning.pytorch as pl
from torchaudio.models import Conformer
from timm.scheduler import CosineLRScheduler
from transformers import get_cosine_schedule_with_warmup
from loguru import logger


MODE = Literal["pretrain", "finetune"]


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        pe = torch.zeros(1, max_len, hidden_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MelodyFixerModel(pl.LightningModule):

    def __init__(self, hidden_dim: int, output_dim: int, n_measures: int = 10, n_steps_per_measure: int = 16, n_mask: int = 4, mode: MODE = "pretrain") -> None:
        super().__init__()
        self.mode = mode
        self.n_measures = n_measures
        self.n_steps_per_measure = n_steps_per_measure
        self.n_steps = n_measures * n_steps_per_measure
        self.n_mask = n_mask
        self.conformer = Conformer(
            input_dim=hidden_dim,
            num_heads=4,
            ffn_dim=hidden_dim,
            num_layers=4,
            depthwise_conv_kernel_size=13,
            dropout=0.1,
        )
        self.melody_embedding = nn.Embedding(129, hidden_dim)
        self.chord_embedding = nn.Embedding(12, hidden_dim)
        self.pe = PositionalEncoding(hidden_dim=hidden_dim, max_len=self.n_steps)
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
                for idx in np.random.permutation(range(self.n_measures))[:self.n_mask]:
                    mask[idx * self.n_steps_per_measure:(idx + 1) * self.n_steps_per_measure, :] = 0.
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

        embedding = self.pe(masked_melody_embedding + chord_embedding)

        lengths = torch.ones(batch_size, dtype=torch.long, device=device) * self.n_steps
        out, _ = self.conformer(embedding, lengths)
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
            if self.current_epoch < 10:
                logger.info(f"freeze pretrained")
                for param in self.melody_embedding.parameters():
                    param.requires_grad = False
                for param in self.chord_embedding.parameters():
                    param.requires_grad = False
                for param in self.conformer.parameters():
                    param.requires_grad = False
                for param in self.linear.parameters():
                    param.requires_grad = False
            else:
                for param in self.melody_embedding.parameters():
                    param.requires_grad = True
                for param in self.chord_embedding.parameters():
                    param.requires_grad = True
                for param in self.conformer.parameters():
                    param.requires_grad = True
                for param in self.linear.parameters():
                    param.requires_grad = True

    def configure_optimizers(self):
        if self.mode == "pretrain":
            lr=1e-3
            min_lr=3e-5
            t_initial = 500
        else:
            lr=3e-4
            min_lr=1e-5
            t_initial = 100

        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": CosineLRScheduler(optimizer, t_initial=t_initial, lr_min=min_lr, warmup_t=20, warmup_lr_init=1e-5, warmup_prefix=True),
                "interval": "epoch",
            },
        }


if __name__ == '__main__':
    elf, hidden_dim: int, output_dim: int, n_measures: int = 10, n_steps_per_measure: int = 16, n_mask: int = 4, mode: MODE = "pretrain")
    hidden_dim = 128
    n_measures = 10
    n_steps_per_measure = 16
    model = MelodyFixerModel(hidden_dim=hidden_dim, output_dim=129, n_measures=n_measures, n_steps_per_measure=n_steps_per_measure, n_mask=4, map_location)
    notes = torch.randint(129, (n_measures * n_steps_per_measure,)).unsqueeze(0).float()
    chords = F.one_hot(torch.randint(12, (n_measures * n_steps_per_measure,)), num_classes=12).unsqueeze(0).float()

    out = model.forward({"notes": notes})
    print(out.size())

    out = model.forward({"notes": notes, "chord": chords})
    print(out.size())