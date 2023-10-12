import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import lightning.pytorch as pl


class Encoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_dim,
            hidden_dim,
            1,
            batch_first=True,
            bidirectional=True
        )
        self.linear_mean = nn.Linear(hidden_dim * 2, output_dim)
        self.linear_lnvar = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, (h, c) = self.rnn(batch["notes"])
        n_layer, batch_size, hidden_dim = h.size()
        # FIXME: RNNのlayer数が1前提
        z = h.transpose(0, 1).reshape((batch_size, n_layer * hidden_dim))
        z = F.tanh(z)
        mean = self.linear_mean(z)
        lnvar = self.linear_lnvar(z)
        std = torch.exp(0.5 * lnvar)
        eps = torch.randn_like(std)
        latent = mean + eps * std
        return latent, h, c


class Decoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_dim,
            hidden_dim,
            1,
            batch_first=True,
            bidirectional=False
        )
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.output_dim = output_dim
        self.n_layer = 1
        self.hidden_dim = hidden_dim

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = batch["chords"].size()
        latent = batch["latent"].unsqueeze(1).repeat(1, seq_len, 1)
        if "notes" in batch:
            previous_notes = torch.zeros_like(batch["notes"])
            previous_notes[:, 1:, :] = batch["notes"][:, :-1, :]
            x = torch.cat([latent, previous_notes, batch["chords"]], dim=2)
            out, (h, c) = self.rnn(x)
            out = self.linear(F.tanh(out))
            return out, h, c
        else:
            c_0 = torch.zeros((self.n_layer, batch_size, self.hidden_dim), dtype=torch.float)
            if 'c' in batch:
                c_0 = batch['c']
            h_0 = torch.zeros((self.n_layer, batch_size, self.hidden_dim), dtype=torch.float)
            if 'h' in batch:
                h_0 = batch['h']
            previous_notes = torch.zeros((batch_size, seq_len, self.output_dim), dtype=torch.float)
            for i in range(seq_len):
                x = torch.cat([latent, previous_notes, batch["chords"]], dim=2)
                out, (h, c) = self.rnn(x, (h_0, c_0))
                out = self.linear(F.tanh(out))
                if i < seq_len - 1:
                    previous_notes = torch.zeros((batch_size, seq_len, self.output_dim), dtype=torch.float)
                    predicted_note_indices = torch.argmax(out, dim=2)
                    for j, index_slice in enumerate(predicted_note_indices):
                        for k , index in enumerate(index_slice):
                            if k < len(index_slice) - 1:
                                previous_notes[j, k + 1, index] = 1
            return out, h, c


class Seq2SeqMelodyGenerationModel(pl.LightningModule):

    def __init__(self, melody_dim: int=54, base_dim: int=12, hidden_dim: int=1024, latent_dim: int=32):
        super().__init__()
        self.encoder = Encoder(melody_dim, latent_dim, hidden_dim)
        self.decoder = Decoder(melody_dim + base_dim + latent_dim, melody_dim, hidden_dim)
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        latent, _, _ = self.encoder(batch)
        batch["latent"] = latent
        out, _, _ = self.decoder(batch)
        return out

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = self(batch)
        loss = self.criterion(out.transpose(1, 2), batch["labels"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {
            "loss": loss
        }

    def configure_optimizers(self):
        return optim.Adam(
            self.parameters(),
            lr=1e-3
        )