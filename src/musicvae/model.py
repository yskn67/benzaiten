from typing import Literal

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import lightning.pytorch as pl
from loguru import logger
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


MODE = Literal["pretrain", "finetune"]


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
        # NOTE: 一旦local_featuresを無効にしてみる
        # FIXME: とりあえずカーネルサイズやフィルターの数は決め打ち
        # self.local_filter = nn.Conv1d(
        #     hidden_dim * 2,  #bidirectionalなので
        #     3,  # フィルター数
        #     kernel_size=9,
        #     padding=4,
        #     bias=False
        # )
        self.linear_mean = nn.Linear(hidden_dim * 2, output_dim)
        self.linear_lnvar = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        out, (h, c) = self.rnn(batch["notes"])

        # NOTE: 一旦local_featuresを無効にしてみる
        # local_weights = self.local_filter(out.transpose(1, 2))
        # local_weights = F.softmax(local_weights, dim=1).mean(dim=1)
        # local_features = (out * local_weights.unsqueeze(dim=2)).sum(dim=1)

        # FIXME: RNNのlayer数が1前提
        n_layer, batch_size, hidden_dim = h.size()
        global_features = h.transpose(0, 1).reshape((batch_size, n_layer * hidden_dim))

        # NOTE: 一旦local_featuresを無効にしてみる
        # local_features = torch.zeros_like(local_features, dtype=torch.float, device=local_features.device)
        # z = F.tanh(torch.cat([global_features, local_features], dim=1))
        z = F.tanh(global_features)
        mean = self.linear_mean(z)
        lnvar = self.linear_lnvar(z)
        std = torch.exp(0.5 * lnvar)
        eps = torch.randn_like(std)
        latent = mean + eps * std
        return latent


class Decoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, n_steps_per_measure: int = 16) -> None:
        super().__init__()
        self.n_layers = 1
        self.n_steps_per_measure = n_steps_per_measure
        self.measure_rnn = nn.LSTM(
            input_dim,
            hidden_dim,
            self.n_layers,
            batch_first=True,
            bidirectional=False
        )
        self.beat_rnn = nn.LSTM(
            hidden_dim + output_dim,
            hidden_dim,
            self.n_layers,
            batch_first=True,
            bidirectional=False
        )
        self.chord_embedding = nn.Embedding(12, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

    def _predicted_inputs(self, batch_size: int, previous_outputs: list[torch.Tensor], current_output: torch.Tensor | None = None, predict_index: int | None = None) -> torch.Tensor:
        predicted_inputs = torch.zeros((batch_size, self.n_steps_per_measure, self.output_dim), dtype=torch.float)
        # 2小節目以降は前の小節の最後のnoteを入力にする
        if len(previous_outputs) > 0:
            predicted_inputs[:, 0, :] = torch.zeros_like(previous_outputs[-1][:, -1, :]).scatter_(1, previous_outputs[-1][:, -1, :].argmax(dim=1).unsqueeze(1), 1.)

        # 直近の推論結果を次の推論の入力に活かす
        if current_output is not None:
            predicted_inputs[:, 1:predict_index + 2, :] = torch.zeros_like(current_output[:, 0: predict_index + 1, :]).scatter_(2, current_output[:, 0: predict_index + 1, :].argmax(dim=2).unsqueeze(2), 1.)

        return predicted_inputs

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = batch["latent"].size(dim=0)
        device = batch["latent"].device
        latents_per_measure, _ = self.measure_rnn(batch["latent"])
        latents_per_measure = F.tanh(latents_per_measure)
        latent_dim_per_measure = latents_per_measure.size(dim=2)

        previous_beat_h = torch.zeros((self.n_layers, batch_size, self.hidden_dim), dtype=torch.float, device=device)
        previous_beat_c = torch.zeros((self.n_layers, batch_size, self.hidden_dim), dtype=torch.float, device=device)
        outputs = []

        if "inputs" in batch:  # teacher forcing
            for i, (inputs_per_measure, latent) in enumerate(zip(batch["inputs"].transpose(0, 1), latents_per_measure.transpose(0, 1))):
                latent = latent.unsqueeze(1).repeat(1, self.n_steps_per_measure, 1)  # (batch_size, n_steps, latent_dim_per_measure)
                # chords tensor (batch_size, n_measure, n_steps, 12)
                if "chords" in batch:
                    logger.debug(f"batch_size: {batch_size}, beat_latent_dim: {latent_dim_per_measure}")
                    logger.debug(f"latents_per_measure size: {latents_per_measure.size()}")
                    logger.debug(f"latent size: {latent.size()}")
                    nonzero = (batch["chords"][:, i, :, :].reshape(batch_size * self.n_steps_per_measure, 12) == 1).nonzero()
                    logger.debug(f"nonzero size: {nonzero.size()}")
                    embedding = self.chord_embedding(nonzero[:, 1])
                    expand_index = nonzero[:, 0].repeat(latent_dim_per_measure, 1).transpose(0, 1)
                    logger.debug(f"exmapd_index size: {expand_index.size()}, embedding size: {embedding.size()}")
                    chord_filter = torch.zeros(batch_size * self.n_steps_per_measure, latent_dim_per_measure, dtype=torch.float, device=device).scatter_reduce(0, expand_index, embedding, reduce="sum")
                    chord_filter = F.tanh(chord_filter.reshape(batch_size, self.n_steps_per_measure, latent_dim_per_measure))
                    latent = latent * chord_filter

                x = torch.cat([latent, inputs_per_measure], dim=2)
                out, (previous_beat_h, previous_beat_c) = self.beat_rnn(x, (previous_beat_h, previous_beat_c))
                out = self.linear(F.tanh(out))
                outputs.append(out)
            return torch.cat(outputs, dim=1)
        else:
            # 潜在変数の長さで生成する小節数を判断する
            for i, latent in enumerate(latents_per_measure.transpose(0, 1)):
                latent = latent.unsqueeze(1).repeat(1, self.n_steps_per_measure, 1)  # (batch_size, n_steps, latent_dim_per_measure)
                # chords tensor (batch_size, n_measure, n_steps, 12)
                if "chords" in batch:
                    nonzero = (batch["chords"][:, i, :, :].reshape(batch_size * self.n_steps_per_measure, 12) == 1).nonzero()
                    embedding = self.chord_embedding(nonzero[:, 1])
                    expand_index = nonzero[:, 0].repeat(latent_dim_per_measure, 1).transpose(0, 1)
                    chord_filter = torch.zeros(batch_size * self.n_steps_per_measure, latent_dim_per_measure, dtype=torch.float, device=device).scatter_reduce(0, expand_index, embedding, reduce="sum")
                    chord_filter = F.tanh(chord_filter.reshape(batch_size, self.n_steps_per_measure, latent_dim_per_measure))
                    latent = latent * chord_filter

                predicted_inputs = self._predicted_inputs(batch_size, outputs).to(device)

                for j in range(self.n_steps_per_measure):
                    x = torch.cat([latent, predicted_inputs], dim=2)
                    out, (previous_beat_h, previous_beat_c) = self.beat_rnn(x, (previous_beat_h, previous_beat_c))
                    out = self.linear(F.tanh(out))
                    if j < self.n_steps_per_measure - 1:
                        predicted_inputs = self._predicted_inputs(batch_size, outputs, current_output=out, predict_index=i).to(device)

                outputs.append(out)
            return torch.cat(outputs, dim=1)


class MusicVaeModel(pl.LightningModule):

    def __init__(self, encoder_hidden_dim: int=512, decoder_hidden_dim: int=1024, latent_dim: int=32, n_steps_per_measure: int=16, mode: MODE="pretrain") -> None:
        super().__init__()
        self.mode = mode
        self.encoder = Encoder(129, latent_dim, encoder_hidden_dim)
        self.decoder = Decoder(latent_dim, 129, decoder_hidden_dim, n_steps_per_measure=n_steps_per_measure)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        latent = self.encoder(batch)
        batch["latent"] = latent.unsqueeze(1).repeat(1, batch["inputs"].size(dim=1), 1)
        out = self.decoder(batch)
        return out

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = self(batch)
        loss = self.criterion(out.transpose(1, 2), batch["labels"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {
            "loss": loss
        }

    def on_train_epoch_start(self) -> None:
        if self.mode == "finetune":
            if self.current_epoch < 2:
                logger.info(f"freeze pretrained")
                for param in self.encoder.parameters():
                    param.requires_grad = False
                for param in self.decoder.measure_rnn.parameters():
                    param.requires_grad = False
                for param in self.decoder.beat_rnn.parameters():
                    param.requires_grad = False
                for param in self.decoder.linear.parameters():
                    param.requires_grad = False
            else:
                for param in self.encoder.parameters():
                    param.requires_grad = True
                for param in self.decoder.measure_rnn.parameters():
                    param.requires_grad = True
                for param in self.decoder.beat_rnn.parameters():
                    param.requires_grad = True
                for param in self.decoder.linear.parameters():
                    param.requires_grad = True

    def configure_optimizers(self):
        if self.mode == "pretrain":
            warmup_epochs=20
            max_epochs=300
            warmup_start_lr=1e-5
            eta_min=1e-5
            lr=1e-3
        else:
            warmup_epochs=5
            max_epochs=50
            warmup_start_lr=1e-6
            eta_min=1e-6
            lr=1e-4

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
    melody_dim = 32
    latent_dim = 4
    hidden_dim = 2
    n_measures = 4
    n_steps_per_measure = 16
    encoder = Encoder(melody_dim, latent_dim, hidden_dim)
    decoder = Decoder(latent_dim, melody_dim, hidden_dim, n_steps_per_measure=n_steps_per_measure)
    inputs = F.one_hot(torch.randint(melody_dim, (n_measures, n_steps_per_measure)), num_classes=melody_dim).unsqueeze(0).float()
    notes = F.one_hot(torch.randint(melody_dim, (n_measures * n_steps_per_measure,)), num_classes=melody_dim).unsqueeze(0).float()
    chords = F.one_hot(torch.randint(12, (n_measures, n_steps_per_measure)), num_classes=12).unsqueeze(0).float()

    latent = encoder({"notes": notes})
    print(latent.size())

    rand_latent = torch.randn(1, latent_dim)
    print(rand_latent.size())

    out = decoder({
        "inputs": inputs,
        "latent": latent.unsqueeze(1).repeat(1, n_measures, 1)
    })
    print(out.size())

    rand_out = decoder({
        "latent": rand_latent.unsqueeze(1).repeat(1, n_measures, 1)
    })
    print(rand_out.size())

    out_with_chord = decoder({
        "inputs": inputs,
        "latent": latent.unsqueeze(1).repeat(1, n_measures, 1),
        "chords": chords,
    })
    print(out_with_chord.size())

    rand_out_with_chord = decoder({
        "latent": rand_latent.unsqueeze(1).repeat(1, n_measures, 1),
        "chords": chords,
    })
    print(rand_out_with_chord.size())

    decoder = Decoder(3, 5, 3, n_steps_per_measure=4)
    print(decoder._predicted_inputs(2, []))
    print(decoder._predicted_inputs(2, [
        torch.tensor([[[0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0.],
                       [0., 0.1, 0.2, 0.3, 0.]],
                      [[0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0.],
                       [0.1, 0., 0., 0., 0.]]], dtype=torch.float)
    ]))
    print(decoder._predicted_inputs(2, [
        torch.tensor([[[0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0.],
                       [0., 0.1, 0.2, 0.3, 0.]],
                      [[0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., 0.],
                       [0.1, 0., 0., 0., 0.]]], dtype=torch.float)
    ],
        current_output=torch.tensor([[[0., 0., 0.2, 0., 0.],
                                      [0., 0.2, 0., 0., 0.],
                                      [0.2, 0., 0., 0., 0.],
                                      [0., 0., 0., 0., 0.2]],
                                     [[0., 0.2, 0., 0., 0.],
                                      [0., 0., 0.2, 0., 0.],
                                      [0., 0., 0., 0.2, 0.],
                                      [0., 0., 0., 0., 0.2]]], dtype=torch.float),
        predict_index=1
    ))