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

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
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

        previous_beat_h = torch.zeros((self.n_layers, batch_size, self.hidden_dim), dtype=torch.float, device=device)
        previous_beat_c = torch.zeros((self.n_layers, batch_size, self.hidden_dim), dtype=torch.float, device=device)
        outputs = []

        if "inputs" in batch:
            for inputs_per_measure, latent in zip(batch["inputs"].transpose(0, 1), latents_per_measure.transpose(0, 1)):
                latent = latent.unsqueeze(1).repeat(1, self.n_steps_per_measure, 1)
                x = torch.cat([latent, inputs_per_measure], dim=2)
                out, (previous_beat_h, previous_beat_c) = self.beat_rnn(x, (previous_beat_h, previous_beat_c))
                out = self.linear(F.tanh(out))
                outputs.append(out)
            return torch.cat(outputs, dim=1)
        else:
            # 潜在変数の長さで生成する小節数を判断する
            for latent in latents_per_measure.transpose(0, 1):
                latent = latent.unsqueeze(1).repeat(1, self.n_steps_per_measure, 1)
                predicted_inputs = self._predicted_inputs(batch_size, outputs).to(device)

                for i in range(self.n_steps_per_measure):
                    x = torch.cat([latent, predicted_inputs], dim=2)
                    out, (previous_beat_h, previous_beat_c) = self.beat_rnn(x, (previous_beat_h, previous_beat_c))
                    out = self.linear(F.tanh(out))
                    if i < self.n_steps_per_measure - 1:
                        predicted_inputs = self._predicted_inputs(batch_size, outputs, current_output=out, predict_index=i).to(device)

                outputs.append(out)
            return torch.cat(outputs, dim=1)


class MusicVaeModel(pl.LightningModule):

    def __init__(self, hidden_dim: int=1024, latent_dim: int=32, n_steps_per_measure: int=16):
        super().__init__()
        self.encoder = Encoder(129, latent_dim, hidden_dim)
        self.decoder = Decoder(latent_dim, 129, hidden_dim, n_steps_per_measure=n_steps_per_measure)
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

    def configure_optimizers(self):
        return optim.Adam(
            self.parameters(),
            lr=1e-3
        )


if __name__ == '__main__':
    melody_dim = 32
    latent_dim = 4
    hidden_dim = 2
    n_measures = 4
    n_steps_per_measure = 16
    encoder = Encoder(melody_dim, latent_dim, hidden_dim)
    decoder = Decoder(latent_dim, melody_dim, hidden_dim, n_steps_per_measure=n_steps_per_measure)
    inputs = F.one_hot(torch.randint(melody_dim, (n_measures, n_steps_per_measure)), num_classes=melody_dim).unsqueeze(0).float()
    print(inputs.size())

    latent = encoder({"inputs": inputs})
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