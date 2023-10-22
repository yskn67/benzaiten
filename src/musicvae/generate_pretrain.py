import os

import hydra
import torch
import numpy as np
import mido
import midi2audio
from omegaconf import DictConfig, OmegaConf

from model import MusicVaeModel


# ピアノロール（one-hot vector列）をノートナンバー列に変換
def calc_notenums_from_pianoroll(pianoroll, min_note_number: int = 36):
    notenums = []
    for i in range(pianoroll.shape[0]):
        n = np.argmax(pianoroll[i, :])
        nn = -1 if n == pianoroll.shape[1] - 1 else n + min_note_number
        notenums.append(nn)
    return notenums


# 連続するノートナンバーを統合して (notenums, durations) に変換
def calc_durations(notenums):
    N = len(notenums)
    duration = [1] * N
    for i in range(N):
        k = 1
        while i + k < N:
            if notenums[i] > 0 and notenums[i] == notenums[i + k]:
                notenums[i + k] = 0
                duration[i] += 1
            else:
                break
            k += 1
    return notenums, duration


class MidiGenerator:
    def __init__(
        self,
        notenums,
        durations,
        transpose,
        ticks_per_beat: int = 480,
        n_beats: int = 4,
        n_parts_of_beat: int = 4
    ) -> None:
        self.notenums = notenums
        self.durations = durations
        self.transpose = transpose
        self.ticks_per_beat = ticks_per_beat
        self.n_beats = n_beats
        self.n_parts_of_beat = n_parts_of_beat

    def _make_midi_track(self, ticks_per_beat=None):
        if ticks_per_beat is None:
            ticks_per_beat = self.ticks_per_beat

        track = mido.MidiTrack()
        prev_tick = 0
        for i in range(len(self.notenums)):
            if self.notenums[i] > 0:
                curr_tick = int(i * ticks_per_beat / self.n_parts_of_beat)
                track.append(mido.Message('note_on', note=self.notenums[i] + self.transpose,
                                          velocity=127, time=curr_tick - prev_tick))
                prev_tick = curr_tick
                curr_tick = int((i + self.durations[i]) * ticks_per_beat / self.n_parts_of_beat)
                track.append(mido.Message('note_off', note=self.notenums[i] + self.transpose,
                                          velocity=127, time=curr_tick - prev_tick))
                prev_tick = curr_tick
        return track

    def _replace_prog_chg(self, midi, melody_ch: int = 0, melody_prog_chg: int = 73):
        for track in midi.tracks:
            for msg in track:
                if msg.type == 'program_change' and msg.channel == melody_ch:
                    msg.program = melody_prog_chg

    def make_midi_for_check(self, dst_filename):
        midi = mido.MidiFile(type=1)
        midi.ticks_per_beat = self.ticks_per_beat
        midi.tracks.append(self._make_midi_track())
        midi.save(dst_filename)


@hydra.main(version_base=None, config_path="../../conf/musicvae", config_name="config")
def main(cfg: DictConfig) -> None:
    os.makedirs(cfg.generate.output_dir, exist_ok=True)
    midi_file_path = os.path.join(cfg.generate.output_dir, cfg.generate.midi_file)
    wav_file_path = os.path.join(cfg.generate.output_dir, cfg.generate.wav_file)

    artifact_path = os.path.join(cfg.generate.output_dir, cfg.name)
    checkpoint_path = os.path.join(artifact_path, "best_model.ckpt")
    pretrain_cfg = OmegaConf.load(os.path.join(artifact_path, "config.yaml"))
    model = MusicVaeModel.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
        encoder_hidden_dim=pretrain_cfg.model.encoder_hidden_dim,
        decoder_hidden_dim=pretrain_cfg.model.decoder_hidden_dim,
        latent_dim=pretrain_cfg.model.latent_dim,
        n_steps_per_measure=pretrain_cfg.data.n_beats * cfg.data.n_parts_of_beat,
    )
    model.eval()

    with torch.no_grad():
        batch = {
            "latent": torch.randn(1, cfg.model.latent_dim).unsqueeze(1).repeat(1, 8, 1),
        }
        out = model.decoder(batch)

    pianoroll = out.squeeze(0).numpy()
    notenums = calc_notenums_from_pianoroll(pianoroll, min_note_number=0)
    notenums, durations = calc_durations(notenums)
    midi_generator = MidiGenerator(
        notenums,
        durations,
        12,
        ticks_per_beat=cfg.data.ticks_per_beat,
        n_beats=cfg.data.n_beats,
        n_parts_of_beat=cfg.data.n_parts_of_beat
    )

    midi_generator.make_midi_for_check(midi_file_path)
    fs = midi2audio.FluidSynth(sound_font="/usr/share/sounds/sf2/FluidR3_GM.sf2")
    fs.midi_to_audio(midi_file_path, wav_file_path)


if __name__ == "__main__":
    main()