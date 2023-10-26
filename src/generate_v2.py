import csv
import os
import sys

import hydra
import torch
import numpy as np
import music21
import mido
import midi2audio
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from musicvae.model import MusicVaeModel
from melodyfixer.model import MelodyFixerModel


MELODY_CH = 0


def read_chord_file(file: str, n_beats: int = 4, n_parts_of_beat: int = 4) -> list[list[music21.harmony.ChordSymbol]]:
    csv_data = {}  # 小節ごとに
    with open(file) as f:
        reader = csv.reader(f)
        for row in reader:
            m = int(row[0])
            if m not in csv_data:
                csv_data[m] = []

            csv_data[m].append(row)

    # イントロ用
    # イントロの小節は1小節目の最初のコードを利用する
    csv_data[-1]= [["-1"] + row[1:] for row in csv_data[0]]

    chords = []
    for m, rows in sorted(csv_data.items(), key=lambda x: x[0]):
        chords_per_measure = [None] * n_beats * n_parts_of_beat
        for row in rows:
            b = int(row[1]) # 拍番号（0始まり、今回は0または2）
            chords_per_measure[n_beats * b] = music21.harmony.ChordSymbol(
                root=row[2],
                kind=row[3],
                bass=row[4]
            )
        for i in range(len(chords_per_measure)):
            if chords_per_measure[i] != None:
                chord = chords_per_measure[i]
            else:
                chords_per_measure[i] = chord
        chords.append(chords_per_measure)
    return chords


def transform_manyhot_chords(chords: list[list[music21.harmony.ChordSymbol]]) -> np.array:
    manyhot_chords = []
    for bar_chords in chords:
        manyhot_bar_chords = np.zeros((len(bar_chords), 12), dtype=int)
        for i, chord in enumerate(bar_chords):
            if chord is not None:
                for note in chord._notes:
                    manyhot_bar_chords[i, note.pitch.midi % 12] = 1
        manyhot_chords.append(manyhot_bar_chords)
    return np.array(manyhot_chords, dtype=int)


def postprocess(out: torch.Tensor, n_parts_of_beat: int = 4) -> torch.Tensor:
    """
    モデルの出力の後処理
    イントロは最後の四分音符分だけ出力し，それ以外を休符とする
    アウトロは最初の四分音符分の最後の音を四分音符2つ分のばす
    """
    # イントロ処理
    out[:n_parts_of_beat * 3, :] = 0.
    out[:n_parts_of_beat * 3, -1] = 1.

    # アウトロ処理
    padding_melody = out[-n_parts_of_beat * 3 - 1, :]
    out[-n_parts_of_beat * 3:-n_parts_of_beat * 1, :] = padding_melody.unsqueeze(0).repeat(n_parts_of_beat * 2, 1)
    out[-n_parts_of_beat * 1:, :] = 0.
    out[-n_parts_of_beat * 1:, -1] = 1.

    return out


def calc_notenums_from_pianoroll(pianoroll, min_note_number: int = 36):
    """
    ピアノロール（one-hot vector列）をノートナンバー列に変換
    """
    notenums = []
    for i in range(pianoroll.shape[0]):
        n = np.argmax(pianoroll[i, :])
        nn = -1 if n == pianoroll.shape[1] - 1 else n + min_note_number
        notenums.append(nn)
    return notenums


def calc_durations(notenums):
    """
    連続するノートナンバーを統合して (notenums, durations) に変換
    """
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
        intro_blank_measures: int = 4,
        n_beats: int = 4,
        n_parts_of_beat: int = 4
    ) -> None:
        self.notenums = notenums
        self.durations = durations
        self.transpose = transpose
        self.ticks_per_beat = ticks_per_beat
        self.intro_blank_measures = intro_blank_measures
        self.n_beats = n_beats
        self.n_parts_of_beat = n_parts_of_beat

    def _make_midi_track(self, ticks_per_beat=None):
        if ticks_per_beat is None:
            ticks_per_beat = self.ticks_per_beat

        track = mido.MidiTrack()
        # Logic Proにインポートしたときに空白小節がトリミングされないように、
        # ダミーのチャンネルメッセージとして、オール・ノート・オフを挿入
        track.append(mido.Message('control_change', channel=MELODY_CH, control=123, value=0))
        init_tick = self.intro_blank_measures * self.n_beats * ticks_per_beat
        prev_tick = 0
        for i in range(len(self.notenums)):
            if self.notenums[i] > 0:
                curr_tick = int(i * ticks_per_beat / self.n_parts_of_beat) + init_tick
                track.append(mido.Message('note_on', channel=MELODY_CH, note=self.notenums[i] + self.transpose,
                                          velocity=127, time=curr_tick - prev_tick))
                prev_tick = curr_tick
                curr_tick = int((i + self.durations[i]) * ticks_per_beat / self.n_parts_of_beat) + init_tick
                track.append(mido.Message('note_off', channel=MELODY_CH, note=self.notenums[i] + self.transpose,
                                          velocity=127, time=curr_tick - prev_tick))
                prev_tick = curr_tick
        return track

    def _replace_prog_chg(self, midi, melody_ch: int = 0, melody_prog_chg: int = 73):
        for track in midi.tracks:
            for msg in track:
                if msg.type == 'program_change' and msg.channel == melody_ch:
                    msg.program = melody_prog_chg

    def make_midi_for_submission(self, dst_filename):
        midi = mido.MidiFile(type=1)
        midi.ticks_per_beat = self.ticks_per_beat
        midi.tracks.append(self._make_midi_track())
        midi.save(dst_filename)

    def make_midi_for_check(self, src_filename, dst_filename):
        midi = mido.MidiFile(src_filename)
        self._replace_prog_chg(midi)
        midi.tracks.append(self._make_midi_track(ticks_per_beat=midi.ticks_per_beat))
        midi.save(dst_filename)


@hydra.main(version_base=None, config_path="../conf", config_name="config_v2")
def main(cfg: DictConfig) -> None:
    logger.remove()
    logger.add(sys.stderr, level=cfg.logger.level)

    os.makedirs(cfg.data.output_dir, exist_ok=True)
    chord_file_path = os.path.join(cfg.data.input_dir, cfg.data.chord_file)
    backing_file_path = os.path.join(cfg.data.input_dir, cfg.data.backing_file)
    submission_midi_file_path = os.path.join(cfg.data.output_dir, cfg.data.submission_midi_file)
    full_midi_file_path = os.path.join(cfg.data.output_dir, cfg.data.full_midi_file)
    wav_file_path = os.path.join(cfg.data.output_dir, cfg.data.wav_file)

    musicvae_artifact_path = os.path.join(cfg.model.musicvae_path, cfg.model.musicvae_name)
    musicvae_pretrain_cfg = OmegaConf.load(os.path.join(musicvae_artifact_path, "config.pretrain.yaml"))
    musicvae_finetune_cfg = OmegaConf.load(os.path.join(musicvae_artifact_path, "config.finetune.yaml"))
    musicvae_checkpoint_path = os.path.join(musicvae_artifact_path, "best_model.ckpt")

    melodyfixer_artifact_path = os.path.join(cfg.model.melodyfixer_path, cfg.model.melodyfixer_name)
    melodyfixer_pretrain_cfg = OmegaConf.load(os.path.join(melodyfixer_artifact_path, "config.pretrain.yaml"))
    melodyfixer_finetune_cfg = OmegaConf.load(os.path.join(melodyfixer_artifact_path, "config.finetune.yaml"))
    melodyfixer_checkpoint_path = os.path.join(melodyfixer_artifact_path, "best_model.ckpt")

    musicvae = MusicVaeModel.load_from_checkpoint(
        musicvae_checkpoint_path,
        map_location="cpu",
        mode=musicvae_finetune_cfg.model.mode,
        encoder_hidden_dim=musicvae_pretrain_cfg.model.encoder_hidden_dim,
        decoder_hidden_dim=musicvae_pretrain_cfg.model.decoder_hidden_dim,
        latent_dim=musicvae_pretrain_cfg.model.latent_dim,
        n_steps_per_measure=musicvae_pretrain_cfg.data.n_beats * musicvae_pretrain_cfg.data.n_parts_of_beat,
    )
    musicvae.eval()

    melodyfixer = MelodyFixerModel.load_from_checkpoint(
        melodyfixer_checkpoint_path,
        map_location="cpu",
        mode=melodyfixer_finetune_cfg.model.mode,
        hidden_dim=melodyfixer_pretrain_cfg.model.hidden_dim,
        output_dim=129,
        n_measures=melodyfixer_pretrain_cfg.data.n_measures + 2,
        n_steps_per_measure=melodyfixer_pretrain_cfg.data.n_beats * melodyfixer_pretrain_cfg.data.n_parts_of_beat,
        n_mask=cfg.generate.n_mask,
    )
    melodyfixer.eval()

    chords = read_chord_file(
        chord_file_path,
        n_beats=musicvae_pretrain_cfg.data.n_beats,
        n_parts_of_beat=musicvae_pretrain_cfg.data.n_parts_of_beat
    )
    manyhot_chords = transform_manyhot_chords(chords)
    logger.debug(f"manyhot_chords size: {manyhot_chords.shape}")

    with torch.no_grad():
        batch = {
            "latent": torch.randn(1, musicvae_pretrain_cfg.model.latent_dim).unsqueeze(1).repeat(1, 10, 1),
            "chords": torch.tensor(manyhot_chords, dtype=torch.long).unsqueeze(0),
        }
        logger.debug(f"latent size: {batch['latent'].size()}")
        logger.debug(f"chords size: {batch['chords'].size()}")
        out = musicvae.decoder(batch)
        logger.debug(f"model output size: {out.size()}")
        if cfg.generate.use_melodyfixer:
            batch = {
                "notes": out.argmax(dim=2),
                "chords": torch.tensor(manyhot_chords, dtype=torch.long).unsqueeze(0),
            }
            out = melodyfixer.forward(batch)

    if cfg.generate.use_melodyfixer:
        out = out.squeeze(0)
    else:
        out = postprocess(out.squeeze(0), n_parts_of_beat=musicvae_pretrain_cfg.data.n_parts_of_beat)

    pianoroll = out.numpy()
    notenums = calc_notenums_from_pianoroll(pianoroll, min_note_number=0)
    notenums, durations = calc_durations(notenums)
    midi_generator = MidiGenerator(
        notenums,
        durations,
        12,
        ticks_per_beat=musicvae_pretrain_cfg.data.ticks_per_beat,
        intro_blank_measures=musicvae_pretrain_cfg.data.intro_blank_measures,
        n_beats=musicvae_pretrain_cfg.data.n_beats,
        n_parts_of_beat=musicvae_pretrain_cfg.data.n_parts_of_beat
    )

    midi_generator.make_midi_for_submission(submission_midi_file_path)
    midi_generator.make_midi_for_check(backing_file_path, full_midi_file_path)
    fs = midi2audio.FluidSynth(sound_font="/usr/share/sounds/sf2/FluidR3_GM.sf2")
    fs.midi_to_audio(full_midi_file_path, wav_file_path)


if __name__ == "__main__":
    main()