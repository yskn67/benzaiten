import os
import csv

import hydra
import torch
import numpy as np
import music21
import mido
import midi2audio
from omegaconf import DictConfig

from model import Seq2SeqMelodyGenerationModel


def read_chord_file(file: str, n_beats: int = 4, n_parts_of_beat: int = 4) -> list[list[music21.harmony.ChordSymbol]]:
    csv_data = {}  # 小節ごとに
    with open(file) as f:
        reader = csv.reader(f)
        for row in reader:
            m = int(row[0])
            if m not in csv_data:
                csv_data[m] = []

            csv_data[m].append(row)

    chords = []
    for m, rows in csv_data.items():
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


class TransformOnehotInference:

    def __init__(self, min_note_number: int = 36, max_note_number: int = 89) -> None:
        self.min_note_number = min_note_number
        self.max_note_number = max_note_number
        # 休符用に1次元増やす
        self.note_range = max_note_number - min_note_number + 1

    def transform(self, chords) -> list[np.array]:
        manyhot_chords = []
        for bar_chords in chords:
            manyhot_bar_chords = np.zeros((len(bar_chords), 12), dtype=int)
            for i, chord in enumerate(bar_chords):
                if chord is not None:
                    for note in chord._notes:
                        manyhot_bar_chords[i, note.pitch.midi % 12] = 1
            manyhot_chords.append(manyhot_bar_chords)
        return manyhot_chords

TICKS_PER_BEAT = 480
INTRO_BLANK_MEASURES = 4
N_BEATS = 4
BEAT_RESO = 4

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

####### 2023.08.04 追加
# MIDIトラックを生成（make_midiから呼び出される）
def make_midi_track(notenums, durations, transpose, ticks_per_beat):
    track = mido.MidiTrack()
    init_tick = INTRO_BLANK_MEASURES * N_BEATS * ticks_per_beat
    prev_tick = 0
    for i in range(len(notenums)):
        if notenums[i] > 0:
            curr_tick = int(i * ticks_per_beat / BEAT_RESO) + init_tick
            track.append(mido.Message('note_on', note=notenums[i]+transpose,
                                      velocity=100, time=curr_tick - prev_tick))
            prev_tick = curr_tick
            curr_tick = int((i + durations[i]) * ticks_per_beat / BEAT_RESO) + init_tick
            track.append(mido.Message('note_off', note=notenums[i]+transpose,
                                      velocity=100, time=curr_tick - prev_tick))
            prev_tick = curr_tick
    return track

####### 2023.08.04 追加
# プログラムチェンジを指定したものに差し替え
def replace_prog_chg(midi, melody_ch: int = 0, melody_prog_chg: int = 73):
    for track in midi.tracks:
        for msg in track:
            if msg.type == 'program_change' and msg.channel == melody_ch:
                msg.program = melody_prog_chg

####### 2023.08.04 追加
# MIDIファイル（提出用、伴奏なし）を生成
def make_midi_for_submission(notenums, durations, transpose, dst_filename):
    midi = mido.MidiFile(type=1)
    midi.ticks_per_beat = TICKS_PER_BEAT
    midi.tracks.append(make_midi_track(notenums, durations, transpose, TICKS_PER_BEAT))
    midi.save(dst_filename)

####### 2023.08.04 修正
# MIDIファイル（チェック用、伴奏あり）を生成
def make_midi_for_check(notenums, durations, transpose, src_filename, dst_filename):
    midi = mido.MidiFile(src_filename)
    replace_prog_chg(midi)
    midi.tracks.append(make_midi_track(notenums, durations, transpose, midi.ticks_per_beat))
    midi.save(dst_filename)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    chord_file_path = os.path.join(cfg.generate.input_dir, cfg.generate.chord_file)
    backing_file_path = os.path.join(cfg.generate.input_dir, cfg.generate.backing_file)
    submission_midi_file_path = os.path.join(cfg.generate.output_dir, cfg.generate.submission_midi_file)
    full_midi_file_path = os.path.join(cfg.generate.output_dir, cfg.generate.full_midi_file)
    wav_file_path = os.path.join(cfg.generate.output_dir, "output.wav")

    chords = read_chord_file(chord_file_path, n_beats=cfg.data.n_beats, n_parts_of_beat=cfg.data.n_parts_of_beat)
    manyhot_chords = TransformOnehotInference().transform(chords)

    model = Seq2SeqMelodyGenerationModel.load_from_checkpoint("lightning_logs/version_5/checkpoints/epoch=99-step=5400.ckpt")
    model.eval()

    with torch.no_grad():
        latent = torch.randn(1, 32)
        h_n, c_n = (None, None)
        outs = []
        for chords_per_measure in manyhot_chords:
            batch = {
                "latent": latent,
                "chords": torch.tensor(chords_per_measure, dtype=torch.float).unsqueeze(0)
            }
            if h_n is not None:
                batch['h'] = h_n
            if c_n is not None:
                batch['c'] = c_n
            out, h_n, c_n = model.decoder(batch)
            outs.append(out.squeeze(0))
        out = torch.cat(outs, dim=0)

    pianoroll = out.numpy()
    notenums = calc_notenums_from_pianoroll(pianoroll, min_note_number=cfg.data.min_note_number)
    notenums, durations = calc_durations(notenums)
    make_midi_for_submission(notenums, durations, 12, submission_midi_file_path)
    make_midi_for_check(notenums, durations, 12, backing_file_path, full_midi_file_path)
    fs = midi2audio.FluidSynth(sound_font="/usr/share/sounds/sf2/FluidR3_GM.sf2")
    fs.midi_to_audio(full_midi_file_path, wav_file_path)


if __name__ == "__main__":
    main()