import os
import csv

import torch
import numpy as np
import music21
import mido
import midi2audio

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

NOTENUM_FROM = 36
TICKS_PER_BEAT = 480
MELODY_PROG_CHG = 73
MELODY_CH = 0
INTRO_BLANK_MEASURES = 4
N_BEATS = 4
BEAT_RESO = 4

# ピアノロール（one-hot vector列）をノートナンバー列に変換
def calc_notenums_from_pianoroll(pianoroll):
    notenums = []
    for i in range(pianoroll.shape[0]):
        n = np.argmax(pianoroll[i, :])
        nn = -1 if n == pianoroll.shape[1] - 1 else n + NOTENUM_FROM
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
def replace_prog_chg(midi):
    for track in midi.tracks:
        for msg in track:
            if msg.type == 'program_change' and msg.channel == MELODY_CH:
                msg.program = MELODY_PROG_CHG

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

####### 2023.08.04 修正
# ピアノロールを描画し、MIDIファイルを再生
def show_and_play_midi(pianoroll, transpose, src_filename, dst_filename1, dst_filename2):
    notenums = calc_notenums_from_pianoroll(pianoroll)
    notenums, durations = calc_durations(notenums)
    ###### 2023.08.04 変更
    make_midi_for_submission(notenums, durations, transpose, dst_filename1)
    make_midi_for_check(notenums, durations, transpose, src_filename, dst_filename2)
    fs = midi2audio.FluidSynth(sound_font="/usr/share/sounds/sf2/FluidR3_GM.sf2")
    fs.midi_to_audio(dst_filename2, "output.wav")


# @hydra.main(version_base=None, config_path="../conf", config_name="config")
# def main(cfg: DictConfig) -> None:
def main() -> None:
    basedir = "../data/input/origin/"
    #backing_file = "sample1_backing.mid"       # 適宜変更すること
    #chord_file = "sample1_chord.csv"           # 適宜変更すること
    backing_file = "sample5_backing.mid"       # 適宜変更すること
    chord_file = "sample5_chord.csv"           # 適宜変更すること

    # 2023.08.04 変更
    # output_file1 = "output1.mid"                # 自分のエントリーネームに変更すること
    # output_file2 = "output2.mid"
    output_file1 = "output_sample5_7.mid"                # 自分のエントリーネームに変更すること
    output_file2 = "output_sample5_7_full.mid"

    chords = read_chord_file(os.path.join(basedir, chord_file))
    manyhot_chords = TransformOnehotInference().transform(chords)

    model = Seq2SeqMelodyGenerationModel.load_from_checkpoint("../lightning_logs/version_5/checkpoints/epoch=99-step=5400.ckpt")
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

    show_and_play_midi(pianoroll, 12, basedir + backing_file,
                    basedir + output_file1, basedir + output_file2)


if __name__ == "__main__":
    main()