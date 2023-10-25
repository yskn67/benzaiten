import json
import os
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

from score import Score


class MelodyFixerDataset(torch.utils.data.Dataset):

    def __init__(self, files: list[str]) -> None:
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.tensor]:
        with open(self.files[idx], "rt") as f:
            pitchs_per_measure = json.load(f)

        # 休符(-1)は128番目に割り当てる
        notes = [128 if p == -1 else p for pitchs in pitchs_per_measure for p in pitchs]
        return {
            "notes": torch.tensor(notes, dtype=torch.long),
        }


class MelodyFixerFinetuneDataset(torch.utils.data.Dataset):

    def __init__(self, data) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.tensor]:
        notes_per_measure, chords_per_measure = self.data[idx]

        # chord
        manyhot_chords = []
        for measure_chords in chords_per_measure:
            manyhot_measure_chords = np.zeros((len(measure_chords), 12), dtype=int)
            for i, chord in enumerate(measure_chords):
                if chord is not None:
                    for note in chord._notes:
                        manyhot_measure_chords[i, note.pitch.midi % 12] = 1
            manyhot_chords.append(manyhot_measure_chords)

        # 休符(None)は128番目に割り当てる
        notes = [128 if n is None else n.pitch.midi for notes in notes_per_measure for n in notes]
        return {
            "notes": torch.tensor(notes, dtype=torch.long),
            "chords": torch.tensor(np.array(manyhot_chords, dtype=int), dtype=torch.long),
        }


def _postprocess(notes, n_parts_of_beat: int):
    # イントロは最後の四分音符分のみ使用
    notes[0][:n_parts_of_beat * 3] = [None] * (n_parts_of_beat * 3)
    # アウトロは最初の四分音符を伸ばす
    notes[-1][n_parts_of_beat * 1:n_parts_of_beat * 3] = [notes[-1][n_parts_of_beat * 1 - 1]] * (n_parts_of_beat * 2)
    notes[-1][n_parts_of_beat * 3:] = [None] * n_parts_of_beat
    return notes


def _process_score(fpath: str, n_bar: int, n_beats: int, n_parts_of_beat: int) -> list[tuple]:
    data = []
    score = Score.load_musicxml(fpath, n_beats=n_beats, n_parts_of_beat=n_parts_of_beat)
    notes_per_measure = score.get_notes()
    chords_per_measure = score.get_chords()
    for i in range(0, len(notes_per_measure), n_bar):
        if i + n_bar + 1 > len(notes_per_measure):
            continue

        if i == 0:
            notes = [[None] * (n_beats * n_parts_of_beat)] + notes_per_measure[i: i + n_bar + 1]
            chords = [[chords_per_measure[0][0]] * (n_beats * n_parts_of_beat)] + chords_per_measure[i: i + n_bar + 1]
        else:
            notes = notes_per_measure[i - 1: i + n_bar + 1]
            chords = chords_per_measure[i - 1: i + n_bar + 1]

        if len(notes) != len(chords):
            continue

        notes = _postprocess(notes, n_parts_of_beat=n_parts_of_beat)
        data.append((notes, chords))

    return data


def make_finetune_dataset(flist: list[str], n_bar: int = 8, n_beats: int = 4, n_parts_of_beat: int = 4) -> torch.utils.data.Dataset:
    args = [(fpath, n_bar, n_beats, n_parts_of_beat) for fpath in flist]
    with Pool(processes=max(os.cpu_count() - 1, 1)) as pool:
        results = pool.starmap(_process_score, args)

    data = [r for results_per_process in results for r in results_per_process]
    return MelodyFixerFinetuneDataset(data)


if __name__ == "__main__":
    dataset = MelodyFixerDataset(files=["data/preprocess_fixer/maestro/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav_0.json"])
    print(dataset[0])
    print(dataset[0]["notes"].size())
    d = make_finetune_dataset(["data/input/openewld/OpenEWLD-0.1/dataset/Albert_Fitz-William_Penn/The_Honeysuckle_and_the_Bee/The_Honeysuckle_and_the_Bee.mxl"])[0]
    print(d)
    print(d["notes"].size())
    print(d["chords"].size())