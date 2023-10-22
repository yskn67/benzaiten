import json
import os
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

from score import Score


class MusicVaeDataset(torch.utils.data.Dataset):

    def __init__(self, files: list[str]) -> None:
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.tensor]:
        with open(self.files[idx], "rt") as f:
            pitchs_per_measure = json.load(f)

        # 休符(-1)は128番目に割り当てる
        pitchs_per_measure = [[128 if p == -1 else p for p in pitchs] for pitchs in pitchs_per_measure]
        # 一つ前のpitchを入力とする
        input_pitchs_per_measure = []
        previous_pitch = 128
        for pitchs in pitchs_per_measure:
            input_pitchs = [previous_pitch] + pitchs[:-1]
            input_pitchs_per_measure.append(input_pitchs)
            previous_pitch = pitchs[-1]

        pitchs = [p for pitchs in pitchs_per_measure for p in pitchs]
        return {
            "inputs": F.one_hot(torch.tensor(input_pitchs_per_measure, dtype=torch.long), num_classes=129).float(),
            "notes": F.one_hot(torch.tensor(pitchs, dtype=torch.long), num_classes=129).float(),
            "labels": torch.tensor(pitchs, dtype=torch.long),
        }


class MusicVaeFinetuneDataset(torch.utils.data.Dataset):

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
        pitchs_per_measure = [[128 if n is None else n.pitch.midi for n in notes] for notes in notes_per_measure]
        # 一つ前のpitchを入力とする
        input_pitchs_per_measure = []
        previous_pitch = 128
        for pitchs in pitchs_per_measure:
            input_pitchs = [previous_pitch] + pitchs[:-1]
            input_pitchs_per_measure.append(input_pitchs)
            previous_pitch = pitchs[-1]

        pitchs = [p for pitchs in pitchs_per_measure for p in pitchs]
        return {
            "inputs": F.one_hot(torch.tensor(input_pitchs_per_measure, dtype=torch.long), num_classes=129).float(),
            "notes": F.one_hot(torch.tensor(pitchs, dtype=torch.long), num_classes=129).float(),
            "chords": torch.tensor(np.array(manyhot_chords, dtype=int), dtype=torch.long),
            "labels": torch.tensor(pitchs, dtype=torch.long),
        }


def each_slice(lst, n_slice):
    s = 0
    n = len(lst)
    while s < n:
        yield lst[s:s + n_slice]
        s += n_slice


def _process_score(fpath: str, n_bar: int, n_beats: int, n_parts_of_beat: int) -> list[tuple]:
    data = []
    score = Score.load_musicxml(fpath, n_beats=n_beats, n_parts_of_beat=n_parts_of_beat)
    for notes, chords in zip(each_slice(score.get_notes(), n_bar), each_slice(score.get_chords(), n_bar)):
        if len(notes) != n_bar or len(chords) != n_bar:
            continue

        data.append((notes, chords))

    return data


def make_finetune_dataset(flist: list[str], n_bar: int = 8, n_beats: int = 4, n_parts_of_beat: int = 4) -> torch.utils.data.Dataset:
    args = [(fpath, n_bar, n_beats, n_parts_of_beat) for fpath in flist]
    with Pool(processes=max(os.cpu_count() - 1, 1)) as pool:
        results = pool.starmap(_process_score, args)

    data = [r for results_per_process in results for r in results_per_process]
    return MusicVaeFinetuneDataset(data)


if __name__ == "__main__":
    dataset = MusicVaeDataset(files=["data/preprocess/maestro/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav_2.json"])
    print(dataset[0])
    d = make_finetune_dataset(["data/input/openewld/OpenEWLD-0.1/dataset/Albert_Fitz-William_Penn/The_Honeysuckle_and_the_Bee/The_Honeysuckle_and_the_Bee.mxl"])[0]
    print(d)
    print(d["inputs"].size())
    print(d["notes"].size())
    print(d["chords"].size())
    print(d["labels"].size())