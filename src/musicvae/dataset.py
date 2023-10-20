import json

import torch
import torch.nn.functional as F


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


if __name__ == "__main__":
    dataset = MusicVaeDataset(files=["data/preprocess/maestro/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav_2.json"])
    print(dataset[0])