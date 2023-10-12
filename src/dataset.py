import numpy as np
import torch

from score import Score
from transform import TransformOnehot


class MusicDataset(torch.utils.data.Dataset):

    def __init__(self, data) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.tensor]:
        return {
            "notes": torch.tensor(self.data[idx]["notes"], dtype=torch.float),
            "chords": torch.tensor(self.data[idx]["chords"], dtype=torch.float),
            "labels": torch.tensor(self.data[idx]["labels"], dtype=torch.long),
        }

def make_onehot_dataset(flist: list[str], n_bar: int = 2, min_note_number: int = 36, max_note_number: int = 89) -> torch.utils.data.Dataset:
    data = []
    for fpath in flist:
        score = Score.load_musicxml(fpath)
        trnasform = TransformOnehot(min_note_number=min_note_number, max_note_number=max_note_number)
        d = trnasform.transform(score)
        for i in range(len(d) - n_bar):
            d2 = {}
            for d3 in d[i: i + n_bar]:
                for k, v in d3.items():
                    if k not in d2:
                        d2[k] = v
                    else:
                        d2[k] = np.concatenate([d2[k], v], axis=0)
            data.append(d2)
    return MusicDataset(data)