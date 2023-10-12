from abc import ABCMeta, abstractmethod
import numpy as np
# from .score import Score
from score import Score


class TransformBase(metaclass=ABCMeta):

    @abstractmethod
    def transform(self, score: Score) -> dict[str, np.array]:
        pass


class TransformOnehot(TransformBase):

    def __init__(self, min_note_number: int = 36, max_note_number: int = 89) -> None:
        self.min_note_number = min_note_number
        self.max_note_number = max_note_number
        # 休符用に1次元増やす
        self.note_range = max_note_number - min_note_number + 1

    def _transform_notes(self, score: Score) -> list[np.array]:
        onehot_notes = []
        for bar_notes in score.notes:
            onehot_bar_notes = np.zeros((len(bar_notes), self.note_range), dtype=int)
            for i, note in enumerate(bar_notes):
                if note is None:
                    onehot_bar_notes[i, -1] = 1
                else:
                    note_idx = note.pitch.midi - self.min_note_number
                    onehot_bar_notes[i, note_idx] = 1
            onehot_notes.append(onehot_bar_notes)
        return onehot_notes

    def _transform_chords(self, score: Score) -> list[np.array]:
        manyhot_chords = []
        for bar_chords in score.chords:
            manyhot_bar_chords = np.zeros((len(bar_chords), 12), dtype=int)
            for i, chord in enumerate(bar_chords):
                if chord is not None:
                    for note in chord._notes:
                        manyhot_bar_chords[i, note.pitch.midi % 12] = 1
            manyhot_chords.append(manyhot_bar_chords)
        return manyhot_chords

    def _transform_labels(self, score: Score) -> list[np.array]:
        labels = []
        for bar_notes in score.notes:
            bar_labels = np.zeros(len(bar_notes), dtype=int)
            for i, note in enumerate(bar_notes):
                if note is None:
                    bar_labels[i] = self.note_range - 1
                else:
                    bar_labels[i] = note.pitch.midi - self.min_note_number
            labels.append(bar_labels)
        return labels

    def transform(self, score: Score) -> list[dict[str, np.array]]:
        return [
            {"notes": n, "chords": c, "labels": l} for n, c, l in zip(
                self._transform_notes(score),
                self._transform_chords(score),
                self._transform_labels(score)
            )
        ]