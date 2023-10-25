from __future__ import annotations

import music21
import numpy as np


class Score:
    """楽譜データ"""

    def __init__(self, n_beats: int = 4, n_parts_of_beat: int = 4) -> None:
        self.n_beats = 4
        self.n_parts_of_beat = 4

    def _transpose(self, score: music21.stream.Score) -> music21.stream.Score:
        """メジャースケールの楽譜はCに，マイナースケールの楽譜はAに移調する

        Args:
            score (music21.stream.Score): 移調前のスコア

        Returns:
            music21.stream.Score: 移調後のスコア
        """
        key = score.analyze("key")
        if key.mode == "major":
            inter = music21.interval.Interval(key.tonic, music21.pitch.Pitch("C"))
        else:
            inter = music21.interval.Interval(key.tonic, music21.pitch.Pitch("A"))
        return score.transpose(inter)

    def _extract_notes(self, score: music21.stream.Score) -> list[list[music21.note.Note]]:
        """メロディの抽出

        Args:
            score (music21.stream.Score): スコア

        Returns:
            list[[music21.note.Note]]: 1小節ごとのメロディ列
        """
        notes = []
        # NOTE: channel 1のメロディのみ使用
        for measure in score.parts[0].getElementsByClass("Measure"):
            notes_per_measure = [None] * self.n_beats * self.n_parts_of_beat
            for note in measure.getElementsByClass("Note"):
                onset = note._activeSiteStoredOffset
                offset = onset + note._duration.quarterLength
                start_idx = int(onset * self.n_parts_of_beat)
                end_idx = int(offset * self.n_parts_of_beat) + 1
                end_idx = end_idx if end_idx < len(notes_per_measure) else len(notes_per_measure)
                for idx in range(start_idx, end_idx):
                    notes_per_measure[idx] = note
            notes.append(notes_per_measure)
        return notes

    def _extract_chords(self, score: music21.stream.Score) -> list[list[music21.harmony.ChordSymbol]]:
        """コードの抽出

        Args:
            score (music21.stream.Score): スコア

        Returns:
            list[[music21.harmony.ChordSymbol]]]: 1小節ごとのコード列
        """
        chords = []
        # NOTE: channel 1のコードのみ使用
        for measure in score.parts[0].getElementsByClass("Measure"):
            chords_per_measure = [None] * self.n_beats * self.n_parts_of_beat
            for chord in measure.getElementsByClass("ChordSymbol"):
                start_idx = int(chord.offset * self.n_parts_of_beat)
                end_idx = int(self.n_beats * self.n_parts_of_beat) + 1
                end_idx = end_idx if end_idx < len(chords_per_measure) else len(chords_per_measure)
                for idx in range(start_idx, end_idx):
                    chords_per_measure[idx] = chord
            chords.append(chords_per_measure)
        return chords

    @classmethod
    def load_musicxml(cls, file: str, n_beats: int = 4, n_parts_of_beat: int = 4) -> Score:
        instance = cls(n_beats=n_beats, n_parts_of_beat=n_parts_of_beat)
        score = music21.converter.parse(file)
        score = instance._transpose(score)
        instance.notes = instance._extract_notes(score)
        instance.chords = instance._extract_chords(score)
        return instance

    def get_notes(self) -> list[list[music21.note.Note]]:
        return self.notes

    def get_chords(self) -> list[list[music21.harmony.ChordSymbol]]:
        return self.chords


if __name__ == "__main__":
    score = Score.load_musicxml("data/input/openewld/OpenEWLD-0.1/dataset/Albert_Fitz-William_Penn/The_Honeysuckle_and_the_Bee/The_Honeysuckle_and_the_Bee.mxl")
    print(score.get_notes()[0])
    print(score.get_chords()[0])