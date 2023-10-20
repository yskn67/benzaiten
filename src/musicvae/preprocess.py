import functools
import json
import os
from glob import glob
from multiprocessing import Pool
from typing import Callable

import hydra
import note_seq
from loguru import logger
from omegaconf import DictConfig


def each_slice(lst, n_slice):
    s = 0
    n = len(lst)
    while s < n:
        yield lst[s:s + n_slice]
        s += n_slice


def preprocess(
    midi_path: str,
    output_dir: str,
    target_n_measures: int = 4,
    target_steps_per_bar: int = 16,
    target_steps_per_quarter: int = 4,
) -> None:
    midi_dirname, midi_basename = os.path.split(midi_path)
    _, midi_parentname = os.path.split(midi_dirname)
    midi_basename_root, _ = os.path.splitext(midi_basename)
    try:
        melody = note_seq.midi_file_to_melody(midi_path)
        # stepsが想定外の場合は処理しない
        if melody.steps_per_bar != target_steps_per_bar or melody.steps_per_quarter != target_steps_per_quarter:
            logger.debug(f"Unexpected steps: {midi_path}")
            return

        melody.squash(0, 128, transpose_to_key=0)
        quantized_seq = note_seq.quantize_note_sequence(melody.to_sequence(), target_steps_per_quarter)
        # 4/4拍子以外は処理しない
        ts = quantized_seq.time_signatures[0]
        if ts.numerator != 4 or ts.numerator != 4:
            logger.debug(f"Not 4/4: {midi_path}")
            return

        n_steps = quantized_seq.total_quantized_steps
        n_measures = n_steps // target_steps_per_bar
        pitchs_per_measure = [[-1] * 16 for _ in range(n_measures)]  # 休符,note_offは-1とする
        for note in quantized_seq.notes:
            for i in range(note.quantized_start_step, note.quantized_end_step):
                if i >= n_measures * target_steps_per_bar:
                    continue
                measure_idx = i // target_steps_per_bar
                step_idx = i % target_steps_per_bar
                pitchs_per_measure[measure_idx][step_idx] = note.pitch
        for i, pitchs_per_n_measure in enumerate(each_slice(pitchs_per_measure, target_n_measures)):
            # 規定の小節数に満たない場合保存しない
            if len(pitchs_per_n_measure) < target_n_measures:
                logger.debug(f"skip because of few length {len(pitchs_per_n_measure)}: {midi_path}")
                continue
            # 半数以上を休符で占める小節が存在する場合は保存しない
            n_rests_per_measure = [sum([1 if p == -1 else 0 for p in pitchs]) for pitchs in pitchs_per_n_measure]
            if any([n_rests >= target_steps_per_bar / 2 for n_rests in n_rests_per_measure]):
                logger.debug(f"skip because of too many rest: {midi_path}")
                continue

            # 規定の小節数ごとに保存
            output_path = os.path.join(output_dir, midi_parentname, midi_basename_root + f"_{i}" + ".json")
            with open(output_path, "wt") as f:
                json.dump(pitchs_per_n_measure, f)
    except Exception as e:
        logger.debug(f"{e} : {midi_path}")


def maestro(preprocess_fn: Callable) -> None:
    os.makedirs("data/preprocess/maestro/2004", exist_ok=True)
    os.makedirs("data/preprocess/maestro/2006", exist_ok=True)
    os.makedirs("data/preprocess/maestro/2008", exist_ok=True)
    os.makedirs("data/preprocess/maestro/2009", exist_ok=True)
    os.makedirs("data/preprocess/maestro/2011", exist_ok=True)
    os.makedirs("data/preprocess/maestro/2013", exist_ok=True)
    os.makedirs("data/preprocess/maestro/2014", exist_ok=True)
    os.makedirs("data/preprocess/maestro/2015", exist_ok=True)
    os.makedirs("data/preprocess/maestro/2017", exist_ok=True)
    os.makedirs("data/preprocess/maestro/2018", exist_ok=True)

    args = [(midi_path, "data/preprocess/maestro") for midi_path in glob("data/input/maestro/maestro-v3.0.0/*/*.midi")]
    with Pool(processes=None) as pool:
        pool.starmap(preprocess_fn, args)


def lakh(preprocess_fn: Callable) -> None:
    for i in range(16):
        parentname = str(hex(i))[2]
        os.makedirs(f"data/preprocess/lakh/{parentname}", exist_ok=True)

    args = [(midi_path, "data/preprocess/lakh") for midi_path in glob("data/input/lakh/lmd_full/*/*.mid")]
    with Pool(processes=None) as pool:
        pool.starmap(preprocess_fn, args)


@hydra.main(version_base=None, config_path="../../conf/musicvae", config_name="config")
def main(cfg: DictConfig) -> None:
    note_seq.melodies_lib.STANDARD_PPQ = cfg.data.ticks_per_beat
    preprocess_fn = functools.partial(
        preprocess,
        target_n_measures=cfg.data.n_measures,
        target_steps_per_bar=cfg.data.n_beats * cfg.data.n_parts_of_beat,
        target_steps_per_quarter=cfg.data.n_parts_of_beat,
    )
    maestro(preprocess_fn)
    # lakh(preprocess_fn)


if __name__ == "__main__":
    main()