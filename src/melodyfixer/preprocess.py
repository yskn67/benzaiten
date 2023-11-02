"""
These codes are licensed under CC0.
"""

import functools
import gc
import json
import os
import sys
from copy import deepcopy
from glob import glob
from multiprocessing import Pool
from typing import Callable

import hydra
import note_seq
from loguru import logger
from omegaconf import DictConfig


PREPROCESS_NAME = "preprocess_fixer_notranspose"


def preprocess(
    midi_path: str,
    output_dir: str,
    target_n_measures: int = 4,
    target_steps_per_bar: int = 16,
    target_steps_per_quarter: int = 4,
) -> None:
    _, midi_basename = os.path.split(midi_path)
    midi_basename_root, _ = os.path.splitext(midi_basename)
    try:
        melody = note_seq.midi_file_to_melody(midi_path)
        # stepsが想定外の場合は処理しない
        if melody.steps_per_bar != target_steps_per_bar or melody.steps_per_quarter != target_steps_per_quarter:
            logger.debug(f"Unexpected steps: {midi_path}")
            return

        # NOTE(2023-10-23): pretrain modelが生成するメロディに多様性がないので移調しないパターンも試す
        # melody.squash(0, 128, transpose_to_key=0)
        quantized_seq = note_seq.quantize_note_sequence(melody.to_sequence(), target_steps_per_quarter)
        # 4/4拍子以外は処理しない
        ts = quantized_seq.time_signatures[0]
        if ts.numerator != 4 or ts.numerator != 4:
            logger.debug(f"Not 4/4: {midi_path}")
            return

        n_steps = quantized_seq.total_quantized_steps
        n_measures = n_steps // target_steps_per_bar
        pitchs_per_measure = [[-1] * target_steps_per_bar for _ in range(n_measures)]  # 休符,note_offは-1とする
        for note in quantized_seq.notes:
            for i in range(note.quantized_start_step, note.quantized_end_step):
                if i >= n_measures * target_steps_per_bar:
                    continue
                measure_idx = i // target_steps_per_bar
                step_idx = i % target_steps_per_bar
                pitchs_per_measure[measure_idx][step_idx] = note.pitch
        for i in range(0, len(pitchs_per_measure), target_n_measures):
            # 規定の小節数に満たない場合保存しない
            if i + target_n_measures + 1 > len(pitchs_per_measure):
                logger.debug(f"skip because of few length {len(pitchs_per_measure)} {i}: {midi_path}")
                continue

            # +1はアウトロ用
            pitchs_per_n_measure = deepcopy(pitchs_per_measure[i:i + target_n_measures + 1])

            # 最初の小節が休符のみの場合は保存しない
            if all([p == -1 for p in pitchs_per_n_measure[0]]):
                logger.debug(f"skip because of all rest: {pitchs_per_n_measure[0]} {midi_path}")
                continue

            # イントロの追加
            if i == 0:
                pitchs_per_n_measure = [[-1] * target_steps_per_bar] + pitchs_per_n_measure
            else:
                pitchs_per_n_measure = [deepcopy(pitchs_per_measure[i - 1])] + pitchs_per_n_measure

            # 後処理
            # イントロは最後の四分音符分のみ使用
            pitchs_per_n_measure[0][:target_steps_per_quarter * 3] = [-1] * (target_steps_per_quarter * 3)
            # アウトロは最初の四分音符を伸ばす
            pitchs_per_n_measure[-1][target_steps_per_quarter * 1:target_steps_per_quarter * 3] = [pitchs_per_n_measure[-1][target_steps_per_quarter * 1 - 1]] * (target_steps_per_quarter * 2)
            pitchs_per_n_measure[-1][target_steps_per_quarter * 3:] = [-1] * target_steps_per_quarter

            # 規定の小節数ごとに保存
            output_path = os.path.join(output_dir, midi_basename_root + f"_{i}" + ".json")
            with open(output_path, "wt") as f:
                json.dump(pitchs_per_n_measure, f)
    except Exception as e:
        logger.debug(f"{e} : {midi_path}")

    gc.collect()


def maestro(preprocess_fn: Callable) -> None:
    os.makedirs(f"data/{PREPROCESS_NAME}/maestro/2004", exist_ok=True)
    os.makedirs(f"data/{PREPROCESS_NAME}/maestro/2006", exist_ok=True)
    os.makedirs(f"data/{PREPROCESS_NAME}/maestro/2008", exist_ok=True)
    os.makedirs(f"data/{PREPROCESS_NAME}/maestro/2009", exist_ok=True)
    os.makedirs(f"data/{PREPROCESS_NAME}/maestro/2011", exist_ok=True)
    os.makedirs(f"data/{PREPROCESS_NAME}/maestro/2013", exist_ok=True)
    os.makedirs(f"data/{PREPROCESS_NAME}/maestro/2014", exist_ok=True)
    os.makedirs(f"data/{PREPROCESS_NAME}/maestro/2015", exist_ok=True)
    os.makedirs(f"data/{PREPROCESS_NAME}/maestro/2017", exist_ok=True)
    os.makedirs(f"data/{PREPROCESS_NAME}/maestro/2018", exist_ok=True)

    args = []
    for midi_path in glob("data/input/maestro/maestro-v3.0.0/*/*.midi"):
        args.append((midi_path, os.path.join(f"data/{PREPROCESS_NAME}/maestro", midi_path.split("/")[-2])))
    with Pool(processes=max(os.cpu_count() - 4, 1)) as pool:
        pool.starmap(preprocess_fn, args)


def lakh(preprocess_fn: Callable) -> None:
    for i in range(16):
        parentname = str(hex(i))[2]
        os.makedirs(f"data/{PREPROCESS_NAME}/lakh/{parentname}", exist_ok=True)

    # 既に処理済みのものはskip
    already_processed = {os.path.split(os.path.splitext(fpath)[0])[-1].split("_")[0] for fpath in glob(f"data/{PREPROCESS_NAME}/lakh/*/*.json")}

    args = []
    for midi_path in glob("data/input/lakh/lmd_full/*/*.mid"):
        basename = os.path.split(os.path.splitext(midi_path)[0])[-1]
        if basename in already_processed:
            continue

        args.append((midi_path, os.path.join(f"data/{PREPROCESS_NAME}/lakh", midi_path.split("/")[-2])))

    logger.info(f"midi file count: {len(args)}")
    with Pool(processes=max(os.cpu_count() - 4, 1)) as pool:
        pool.starmap(preprocess_fn, args)


def lakh_matched(preprocess_fn: Callable) -> None:
    for i in range(16):
        parentname = str(hex(i))[2]
        os.makedirs(f"data/{PREPROCESS_NAME}/lakh_matched/{parentname}", exist_ok=True)

    # 既に処理済みのものはskip
    already_processed = {os.path.split(os.path.splitext(fpath)[0])[-1].split("_")[0] for fpath in glob(f"data/{PREPROCESS_NAME}/lakh_matched/*/*.json")}

    # lakh_matchedは重複するファイルが多く存在するので除外する
    args = {}
    for midi_path in glob("data/input/lakh_matched/lmd_matched/**/*.mid", recursive=True):
        basename = os.path.split(os.path.splitext(midi_path)[0])[-1]
        if basename in already_processed:
            continue

        parentname = basename[0]
        args[basename] = (midi_path, os.path.join(f"data/{PREPROCESS_NAME}/lakh_matched", parentname))
    args = list(args.values())

    logger.info(f"midi file count: {len(args)}")
    with Pool(processes=max(os.cpu_count() - 4, 1)) as pool:
        pool.starmap(preprocess_fn, args)


def infinite_bach(preprocess_fn: Callable) -> None:
    os.makedirs(f"data/{PREPROCESS_NAME}/infinite_bach", exist_ok=True)
    args = []
    for midi_path in glob("data/input/infinite_bach/infinite-bach/data/chorales/midi/*.mid"):
        args.append((midi_path, f"data/{PREPROCESS_NAME}/infinite_bach"))

    logger.info(f"midi file count: {len(args)}")
    with Pool(processes=max(os.cpu_count() - 4, 1)) as pool:
        pool.starmap(preprocess_fn, args)


def weimar_midi(preprocess_fn: Callable) -> None:
    os.makedirs(f"data/{PREPROCESS_NAME}/weimar_midi", exist_ok=True)
    args = []
    for midi_path in glob("data/input/weimar_midi/*.mid"):
        args.append((midi_path, f"data/{PREPROCESS_NAME}/weimar_midi"))

    logger.info(f"midi file count: {len(args)}")
    with Pool(processes=max(os.cpu_count() - 4, 1)) as pool:
        pool.starmap(preprocess_fn, args)


@hydra.main(version_base=None, config_path="../../conf/melodyfixer", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.remove()
    logger.add(sys.stderr, level=cfg.logger.level)
    note_seq.melodies_lib.STANDARD_PPQ = cfg.data.ticks_per_beat
    preprocess_fn = functools.partial(
        preprocess,
        target_n_measures=cfg.data.n_measures,
        target_steps_per_bar=cfg.data.n_beats * cfg.data.n_parts_of_beat,
        target_steps_per_quarter=cfg.data.n_parts_of_beat,
    )
    # 処理時間短い順
    weimar_midi(preprocess_fn)
    infinite_bach(preprocess_fn)
    maestro(preprocess_fn)
    lakh_matched(preprocess_fn)
    # NOTE: ファイル数が多い & メモリリークが発生するのでskip
    # lakh(preprocess_fn)
    logger.info("Finished!")


if __name__ == "__main__":
    main()