"""
These codes are licensed under MIT License.
Copyright (C) 2023 yskn67
https://github.com/yskn67/benzaiten/blob/2nd/LICENSE
"""

import os
import sys
import shutil

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.remove()
    logger.add(sys.stderr, level=cfg.logger.level)

    os.makedirs(cfg.data.output_dir, exist_ok=True)
    submission_midi_file_path = os.path.join(cfg.data.output_dir, cfg.data.submission_midi_file)
    logger.info(f"submit {submission_midi_file_path}")
    shutil.copy(submission_midi_file_path, os.path.join(cfg.data.output_dir, "yskn67.mid"))


if __name__ == "__main__":
    main()
