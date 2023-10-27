#! /bin/bash

set -euxo pipefail

python3 src/generate.py data.submission_midi_file=output_a.mid data.full_midi_file=output_a_full.mid data.wav_file=output_a.wav generate.use_melodyfixer=false
python3 src/generate.py data.submission_midi_file=output_b.mid data.full_midi_file=output_b_full.mid data.wav_file=output_b.wav generate.n_mask=2 generate.mask_measures=[1,2]
python3 src/generate.py data.submission_midi_file=output_c.mid data.full_midi_file=output_c_full.mid data.wav_file=output_c.wav generate.n_mask=2 generate.mask_measures=[7,8]
python3 src/generate.py data.submission_midi_file=output_d.mid data.full_midi_file=output_d_full.mid data.wav_file=output_d.wav generate.n_mask=2 generate.mask_measures=[1,8]