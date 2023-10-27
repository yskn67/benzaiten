#! /bin/bash

set -euxo pipefail

python3 src/generate.py data.submission_midi_file=output_a.mid data.full_midi_file=output_a_full.mid data.wav_file=output_a.wav generate.use_melodyfixer=false
python3 src/generate.py data.submission_midi_file=output_b.mid data.full_midi_file=output_b_full.mid data.wav_file=output_b.wav generate.n_mask=0
python3 src/generate.py data.submission_midi_file=output_c.mid data.full_midi_file=output_c_full.mid data.wav_file=output_c.wav generate.n_mask=1 generate.mask_measures=[1]
python3 src/generate.py data.submission_midi_file=output_d.mid data.full_midi_file=output_d_full.mid data.wav_file=output_d.wav generate.n_mask=8 generate.mask_measures=[0,1,2,5,6,7,8,9]
python3 src/generate.py data.submission_midi_file=output_e.mid data.full_midi_file=output_e_full.mid data.wav_file=output_e.wav generate.n_mask=8 generate.mask_measures=[0,1,2,3,4,7,8,9]