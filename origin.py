import music21
import numpy as np
import matplotlib.pyplot as plt
import mido
import csv
import IPython.display as ipd
import midi2audio
import glob
import tensorflow as tf
import tensorflow_probability as tfp


TOTAL_MEASURES = 240                      # 学習用MusicXMLを読み込む際の小節数の上限
UNIT_MEASURES = 4                         # 1回の生成で扱う旋律の長さ
BEAT_RESO = 4                             # 1拍を何個に分割するか（4の場合は16分音符単位）
N_BEATS = 4                               # 1小節の拍数（今回は4/4なので常に4）
NOTENUM_FROM = 36                         # 扱う音域の下限（この値を含む）
NOTENUM_THRU = 84                         # 扱う音域の上限（この値を含まない）
NOTE_RANGE = NOTENUM_THRU - NOTENUM_FROM  # 音域
INTRO_BLANK_MEASURES = 4                  # ブランクおよび伴奏の小節数の合計
MELODY_LENGTH = 8                         # 生成するメロディの長さ（小節数）

###### 2023.08.04 追加
TICKS_PER_BEAT = 480                      # 四分音符を何ticksに分割するか
MELODY_PROG_CHG = 73                      # メロディの音色（プログラムチェンジ）
MELODY_CH = 0                             # メロディのチャンネル

KEY_ROOT = "C"                            # 生成するメロディの調のルート（"C" or "A"）
KEY_MODE = "major"                        # 生成するメロディの調のモード（"major" or "minor"）


# MusicXMLデータからNote列とChordSymbol列を生成
# 時間分解能は BEAT_RESO にて指定
def make_note_and_chord_seq_from_musicxml(score):
  note_seq = [None] * (TOTAL_MEASURES * N_BEATS * BEAT_RESO)
  chord_seq = [None] * (TOTAL_MEASURES * N_BEATS * BEAT_RESO)

  for element in score.parts[0].elements:
    if isinstance(element, music21.stream.Measure):
      measure_offset = element.offset
      for note in element.notes:
        if isinstance(note, music21.note.Note):
          onset = measure_offset + note._activeSiteStoredOffset
          offset = onset + note._duration.quarterLength
          for i in range(int(onset * BEAT_RESO), int(offset * BEAT_RESO + 1)):
            note_seq[i] = note
        if isinstance(note, music21.harmony.ChordSymbol):
          chord_offset = measure_offset + note.offset
          for i in range(int(chord_offset * BEAT_RESO),
                         int((measure_offset + N_BEATS) * BEAT_RESO + 1)):
            chord_seq[i] = note
  return note_seq, chord_seq

# Note列をone-hot vector列（休符はすべて0）に変換
def note_seq_to_onehot(note_seq):
  M = NOTE_RANGE
  N = len(note_seq)
  matrix = np.zeros((N, M))
  for i in range(N):
    if note_seq[i] != None:
      matrix[i, note_seq[i].pitch.midi - NOTENUM_FROM] = 1
  return matrix

# 音符列を表すone-hot vector列に休符要素を追加
def add_rest_nodes(onehot_seq):
  rest = 1 - np.sum(onehot_seq, axis=1)
  rest = np.expand_dims(rest, 1)
  return np.concatenate([onehot_seq, rest], axis=1)

# 指定された仕様のcsvファイルを読み込んで
# ChordSymbol列を返す
def read_chord_file(file):
  chord_seq = [None] * (MELODY_LENGTH * N_BEATS)
  with open(file) as f:
    reader = csv.reader(f)
    for row in reader:
      m = int(row[0]) # 小節番号（0始まり）
      if m < MELODY_LENGTH:
        b = int(row[1]) # 拍番号（0始まり、今回は0または2）
        chord_seq[m*4+b] = music21.harmony.ChordSymbol(root=row[2],
                                                       kind=row[3],
                                                       bass=row[4])
  for i in range(len(chord_seq)):
    if chord_seq[i] != None:
      chord = chord_seq[i]
    else:
      chord_seq[i] = chord
  return chord_seq

# コード進行からChordSymbol列を生成
# divisionは1小節に何個コードを入れるか
def make_chord_seq(chord_prog, division):
  T = int(N_BEATS * BEAT_RESO / division)
  seq = [None] * (T * len(chord_prog))
  for i in range(len(chord_prog)):
    for t in range(T):
      if isinstance(chord_prog[i], music21.harmony.ChordSymbol):
        seq[i * T + t] = chord_prog[i]
      else:
        seq[i * T + t] = music21.harmony.ChordSymbol(chord_prog[i])
  return seq

# ChordSymbol列をmany-hot (chroma) vector列に変換
def chord_seq_to_chroma(chord_seq):
  N = len(chord_seq)
  matrix = np.zeros((N, 12))
  for i in range(N):
    if chord_seq[i] != None:
      for note in chord_seq[i]._notes:
        matrix[i, note.pitch.midi % 12] = 1
  return matrix

# 空（全要素がゼロ）のピアノロールを生成
def make_empty_pianoroll(length):
  return np.zeros((length, NOTE_RANGE + 1))

# ピアノロール（one-hot vector列）をノートナンバー列に変換
def calc_notenums_from_pianoroll(pianoroll):
  notenums = []
  for i in range(pianoroll.shape[0]):
    n = np.argmax(pianoroll[i, :])
    nn = -1 if n == pianoroll.shape[1] - 1 else n + NOTENUM_FROM
    notenums.append(nn)
  return notenums

# 連続するノートナンバーを統合して (notenums, durations) に変換
def calc_durations(notenums):
  N = len(notenums)
  duration = [1] * N
  for i in range(N):
    k = 1
    while i + k < N:
      if notenums[i] > 0 and notenums[i] == notenums[i + k]:
        notenums[i + k] = 0
        duration[i] += 1
      else:
        break
      k += 1
  return notenums, duration

####### 2023.08.04 追加
# MIDIトラックを生成（make_midiから呼び出される）
def make_midi_track(notenums, durations, transpose, ticks_per_beat):
  track = mido.MidiTrack()
  init_tick = INTRO_BLANK_MEASURES * N_BEATS * ticks_per_beat
  prev_tick = 0
  for i in range(len(notenums)):
    if notenums[i] > 0:
      curr_tick = int(i * ticks_per_beat / BEAT_RESO) + init_tick
      track.append(mido.Message('note_on', note=notenums[i]+transpose,
                                velocity=100, time=curr_tick - prev_tick))
      prev_tick = curr_tick
      curr_tick = int((i + durations[i]) * ticks_per_beat / BEAT_RESO) + init_tick
      track.append(mido.Message('note_off', note=notenums[i]+transpose,
                                velocity=100, time=curr_tick - prev_tick))
      prev_tick = curr_tick
  return track

####### 2023.08.04 追加
# プログラムチェンジを指定したものに差し替え
def replace_prog_chg(midi):
  for track in midi.tracks:
    for msg in track:
      if msg.type == 'program_change' and msg.channel == MELODY_CH:
        msg.program = MELODY_PROG_CHG

####### 2023.08.04 追加
# MIDIファイル（提出用、伴奏なし）を生成
def make_midi_for_submission(notenums, durations, transpose, dst_filename):
  midi = mido.MidiFile(type=1)
  midi.ticks_per_beat = TICKS_PER_BEAT
  midi.tracks.append(make_midi_track(notenums, durations, transpose, TICKS_PER_BEAT))
  midi.save(dst_filename)

####### 2023.08.04 修正
# MIDIファイル（チェック用、伴奏あり）を生成
def make_midi_for_check(notenums, durations, transpose, src_filename, dst_filename):
  midi = mido.MidiFile(src_filename)
  replace_prog_chg(midi)
  midi.tracks.append(make_midi_track(notenums, durations, transpose, midi.ticks_per_beat))
  midi.save(dst_filename)

####### 2023.08.04 修正
# ピアノロールを描画し、MIDIファイルを再生
def show_and_play_midi(pianoroll, transpose, src_filename, dst_filename1, dst_filename2):
  plt.matshow(np.transpose(pianoroll))
  plt.show()
  notenums = calc_notenums_from_pianoroll(pianoroll)
  notenums, durations = calc_durations(notenums)
  ###### 2023.08.04 変更
  make_midi_for_submission(notenums, durations, transpose, dst_filename1)
  make_midi_for_check(notenums, durations, transpose, src_filename, dst_filename2)
  fs = midi2audio.FluidSynth(sound_font="/usr/share/sounds/sf2/FluidR3_GM.sf2")
  fs.midi_to_audio(dst_filename2, "output.wav")
  # ipd.display(ipd.Audio("output.wav"))

# メロディを表すone-hotベクトル、コードを表すmany-hotベクトルの系列に対して、
# UNIT_MEASURES小節分だけ切り出したものを返す
def extract_seq(i, onehot_seq, chroma_seq):
  o = onehot_seq[i*N_BEATS*BEAT_RESO : (i+UNIT_MEASURES)*N_BEATS*BEAT_RESO, :]
  c = chroma_seq[i*N_BEATS*BEAT_RESO : (i+UNIT_MEASURES)*N_BEATS*BEAT_RESO, :]
  return o, c

# メロディを表すone-hotベクトル、コードを表すmany-hotベクトルの系列から、
# モデルの入力、出力用のデータに整えて返す
def calc_xy(o, c):
  o_input = o.copy()
  o_input[1:, :] = o_input[:-1, :]
  o_input[0, :] = np.zeros_like(o_input[0, :])
  x = np.concatenate([o_input, c], axis=1)
  y = o
  return x, y

# メロディを表すone-hotベクトル、コードを表すmany-hotベクトルの系列から
# モデルの入力、出力用のデータを作成して、配列に逐次格納する
def divide_seq(onehot_seq, chroma_seq, x_all, y_all):
  for i in range(0, TOTAL_MEASURES, UNIT_MEASURES):
    o, c, = extract_seq(i, onehot_seq, chroma_seq)
    if np.any(o[:, 0:-1] != 0):
      x, y = calc_xy(o, c)
      x_all.append(x)
      y_all.append(y)


basedir = "./data/input/origin/"
dir = basedir + "omnibook/"

x_all = []
y_all = []
for f in glob.glob(dir + "/*.xml"):
  print(f)
  score = music21.converter.parse(f)
  key = score.analyze("key")
  if key.mode == KEY_MODE:
    inter = music21.interval.Interval(key.tonic, music21.pitch.Pitch(KEY_ROOT))
    score = score.transpose(inter)
    note_seq, chord_seq = make_note_and_chord_seq_from_musicxml(score)
    onehot_seq = add_rest_nodes(note_seq_to_onehot(note_seq))
    chroma_seq = chord_seq_to_chroma(chord_seq)
    divide_seq(onehot_seq, chroma_seq, x_all, y_all)

x_all = np.array(x_all)
y_all = np.array(y_all)

encoded_dim = 32                   # 潜在空間の次元数
seq_length = x_all.shape[1]        # 時間軸上の要素数
input_dim = x_all.shape[2]         # 入力データにおける各時刻のベクトルの次元数
output_dim = y_all.shape[2]        # 出力データにおける各時刻のベクトルの次元数
lstm_dim = 1024                    # LSTM層のノード数

# VAEに用いる事前分布を定義
def make_prior():
  tfd = tfp.distributions
  prior = tfd.Independent(
      tfd.Normal(loc=tf.zeros(encoded_dim), scale=1),
      reinterpreted_batch_ndims=1)
  return prior

# エンコーダを構築
def make_encoder(prior):
  encoder = tf.keras.Sequential()
  encoder.add(tf.keras.layers.LSTM(lstm_dim,
                                   input_shape=(seq_length, input_dim),
                                   use_bias=True, activation="tanh",
                                   return_sequences=False))
  encoder.add(tf.keras.layers.Dense(
      tfp.layers.MultivariateNormalTriL.params_size(encoded_dim),
      activation=None))
  encoder.add(tfp.layers.MultivariateNormalTriL(
      encoded_dim,
      activity_regularizer=tfp.layers.KLDivergenceRegularizer(
          prior, weight=0.001)))
  return encoder

# デコーダを構築
def make_decoder():
  decoder = tf.keras.Sequential()
  decoder.add(tf.keras.layers.RepeatVector(seq_length,
                                           input_dim=encoded_dim))
  decoder.add(tf.keras.layers.LSTM(lstm_dim, use_bias=True,
                                   activation="tanh",
                                   return_sequences=True))
  decoder.add(tf.keras.layers.Dense(output_dim, use_bias=True,
                                    activation="softmax"))
  return decoder

# エンコーダとデコーダを構築し、それらを結合したモデルを構築する
# （入力：エンコーダの入力、
# 　出力：エンコーダの出力をデコーダに入力して得られる出力）
def make_model():
  encoder = make_encoder(make_prior())
  decoder = make_decoder()
  vae = tf.keras.Model(encoder.inputs, decoder(encoder.outputs))
  vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss="categorical_crossentropy", metrics="categorical_accuracy")
  return vae

vae = make_model()                        # VAEモデルを構築
vae.fit(x_all, y_all, epochs=500)         # VAEモデルを学習
vae.save(basedir + "/mymodel.h5")         # 学習済みVAEモデルをファイルに保存

backing_file = "sample1_backing.mid"       # 適宜変更すること
chord_file = "sample1_chord.csv"           # 適宜変更すること

# 2023.08.04 変更
output_file1 = "output1.mid"                # 自分のエントリーネームに変更すること
output_file2 = "output2.mid"

vae = make_model()
vae.load_weights(basedir + "/mymodel.h5")

chord_prog = read_chord_file(basedir + chord_file)
chroma_vec = chord_seq_to_chroma(make_chord_seq(chord_prog, N_BEATS))
pianoroll = make_empty_pianoroll(chroma_vec.shape[0])
for i in range(0, MELODY_LENGTH, UNIT_MEASURES):
  o, c = extract_seq(i, pianoroll, chroma_vec)
  x, y = calc_xy(o, c)
  for j in range(x.shape[0]):
    y_new = vae.predict(np.array([x]))
    if j+1 >= x.shape[0]:
      break
    xj_new = np.zeros_like(y_new[0, :j+1, :])
    for k, l in enumerate(y_new[0, :j+1, :].argmax(axis=1)):
      xj_new[k, l] = 1
    x[1:xj_new.shape[0] + 1, :NOTE_RANGE + 1] = xj_new
  index_from = i * (N_BEATS * BEAT_RESO)
  pianoroll[index_from : index_from + y_new[0].shape[0], :] = y_new[0]
# 2023.08.04 変更
show_and_play_midi(pianoroll, 12, basedir + "/" + backing_file,
                   basedir + output_file1, basedir + output_file2)