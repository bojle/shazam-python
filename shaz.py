# based on https://github.com/MichaelCStrauss/shazam-python/

import numpy as np, matplotlib.pyplot as plt, os, glob, pickle
from typing import List, Dict, Tuple
from tqdm import tqdm
from scipy import signal
from scipy.io.wavfile import read
from collections import defaultdict
import heapq, sys

upper_frequency, frequency_bits = 22050, 10  # global config

def create_hashes(const_map, song_id):
  hashes = {}
  for i, (t, f) in enumerate(const_map):
    for t2, f2 in const_map[i:i+100]:
      dt = t2 - t
      if 1 < dt <= 10:
        fb, f2b = f / upper_frequency * (1<<frequency_bits), f2 / upper_frequency * (1<<frequency_bits)
        h = int(fb) | (int(f2b)<<10) | (int(dt)<<20)
        hashes[h] = (t, song_id)
  return hashes

def create_constellation(x, fs):
  if x.ndim == 2: x = x.mean(axis=1)
  win = 512; x = np.pad(x, (0, win - len(x)%win))
  f, t, Zxx = signal.stft(x, fs, nperseg=win, nfft=win)
  cmap, npeak = [], 15
  for i, z in enumerate(Zxx.T):
    spec = abs(z)
    pk, pr = signal.find_peaks(spec, prominence=0, distance=200)
    top = heapq.nlargest(min(npeak, len(pk)), zip(pr["prominences"], pk))
    cmap += [[i, f[p]] for _, p in top]
  return cmap

def create_database(path):
  songs = sorted(glob.glob(path))
  song_index, db = {}, defaultdict(list)
  for i, file in enumerate(tqdm(songs)):
    song_index[i] = file
    fs, x = read(file)
    const = create_constellation(x, fs)
    for h, v in create_hashes(const, i).items(): db[h].append(v)
  with open("database.pickle", "wb") as f: pickle.dump(db, f)
  with open("song_index.pickle", "wb") as f: pickle.dump(song_index, f)

def find_match(path):
  fs, x = read(path)
  db = pickle.load(open("database.pickle", "rb"))
  lookup = pickle.load(open("song_index.pickle", "rb"))
  def score(hashes):
      hits = {}
      for h, (t1, _) in hashes.items():
          for t0, idx in db.get(h, []):
              hits.setdefault(idx, []).append(t0 - t1)
      scored = {
          idx: max((deltas.count(d), d) for d in set(deltas))
          for idx, deltas in hits.items()
      }
      return sorted(scored.items(), key=lambda x: -x[1][0])
  cm = create_constellation(x, fs)
  hashes = create_hashes(cm, None)
  for i, (s, (score, offset)) in enumerate(score(hashes)):
      print(f"{lookup[s]}: Score {score} at offset {offset}")

if sys.argv[1] == "db":
  create_database("data/*.wav")
elif sys.argv[1] == "fm":
  find_match(sys.argv[2])
