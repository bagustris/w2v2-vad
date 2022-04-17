#!/usr/bin/env python3
# usage: ./predict_vad_w2v2.py -i input.wav

import os
import audeer
import audonnx
import librosa
# import audiofile
import argparse

parser = argparse.ArgumentParser(
	description='Predict Arousal, dominance, and Valence of audio file in the range [0, 1].')
parser.add_argument('input', type=str)
parser.add_argument('-s', '--split', type=str, default='full', 
                    help='chunks or full')
parser.add_argument('-d', '--duration', type=int, default=10, 
                    help='duration of each chunk in seconds if `split` is `chunks`')
args = parser.parse_args()

model_root = 'model'
cache_root = 'cache'

# create cache folder if it doesn't exist
audeer.mkdir(cache_root)


def cache_path(file):
    return os.path.join(cache_root, file)


url = 'https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip'
dst_path = cache_path('model.zip')

if not os.path.exists(dst_path):
    audeer.download_url(
        url, 
        dst_path, 
        verbose=True,
    )
    audeer.extract_archive(
        dst_path, 
        model_root, 
        verbose=True,
    )
    
wav, fs = librosa.load(args.input, sr=16000)
# wav, fs = audiofile.read(args.input)
model = audonnx.load(model_root)

if args.split == 'chunks':
    for i in range(wav.shape[0] // (fs * args.duration)):
        pred = model(
            wav[0 + i * fs * args.duration :  (i+1) * fs * args.duration], fs)
        print(f"Arousal, dominance, valence #{i}: {pred['logits']}")
    if wav.shape[0] % fs != 0:
        pred = model(wav[-(wav.shape[0] % fs) : ], fs)
        print(f"Arousal, dominance, valence #{i+1}: {pred['logits']}")    
elif args.split == 'full':
    pred = model(wav, fs)
    print(f"Arousal, dominance, valence: {pred['logits']}")
else:
    raise ValueError(f"Invalid value for `split`: {args.split}")