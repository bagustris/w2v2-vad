#!/usr/bin/env python3
# usage: ./predict_vad_w2v2.py -i input.wav

import os
import audeer
import audonnx
import audiofile
import argparse

parser = argparse.ArgumentParser(
	description='Predict Valence, arousal, and dominance of  audio file in the range [0, 1].')
parser.add_argument('-i', '--input', type=str)
parser.add_argument('-s', '--split', type=str, default='chunks', 
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
    
wav, fs = audiofile.read(args.input)
model = audonnx.load(model_root)

if args.split == 'chunks':
    for i in range(len(wav) // (fs * args.duration)):
        pred = model(wav[0 + i * fs * args.duration :  (i+1) * fs * args.duration], 
                    fs)
        print(f"Valence, arousal, and dominance #{i}: {pred['logits']}")
else:
    pred = model(wav, fs)
    print(f"Valence, arousal, and dominance: {pred['logits']}")