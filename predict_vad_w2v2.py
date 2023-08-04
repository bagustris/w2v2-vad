#!/usr/bin/env python3
# usage: ./predict_vad_w2v2.py -i input.wav

import os
import audeer
import audonnx
import audresample
import audiofile
import argparse

parser = argparse.ArgumentParser(
	description='Predict Arousal, dominance, and Valence of audio file in the range [0, 1].')
parser.add_argument('-i', '--input', type=str, default='bagus-test_16000.wav')
parser.add_argument('-s', '--split', type=str, default='full',
                    help='chunks or full')
parser.add_argument('-d', '--duration', type=int, default=10,
                    help='duration of each chunk in seconds if `split` is `chunks`')
args = parser.parse_args()

# make ~/models as root directory
root_dir = os.path.expanduser('~/models/w2v2-vad')

# create root directory if it does not exist
if not os.path.exists(root_dir):
    os.makedirs(root_dir, exist_ok=True)

cache_root = audeer.mkdir(os.path.join(root_dir, 'cache'))
model_root = audeer.mkdir(os.path.join(root_dir, 'model'))


def cache_path(file):
    return os.path.join(cache_root, file)


url = 'https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip'
mdl_path = os.path.join(model_root, 'model.onnx')
dst_path = cache_path('model.zip')

if not os.path.exists(mdl_path):
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

# read audiofile
wav, fs = audiofile.read(args.input)
if fs != 16000:
    wav = audresample.resample(wav, fs, 16000)

# load model
model = audonnx.load(model_root)

if args.split == 'chunks':
    for i in range(wav.shape[0] // (fs * args.duration)):
        pred = model(
            wav[(0 + i) * fs * args.duration:  (i+1) * fs * args.duration], fs)
        print(f"Arousal, dominance, valence #{i}: {pred['logits']}")
    # for the last chunk
    if wav.shape[0] % (fs * args.duration) != 0:
        pred = model(wav[-(wav.shape[0] % (fs*args.duration)):], fs)
        print(f"Arousal, dominance, valence #{i+1}: {pred['logits']}")
elif args.split == 'full':
    pred = model(wav, fs)
    print(f"Arousal, dominance, valence: {pred['logits']}")
else:
    raise ValueError(f"Invalid value for `split`: {args.split}")
