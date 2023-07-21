#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import audeer
import audonnx
import torchaudio
import torchaudio.transforms as T
import argparse
from matplotlib.animation import FuncAnimation

# sc = None

def emotion_plot(fs, duration):
    # global sc
    plt.figure(figsize=(7, 7))
    sc = plt.scatter([], [], s=150)
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.vlines(0, -1.1, 1.1, linestyle='dashed', lw=1)
    plt.hlines(0, -1.1, 1.1, linestyle='dashed', lw=1)
    ticks = range(-1, 2)
    plt.xticks(ticks, size=12)
    plt.yticks(ticks, size=12)
    plt.grid(alpha=0.9, linestyle='--')
    plt.xlabel('Valence', size=14)
    plt.ylabel('Arousal', size=14)
    plt.title('Emotion Plot in Valence-Arousal Space', size=16)
    
    chunk_size = fs * duration
    num_chunks = wav[0].shape[0] // chunk_size + 1

    def update_plot(i):
        '''Update the scatter plot.'''
        stard_idx = i * fs * duration
        # for the last chunk
        if i == num_chunks - 1:
            end_idx = wav[0].shape[0]
        else:
            end_idx = (i + 1) * fs * duration
        pred = model(
            wav[0][stard_idx: end_idx], fs)
        v, a = pred['logits'][:, -1], pred['logits'][:, 0]
        v, a = 2 * v - 1, 2 * a - 1
        print(f"valence, arousal #{i}: {v}, {a}")
        sc.set_offsets(np.column_stack((v, a)))

    ani = FuncAnimation(plt.gcf(), update_plot, frames=num_chunks, repeat=False)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Predict Arousal, Dominance, and Valence of audio file and plot it in the Valence-Arousal space.')
    parser.add_argument('input', type=str, help='Path to the input audio file')
    parser.add_argument('-s', '--split', type=str, default='full',
                        help='chunks or full')
    parser.add_argument('-d', '--duration', type=int, default=10,
                        help='Duration of each chunk in seconds if `split` is `chunks`')
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

    # read audiofile
    wav, fs = torchaudio.load(args.input, normalize=True)
    if fs != 16000:
        sampler = T.Resample(fs, 16000)
        wav = sampler(wav)
    # convert tensor to array
    wav = wav.numpy()
    model = audonnx.load(model_root)

   
    emotion_plot(fs, args.duration)
