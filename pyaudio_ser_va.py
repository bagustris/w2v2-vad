import os
import numpy as np
import matplotlib.pyplot as plt
import audeer
import audonnx
import argparse
import pyaudio

def emotion_plot_live(fs, chunk_size):
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

    model = audonnx.load(model_root)

    def update_plot(i, generator):
        '''Update the scatter plot.'''
        audio_data = next(generator)
        pred = model(audio_data, fs)
        v, a = pred['logits'][:, -1], pred['logits'][:, 0]
        v, a = 2 * v - 1, 2 * a - 1
        print(f"valence, arousal #{i}: {v}, {a}")
        sc.set_offsets(np.column_stack((v, a)))

    generator = audio_generator(chunk_size)

    ani = FuncAnimation(plt.gcf(), update_plot, fargs=(generator,), frames=100, repeat=False)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Predict Arousal, Dominance, and Valence of audio from a live microphone and plot it in the Valence-Arousal space.')
    parser.add_argument('-d', '--duration', type=int, default=10,
                        help='Duration of each chunk in seconds if `split` is `chunks`')
    args = parser.parse_args()

    emotion_plot_live(16000, 16000 * args.duration)