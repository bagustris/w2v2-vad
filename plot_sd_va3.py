#!/usr/bin/env python3
"""Plot the live microphone signal(s) with matplotlib.
Matplotlib and NumPy have to be installed.
"""
import argparse
import queue
import sys
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import audonnx
import os
import audeer

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)

    # Fancy indexing with mapping creates a (necessary!) copy:
    q.put(indata[::args.downsample, mapping])

def update_plot(frame):
    """This is called by matplotlib for each plot update."""
    global plotdata
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break

        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data

    # Compute valence and arousal predictions for audiodata
    audio_data = plotdata[:, 0].astype(np.float32)
    v, a = model(audio_data, args.samplerate)['logits'][-1, 0], model(audio_data, args.samplerate)['logits'][0, 0]
    v, a = 2 * v - 1, 2 * a - 1

    sc.set_offsets(np.column_stack((v, a)))
    return (sc,)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '-l', '--list-devices', action='store_true',
        help='show list of audio devices and exit')
    args, remaining = parser.parse_known_args()
    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parser])
    parser.add_argument(
        'channels', type=int, default=[1], nargs='*', metavar='CHANNEL',
        help='input channels to plot (default: the first)')
    parser.add_argument(
        '-d', '--device', type=int_or_str,
        help='input device (numeric ID or substring)')
    parser.add_argument(
        '-w', '--window', type=float, default=200, metavar='DURATION',
        help='visible time slot (default: %(default)s ms)')
    parser.add_argument(
        '-i', '--interval', type=float, default=30,
        help='minimum time between plot updates (default: %(default)s ms)')
    parser.add_argument(
        '-b', '--blocksize', type=int, help='block size (in samples)')
    parser.add_argument(
        '-sr', '--samplerate', type=float, default=16000, help='sampling rate of audio device')
    parser.add_argument(
        '-n', '--downsample', type=int, default=1, metavar='N',
        help='No downsample (default: %(default)s)')
    args = parser.parse_args(remaining)
    if any(c < 1 for c in args.channels):
        parser.error('argument CHANNEL: must be >= 1')
    mapping = [c - 1 for c in args.channels]  # Channel numbers start with 1
    q = queue.Queue()

    # load model, download if necessary
    model_root = '~/models/w2v2-vad/model'
    cache_root = '~/models/w2v2-vad/cache'

    # create cache folder if it doesn't exist
    audeer.mkdir(cache_root)
    
    
    url = 'https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip'
    mdl_path = os.path.join(os.path.expanduser(model_root), 'model.onnx')
    dst_path = os.path.join(os.path.expanduser(cache_root), 'model.zip')
        
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
    model = audonnx.load(model_root)

    try:
        if args.samplerate is None:
            device_info = sd.query_devices(args.device, 'input')
            args.samplerate = device_info['default_samplerate']

        length = int(args.window * args.samplerate / (1000 * args.downsample))
        plotdata = np.zeros((length, len(args.channels))) * 5  # 5 seconds

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
        plt.text(0.01, 0.99, 
                 f'Sample rate: {int(args.samplerate/args.downsample)} Hz',
                 transform=plt.gca().transAxes, va='top', ha='left')
        plt.tight_layout()

        ani = FuncAnimation(plt.gcf(), update_plot, interval=args.interval)
        with sd.InputStream(device=args.device, channels=max(args.channels),
                            samplerate=args.samplerate, callback=audio_callback):
            plt.show()

    except Exception as e:
        parser.exit(type(e).__name__ + ': ' + str(e))
