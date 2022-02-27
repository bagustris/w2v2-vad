#!/usr/bin/env python3
# usage: predict_vadn.py -i input.wav -m model.joblib

from joblib import load
import opensmile
import argparse
from sklearn.preprocessing import StandardScaler
import audiofile

parser = argparse.ArgumentParser(
	description='Predict Valence, arousal, dominance, and naturalness from audio file. Usage: ./predict_vadn.py -i input.wav -m model.joblib')
parser.add_argument('-i', '--input', type=str)
parser.add_argument('-m', '--model', type=str,
                    default='model/nn_sner_v1.0.joblib')
parser.add_argument('-s', '--split', type=str,
                    default='chunks', help='chunks (10s) or full (no split')
parser.add_argument('-d', '--duration', type=int, default=10)
args = parser.parse_args()

nn = load(args.model)

# confifugation for acoustic feature extraction
smile_compare = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# Get scaler
scaler = load('model/scaler_feat_sner_v1.0.joblib')

# Choose mode
if args.split == 'chunks':
    wav, fs = audiofile.read(args.input)
    for i in range(len(wav) // (fs * args.duration)):
        feat = smile_compare.process_signal(
            wav[0 + i * fs * args.duration :  (i+1) * fs * args.duration], fs
        )
        scaled_feat = scaler.transform(feat.to_numpy())
        pred = nn.predict(scaled_feat)
        print(f"Valence, arousal, dominance, naturalness #{i}: {pred}")
else:
    feat = smile_compare.process_file(args.input)
    scaled_feat = scaler.transform(feat.to_numpy())
    pred = nn.predict(scaled_feat)
    print(f"Valence, arousal, dominance, naturalness: {pred}")