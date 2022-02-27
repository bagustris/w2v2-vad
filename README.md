# w2v2-vad
A wrapper for Audeering's wav2vector-based dimensional speech emotion recognition.

## Installation
    pip3 install -r requirements.txt
    
## Usage
    python3 predict_vad_w2v2.py input.wav
    
## Example

```
bagus@L140MU:w2v2-vad$ python3 predict_vad_w2v2.py -i bagus-test_16000.wav 
Valence, arousal, and dominance #0: [[0.32293236 0.41639617 0.5942142 ]]
bagus@L140MU:w2v2-vad$ python3 predict_vad_w2v2.py -i bagus-test_16000.wav -d 2
Valence, arousal, and dominance #0: [[0.3404813  0.42247295 0.35256445]]
Valence, arousal, and dominance #1: [[0.22009875 0.322832   0.51018834]]
Valence, arousal, and dominance #2: [[0.3478799  0.4332775  0.45645887]]
Valence, arousal, and dominance #3: [[0.29967275 0.4038131  0.4949872 ]]
Valence, arousal, and dominance #4: [[0.24804251 0.33543587 0.50990975]]
Valence, arousal, and dominance #5: [[0.38564402 0.43214017 0.37035757]]
```

All credit goes to Audeering.
