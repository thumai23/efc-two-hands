from pathlib import Path
import numpy as np

Dirs = ["/Volumes/diedrichsen_data$/data/Chord_exp/EFC_twoHands",
        "/cifs/diedrichsen/data/Chord_exp/EFC_twoHands"]

baseDir = next((Dir for Dir in Dirs if Path(Dir).exists()), None)
dataDir = 'data'

diffCols = [13, 14, 15, 16, 17]
fGain = np.array([1, 1, 1, 1.5, 1.5])

hold_time = .6

fthresh = 1.2  # threshold to exit the baseline area

fsample = 500