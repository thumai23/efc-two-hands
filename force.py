import argparse

from sklearn.preprocessing import MinMaxScaler
import pandas as pd

import numpy as np

import warnings

import globals as gl

import os

def load_mov(filename):
    """
    load .mov file of one block

    :return:
    """

    try:
        with open(filename, 'rt') as fid:
            trial = 0
            A = []
            for line in fid:
                if line.startswith('Trial'):
                    trial_number = int(line.split(' ')[1])
                    trial += 1
                    if trial_number != trial:
                        warnings.warn('Trials out of sequence')
                        trial = trial_number
                    A.append([])
                else:
                    # Convert line to a numpy array of floats and append to the last trial's list
                    data = np.fromstring(line, sep=' ')
                    if A:
                        A[-1].append(data)
                    else:
                        # This handles the case where a data line appears before any 'Trial' line
                        warnings.warn('Data without trial heading detected')
                        A.append([data])

            # Convert all sublists to numpy arrays
            mov = [np.array(trial_data) for trial_data in A]
            # # vizForce = [np.array(trial_data)[:, 9:] for trial_data in A]
            # state = [np.array(trial_data) for trial_data in A]

    except IOError as e:
        raise IOError(f"Could not open {filename}") from e

    return mov


def calc_md(X):
    N, m = X.shape
    F1 = X[0]
    FN = X[-1] - F1  # Shift the end point

    shifted_matrix = X - F1  # Shift all points

    d = list()

    for t in range(1, N - 1):
        Ft = shifted_matrix[t]

        # Project Ft onto the ideal straight line
        proj = np.dot(Ft, FN) / np.dot(FN, FN) * FN

        # Calculate the Euclidean distance
        d.append(np.linalg.norm(Ft - proj))

    d = np.array(d)
    MD = d.mean()

    return MD, d


def get_segment(x, hold_time=gl.hold_time):
    c = np.any(np.abs(x) > gl.fthresh, axis=1)

    start_samp_exec = np.argmax(c)

    if hold_time is None:
        starttime = start_samp_exec / gl.fsample
        d = np.all(np.abs(x) > gl.ftarget, axis=1)
        end_samp_exec = np.argmax(d) if np.any(d) else None
        endtime = end_samp_exec / gl.fsample if end_samp_exec is not None else None
        x_s = x[start_samp_exec:end_samp_exec]
    else:
        x_s = x[start_samp_exec:-int(hold_time * gl.fsample)]
        starttime = start_samp_exec / gl.fsample
        execTime = ((len(x) - int(hold_time * gl.fsample)) / gl.fsample) - starttime

    # assert execTime > 0
    # assert starttime > 0

    return x_s, starttime, execTime

def calc_single_trial(day=None, sn=None):

    behavioural_dict = {
        'BN': [],
        'TN': [],
        'participant_id': [],
        'subNum': [],
        'chordID': [],
        'chord': [],
        'hand': [],
        'trialPoint': [],
        'repetition': [],
        'day': [],
        'MD': [],
        'RT': [],
        'ET': [],
        'thumb_force': [],
        'index_force': [],
        'middle_force': [],
        'ring_force': [],
        'pinkie_force': [],
    }

    path = os.path.join(gl.baseDir, gl.dataDir, f"efc_2hands_day0{day}")

    pinfo = pd.read_csv(os.path.join(gl.baseDir, 'participants.tsv'), sep='\t')
    blocks = ['1', '2', '3', '4', '5', '6', '7', '8']

    dat = pd.read_csv(os.path.join(path, f"efc_2hands_{sn}.dat"), sep="\t")

    # nblocks = len(pinfo[pinfo['participant_id'] == p][f'blocks Chords day{day}'][0].split(','))

    for block in blocks:

        block = int(block)

        print(f"participant_id:subj{sn}, "
              f"day:{day}, "
              f"block:{block}")

        filename = os.path.join(path, f'efc_2hands_{sn}_{block:02d}.mov')

        mov = load_mov(filename)

        dat_tmp = dat[dat.BN == block].reset_index()  # .dat file for block

        for tr in range(len(mov)):

            if tr == 0 or dat_tmp.iloc[tr].chordID != dat_tmp.iloc[tr - 1].chordID:
                rep = 1
            else:
                rep += 1

            chordID = dat_tmp.iloc[tr].chordID.astype(int).astype(str)
            chord = 'trained' if chordID in pinfo[pinfo['subNum'] == sn]['trained'][0].split('.') else 'untrained'

            # add trial info to dictionary
            behavioural_dict['BN'].append(dat_tmp.iloc[tr].BN)
            behavioural_dict['TN'].append(dat_tmp.iloc[tr].TN)
            behavioural_dict['subNum'].append(sn)
            behavioural_dict['participant_id'].append(f'subj{sn}')
            behavioural_dict['chordID'].append(chordID)
            behavioural_dict['trialPoint'].append(dat_tmp.iloc[tr].trialPoint)
            behavioural_dict['chord'].append(chord)
            behavioural_dict['day'].append(day)
            behavioural_dict['repetition'].append(rep)

            if dat_tmp.iloc[tr].trialPoint == 1:

                forceRaw = mov[tr][:, gl.diffCols][mov[tr][:, 0] == 3] * gl.fGain  # take only states 3 (i.e., WAIT_EXEC)

                # calc single trial metrics
                force, rt, et = get_segment(forceRaw, hold_time=gl.hold_time)

                # force_avg = force.mean(axis=0)
                md, _ = calc_md(force)

                assert rt > 0, "negative reaction time"
                assert et > 0, "negative execution time"
                assert md > 0, "negative mean deviation"

                # add measures to dictionary
                behavioural_dict['RT'].append(rt)
                behavioural_dict['ET'].append(et)
                behavioural_dict['MD'].append(md)
                # behavioural_dict['thumb_force'].append(force_avg[0])
                # behavioural_dict['index_force'].append(force_avg[1])
                # behavioural_dict['middle_force'].append(force_avg[2])
                # behavioural_dict['ring_force'].append(force_avg[3])
                # behavioural_dict['pinkie_force'].append(force_avg[4])

            else:

                # add to dictionary
                behavioural_dict['RT'].append(None)
                behavioural_dict['ET'].append(None)
                behavioural_dict['MD'].append(None)
                # behavioural_dict['thumb_force'].append(None)
                # behavioural_dict['index_force'].append(None)
                # behavioural_dict['middle_force'].append(None)
                # behavioural_dict['ring_force'].append(None)
                # behavioural_dict['pinkie_force'].append(None)

    behav = pd.DataFrame(behavioural_dict)

    return behav

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    # parser.add_argument('--day', type=int, default=None)
    # parser.add_argument('--sn', type=int, default=None)

    args = parser.parse_args()

    if args.what == 'single_trial':

        pinfo = pd.read_csv(os.path.join(gl.baseDir, 'participants.tsv'), sep='\t')

        for day in np.arange(1, 5, 5):
            for sn in pinfo.subNum.unique():
                behav = calc_single_trial(
                    day=day,
                    sn=sn,
                )

                pass

    else:
        parser.print_help()

if __name__ == '__main__':
    main()

