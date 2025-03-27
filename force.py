import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.signal import filtfilt, firwin

import globals as gl


def lowpass_fir(data, n_ord=None, cutoff=None, fsample=None, padlen=None, axis=-1):
    """
    Low-pass filter to remove high-frequency noise from the EMG signal.

    :param data: Input signal to be filtered.
    :param n_ord: Filter order.
    :param cutoff: Cutoff frequency of the low-pass filter.
    :param fsample: Sampling frequency of the input signal.
    :return: Filtered signal.
    """
    numtaps = int(n_ord * fsample / cutoff)
    b = firwin(numtaps + 1, cutoff, fs=fsample, pass_zero='lowpass')

    filtered_data = filtfilt(b, 1, data, axis=axis, padlen=padlen)

    return filtered_data


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


def calc_rt(X, threshold=gl.fthresh):
    """
    Finds the first sample (row index) where any column in the matrix exceeds the given threshold.

    Parameters:
    matrix (np.ndarray): A 2D numpy array where columns are vertical vectors.
    threshold (float): The threshold value to check against.

    Returns:
    int: The first row index where any column exceeds the threshold, or -1 if no value exceeds.
    """

    exceed_mask = np.any(X > threshold, axis=1)
    exceeding_indices = np.where(exceed_mask)[0]

    return int(exceeding_indices[0]) if exceeding_indices.size > 0 else -1


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


def calc_single_trial_metrics(experiment=None, sn=None, day=None, blocks=None):
    ch_idx = np.array(gl.diffCols)

    dat = pd.read_csv(os.path.join(gl.baseDir, f'day{day}', f'{experiment}_{sn}.dat'),
                      sep='\t')

    pinfo = pd.read_csv(os.path.join(gl.baseDir, 'participants.tsv'), sep='\t')
    trained = pinfo[pinfo['subNum'] == sn].trained_chords.reset_index(drop=True)[0].split('.')
    group = pinfo[pinfo['subNum'] == sn].group.reset_index(drop=True)[0]

    single_trial_metrics = {
        'subNum': [],
        'BN': [],
        'TN': [],
        'day': [],
        'hand': [],
        'thumb': [],
        'index': [],
        'middle': [],
        'ring': [],
        'pinkie': [],
        'thumb_der': [],
        'index_der': [],
        'middle_der': [],
        'ring_der': [],
        'pinkie_der': [],
        'trialPoint': [],
        'RT': [],
        'ET': [],
        'MD': [],
        'trained_hand': [],
        'chordID': [],
        'chord': []

    }
    for bl in blocks:

        dat_tmp = dat[dat['BN'] == int(bl)]

        filename = os.path.join(gl.baseDir, f'day{day}', f'{experiment}_{sn}_{int(bl):02d}.mov')

        mov = load_mov(filename)

        for ntrial, mov_tmp in enumerate(mov):

            print(
                f'Processing... subj{sn},  day {day}, block {bl}, trial {ntrial + 1}, hand {dat_tmp.iloc[ntrial].hand}, {len(mov)} trials found...')

            force_tmp = mov_tmp[mov_tmp[:, 0] == gl.wait_exec][:, ch_idx]

            # if len(force_tmp) > 0:
            # flip if left hand
            if dat_tmp.iloc[ntrial].hand == '1':
                force_tmp = np.flip(force_tmp, axis=1)

            # add gain
            force_tmp = force_tmp * gl.fGain
            force_filt = lowpass_fir(force_tmp, n_ord=4, cutoff=10, fsample=gl.fsample, padlen=len(force_tmp) - 1,
                                     axis=0)

            # else:
                # force_tmp = np.zeros((100, 5)) * np.nan
                # force_filt = np.zeros((100, 5)) * np.nan

            force_der1 = np.gradient(force_filt, 1 / gl.fsample, axis=0)

            force_der1_avg = np.abs(force_der1.mean(axis=0))

            if dat_tmp.iloc[ntrial].trialPoint == 1:
                rt_samples = calc_rt(np.abs(force_tmp))
                et_samples = int(force_tmp.shape[0] - gl.hold_time * gl.fsample)

                assert et_samples >= rt_samples

                MD, _ = calc_md(force_tmp[rt_samples:et_samples])
                force_avg = force_tmp[-et_samples:].mean(axis=0)
            else:
                rt_samples = -1
                et_samples = -1
                MD = -1
                force_avg = force_tmp[-int(gl.hold_time * gl.fsample):].mean(axis=0)

            if dat_tmp.iloc[ntrial]['chordID'].astype(int).astype(str) in trained:
                chord = 'trained'
            else:
                chord = 'untrained'

            single_trial_metrics['subNum'].append(dat_tmp.iloc[ntrial]['subNum'])
            single_trial_metrics['chordID'].append(dat_tmp.iloc[ntrial]['chordID'])
            single_trial_metrics['chord'].append(chord)
            single_trial_metrics['day'].append(dat_tmp.iloc[ntrial]['day'])
            single_trial_metrics['hand'].append(dat_tmp.iloc[ntrial]['hand'])
            single_trial_metrics['thumb'].append(force_avg[0])
            single_trial_metrics['index'].append(force_avg[1])
            single_trial_metrics['middle'].append(force_avg[2])
            single_trial_metrics['ring'].append(force_avg[3])
            single_trial_metrics['pinkie'].append(force_avg[4])
            single_trial_metrics['thumb_der'].append(force_der1_avg[0])
            single_trial_metrics['index_der'].append(force_der1_avg[1])
            single_trial_metrics['middle_der'].append(force_der1_avg[2])
            single_trial_metrics['ring_der'].append(force_der1_avg[3])
            single_trial_metrics['pinkie_der'].append(force_der1_avg[4])
            single_trial_metrics['RT'].append(rt_samples / gl.fsample)
            single_trial_metrics['ET'].append((et_samples - rt_samples) / gl.fsample)
            single_trial_metrics['MD'].append(MD)
            # single_trial_metrics['MD_c++'].append(dat_tmp.iloc[ntrial]['MD'])
            single_trial_metrics['BN'].append(dat_tmp.iloc[ntrial]['BN'])
            single_trial_metrics['TN'].append(dat_tmp.iloc[ntrial]['TN'])
            single_trial_metrics['trained_hand'].append(int(group) == int(dat_tmp.iloc[ntrial]['hand']))
            single_trial_metrics['trialPoint'].append(dat_tmp.iloc[ntrial]['trialPoint'])

    single_trial_metrics = pd.DataFrame(single_trial_metrics)

    return single_trial_metrics


def main(args):
    if args.what == 'single_trial':
        for day in args.days:
            if day == 1:
                single_trial_metrics = calc_single_trial_metrics(args.experiment, args.sn, day, args.blocks)
            else:
                single_trial_metrics = pd.concat([
                    single_trial_metrics,
                    calc_single_trial_metrics(args.experiment, args.sn, day, args.blocks)
                ])
        single_trial_metrics.to_csv(
            os.path.join(gl.baseDir, f'subj{args.sn}_single_trial.tsv'), sep='\t', index=False)

    if args.what == 'single_trial_all':
        for sn in args.snS:
            args = argparse.Namespace(
                what='single_trial',
                experiment=args.experiment,
                sn=sn,
                blocks=args.blocks,
                days=args.days,
            )
            main(args)

    if args.what == 'merge_participants':
        df = pd.DataFrame()
        for sn in args.snS:
            df_tmp = pd.read_csv(os.path.join(gl.baseDir, f'subj{sn}_single_trial.tsv'), sep='\t')
            df = pd.concat([df, df_tmp])
        df.to_csv(os.path.join(gl.baseDir, 'merged_single_trial.tsv'), sep='\t', index=False)


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument('what', nargs='?', default=None)
    parser.add_argument('--experiment', type=str, default='efc_2hands')
    # parser.add_argument('--session', type=str, default=None)
    parser.add_argument('--sn', type=int, default=None)
    parser.add_argument('--snS', nargs='+', type=int, default=[100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    parser.add_argument('--day', type=int, default=None)
    parser.add_argument('--days', nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--blocks', type=int, default=[1, 2, 3, 4, 5, 6, 7, 8])

    args = parser.parse_args()
    main(args)
    end = time.time()

    print(f'Finished in {end - start:.2f} seconds')
