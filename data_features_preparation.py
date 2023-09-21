import zipfile

import scipy.signal as scisig
import cvxEDA
import numpy as np
import pandas as pd
import scipy
import pickle
import os

from utils import read_acc_file, read_wrist_file
from wesad_training import feats

WINDOW_IN_SECONDS = 30
feat_names = None

fs_dict = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4, 'label': 700, 'HR': 1}


def filterSignalFIR(eda, cutoff=0.4, numtaps=64):
    f = cutoff / (fs_dict['ACC'] / 2.0)
    FIR_coeff = scisig.firwin(numtaps, f)

    return scisig.lfilter(FIR_coeff, 1, eda)


def butter_lowpass(cutoff, fs, order=5):
    # Filtering Helper functions
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scisig.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    # Filtering Helper functions outer
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = scisig.lfilter(b, a, data)
    return y


def eda_stats(y):
    Fs = fs_dict['EDA']
    yn = (y - y.mean()) / y.std()
    [r, p, t, l, d, e, obj] = cvxEDA.cvxEDA(yn, 1. / Fs)
    return [r, p, t, l, d, e, obj]


def get_peak_freq(x):
    f, Pxx = scisig.periodogram(x, fs=8)
    psd_dict = {amp: freq for amp, freq in zip(Pxx, f)}
    peak_freq = psd_dict[max(psd_dict.keys())]
    return peak_freq


def get_net_accel(data):
    return (data['ACC_x'] ** 2 + data['ACC_y'] ** 2 + data['ACC_z'] ** 2).apply(lambda x: np.sqrt(x))


def get_window_stats(data, label=-1):
    mean_features = np.mean(data)
    std_features = np.std(data)
    min_features = np.amin(data)
    max_features = np.amax(data)

    features = {'mean': mean_features, 'std': std_features, 'min': min_features, 'max': max_features,
                'label': label}
    return features


def get_slope(series):
    linreg = scipy.stats.linregress(np.arange(len(series)), series)
    slope = linreg[0]
    return slope


def get_samples(data, n_windows, label, predicting):
    global feat_names, y
    global WINDOW_IN_SECONDS

    samples = []
    # Using label freq (700 Hz) as our reference frequency due to it being the largest
    # and thus encompassing the lesser ones in its resolution.
    window_len = fs_dict['label'] * WINDOW_IN_SECONDS

    for i in range(n_windows):
        # Get window of data
        w = data[window_len * i: window_len * (i + 1)]

        # Add/Calc rms acc
        w = pd.concat([w, get_net_accel(w)])

        # Calculate stats for window
        wstats = get_window_stats(data=w, label=label)

        if not predicting:
            # Seperating sample and label
            x = pd.DataFrame(wstats).drop('label', axis=0)
            y = x['label'][0]
            x.drop('label', axis=1, inplace=True)
        else:
            x = pd.DataFrame(wstats).drop('label', axis=1)

        if feat_names is None:
            feat_names = []
            for row in x.index:
                for col in x.columns:
                    if row == 0:
                        row = 'net_acc'
                    feat_names.append('_'.join([row, col]))

        # sample df
        wdf = pd.DataFrame(x.values.flatten()).T
        wdf.columns = feat_names

        if not predicting:
            wdf = pd.concat([wdf, pd.DataFrame({'label': y}, index=[0])], axis=1)

        # More feats
        wdf['BVP_peak_freq'] = get_peak_freq(w['BVP'].dropna())
        wdf['TEMP_slope'] = get_slope(w['TEMP'].dropna())
        samples.append(wdf)

    return pd.concat(samples)


def extract_features(full_path, dirname, train):
    global start_datetime
    label_df = pd.DataFrame()
    baseline_samples = pd.DataFrame()
    stress_samples = pd.DataFrame()
    amusement_samples = pd.DataFrame()

    if train:
        with open(full_path + '/' + dirname + '.pkl', 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        wrist_data = data['signal']['wrist']
        labels = data['label']

        eda_df = pd.DataFrame(wrist_data['EDA'], columns=['EDA'])
        bvp_df = pd.DataFrame(wrist_data['BVP'], columns=['BVP'])
        acc_df = pd.DataFrame(wrist_data['ACC'], columns=['ACC_x', 'ACC_y', 'ACC_z'])
        temp_df = pd.DataFrame(wrist_data['TEMP'], columns=['TEMP'])
        label_df = pd.DataFrame(labels, columns=['label'])

    else:
        zip_file_path = os.path.join(full_path, 'input.zip')
        with zipfile.ZipFile(zip_file_path, 'r') as z:
            times_acc, acc_df, start_datetime, _ = read_acc_file(z, 'ACC.csv')
            times_eda, eda_df, _, _ = read_wrist_file(z, 'EDA.csv', column_name='EDA')
            times_bvp, bvp_df, _, _ = read_wrist_file(z, 'BVP.csv', column_name='BVP')
            times_temp, temp_df, _, _ = read_wrist_file(z, 'TEMP.csv', column_name='TEMP')

    # Filter EDA
    eda_df['EDA'] = butter_lowpass_filter(eda_df['EDA'], 1.0, fs_dict['EDA'], 6)

    # Filter ACM
    for _ in acc_df.columns:
        acc_df[_] = filterSignalFIR(acc_df.values)

    # Adding indices for combination due to differing sampling frequencies
    eda_df.index = [(1 / fs_dict['EDA']) * i for i in range(len(eda_df))]
    bvp_df.index = [(1 / fs_dict['BVP']) * i for i in range(len(bvp_df))]
    acc_df.index = [(1 / fs_dict['ACC']) * i for i in range(len(acc_df))]
    temp_df.index = [(1 / fs_dict['TEMP']) * i for i in range(len(temp_df))]

    if train:
        label_df.index = [(1 / fs_dict['label']) * i for i in range(len(label_df))]

    r, p, t, l, d, e, obj = eda_stats(eda_df['EDA'])
    eda_df['EDA_phasic'] = r
    eda_df['EDA_smna'] = p
    eda_df['EDA_tonic'] = t

    df = eda_df.join(bvp_df, how='outer')
    df = df.join(temp_df, how='outer')
    df = df.join(acc_df, how='outer')

    if train:
        df = df.join(label_df, how='outer')
        df['label'] = df['label'].fillna(method='bfill')
    df.reset_index(drop=True, inplace=True)

    if train:
        grouped = df.groupby('label')
        baseline = grouped.get_group(0)
        stress = grouped.get_group(2)
        amusement = grouped.get_group(1)

        n_baseline_wdws = int(len(baseline) / (fs_dict['label'] * WINDOW_IN_SECONDS))
        if n_baseline_wdws != 0:
            baseline_samples = get_samples(baseline, n_baseline_wdws, 0, predicting=False)

        n_stress_wdws = int(len(stress) / (fs_dict['label'] * WINDOW_IN_SECONDS))
        if n_stress_wdws != 0:
            stress_samples = get_samples(stress, n_stress_wdws, 2, predicting=False)

        n_amusement_wdws = int(len(amusement) / (fs_dict['label'] * WINDOW_IN_SECONDS))
        if n_amusement_wdws != 0:
            amusement_samples = get_samples(amusement, n_amusement_wdws, 1, predicting=False)

        dataframes = [baseline_samples, stress_samples, amusement_samples]

        non_empty_dfs = [df for df in dataframes if not df.empty]
        all_samples = pd.concat(non_empty_dfs)
        all_samples = pd.concat([all_samples.drop('label', axis=1), pd.get_dummies(all_samples['label'])], axis=1)

        all_samples.to_csv(f'{full_path}/{dirname}_feats.csv')

    else:
        n_wdws = int(len(df) / (fs_dict['label'] * WINDOW_IN_SECONDS))
        samples = get_samples(df, n_wdws, label=-1, predicting=True)

        desired_columns = [col for col in feats if col not in ['label', 'subject']]
        samples = samples[desired_columns]
        return df, eda_df, bvp_df, temp_df, acc_df, samples, start_datetime


def combine_files(wesad_base_directory, subjects):
    df_list = []
    for s in subjects:
        full_path = os.path.join(wesad_base_directory, s)
        df = pd.read_csv(f'{full_path}/{s}_feats.csv', index_col=0)
        df['subject'] = s
        df_list.append(df)

    df = pd.concat(df_list)

    df['label'] = (df['0'].astype(int).astype(str) + df['1'].astype(int).astype(str) + df['2'].astype(int).astype(
        str)).apply(lambda x: x.find('1'))
    df.drop(['0', '1', '2'], axis=1, inplace=True)

    df.reset_index(drop=True, inplace=True)

    df.to_csv(f'data/WESAD/all_combined_feats.csv')
