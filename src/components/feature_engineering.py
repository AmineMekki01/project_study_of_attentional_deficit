from typing import Tuple
import numpy as np
import pandas as pd
import mne
from scipy.stats import skew, kurtosis
from scipy.fftpack import fft
from scipy.signal import welch
from pywt import wavedec


def get_features(data: mne.epochs, num_channels: int = 8) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    This function takes the data and the number of channels as input. It returns the features, the target and the features dataframe.   

    Parameters  
    ----------  
    data : mne.epochs  
        The data.   

    num_channels : int  
        The number of channels.

    Returns 
    ------- 
    features : numpy.ndarray  
        The features.

    target : numpy.ndarray  
        The target. 

    features_df : pandas.core.frame.DataFrame   
        The features dataframe.

    """

    features = []

    col_names = []

    for epoch in data._data:
        epoch_features = []
        for channel_idx, channel_data in enumerate(epoch):

            mean = np.mean(channel_data)
            var = np.var(channel_data)
            skewness = skew(channel_data)
            kurt = kurtosis(channel_data)

            fft_vals = np.abs(fft(channel_data))
            fft_mean = np.mean(fft_vals)
            fft_var = np.var(fft_vals)
            _, psd = welch(channel_data, fs=256)
            psd_mean = np.mean(psd)
            psd_var = np.var(psd)
            spectral_entropy = -np.sum(psd*np.log2(psd))

            coeffs = wavedec(channel_data, 'db1', level=4)
            wavelet_coeffs_mean = np.mean(coeffs[-1])
            wavelet_coeffs_var = np.var(coeffs[-1])

            first_diff = np.diff(channel_data)
            second_diff = np.diff(first_diff)
            activity = np.var(channel_data)
            mobility = np.sqrt(np.var(first_diff) / activity)
            complexity = np.sqrt(np.var(second_diff) /
                                 np.var(first_diff)) / mobility

            min_val = np.min(channel_data)
            max_val = np.max(channel_data)
            zero_crossings = len(np.where(np.diff(np.sign(channel_data)))[0])

            channel_features = [mean, var, skewness, kurt,
                                fft_mean, fft_var, psd_mean, psd_var, spectral_entropy,
                                wavelet_coeffs_mean, wavelet_coeffs_var,
                                activity, mobility, complexity,
                                min_val, max_val, zero_crossings]

            epoch_features.extend(channel_features)

        features.append(epoch_features)

    channel_names = ['mean', 'var', 'skewness', 'kurt',
                     'fft_mean', 'fft_var', 'psd_mean', 'psd_var', 'spectral_entropy',
                     'wavelet_coeffs_mean', 'wavelet_coeffs_var',
                     'activity', 'mobility', 'complexity',
                     'min_val', 'max_val', 'zero_crossings']

    for i in range(1, num_channels + 1):
        for feature in channel_names:
            col_names.append(f'Ch{i}_{feature}')

    features_df = pd.DataFrame(features, columns=col_names)

    target = data.events[:, -1]

    return np.array(features), target, features_df
