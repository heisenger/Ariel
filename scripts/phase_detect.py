import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, medfilt, find_peaks
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import pywt
from skimage.restoration import denoise_nl_means, estimate_sigma
import matplotlib.pyplot as plt


def smooth_data(data, window_size):
    # x = np.arange(len(data))
    # p = np.polyfit(x, data, 3)  # Linear fit (degree 1)
    # trend = np.polyval(p, x)
    # detrended_data = data - trend

    # return savgol_filter(
    #     detrended_data, window_size, 3
    # )  # window size 51, polynomial order 3
    return savgol_filter(data, window_size, 3)  # window size 51, polynomial order 3


def optimize_breakpoint(
    data, initial_breakpoint=50, window_size=20, buffer_size=3, smooth_window=5
):
    """
    input: 1D time series; first breakpoint guess
    output: first breakpoint
    """
    best_breakpoint = initial_breakpoint
    best_score = float("-inf")
    midpoint = len(data) // 2
    smoothed_data = smooth_data(data, smooth_window)
    # plt.plot(smoothed_data)
    # smoothed_data = data

    data_len = len(data)
    for i in range(-window_size, window_size):
        new_breakpoint = initial_breakpoint + i
        if new_breakpoint - buffer_size < 0 or new_breakpoint + buffer_size >= data_len:
            continue

        region1 = data[: new_breakpoint - buffer_size]
        region2 = data[
            new_breakpoint + buffer_size : data_len - new_breakpoint - buffer_size
        ]
        region3 = data[data_len - new_breakpoint + buffer_size :]

        if len(region1) == 0 or len(region2) == 0 or len(region3) == 0:
            continue

        # calc on smoothed data
        breakpoint_region1 = smoothed_data[
            new_breakpoint - buffer_size : new_breakpoint + buffer_size
        ]
        breakpoint_region2 = smoothed_data[
            -(new_breakpoint + buffer_size) : -(new_breakpoint - buffer_size)
        ]

        # mean_diff = abs(np.mean(region1) - np.mean(region2)) + abs(
        #     np.mean(region2) - np.mean(region3)
        # )
        # var_sum = np.var(region1) + np.var(region2) + np.var(region3)

        range_at_breakpoint1 = np.ptp(
            breakpoint_region1
        )  # ptp: peak to peak (max - min)
        range_at_breakpoint2 = np.ptp(breakpoint_region2)

        mean_range_at_breakpoint = (range_at_breakpoint1 + range_at_breakpoint2) / 2

        score = mean_range_at_breakpoint  # 0 * mean_diff - 0 * var_sum +  * 1

        # print(new_breakpoint, score)
        if score > best_score:
            best_score = score
            best_breakpoint = new_breakpoint
    # print(best_score)

    return best_breakpoint, data_len - best_breakpoint


def optimize_breakpoint_old(
    data, initial_breakpoint=50, window_size=20, buffer_size=5, smooth_window=5
):
    """
    input: 1D time series; first breakpoint guess
    output: first breakpoint
    """
    best_breakpoint = initial_breakpoint
    best_score = float("-inf")
    midpoint = len(data) // 2
    smoothed_data = smooth_data(data, smooth_window)

    for i in range(-window_size, window_size):
        new_breakpoint = initial_breakpoint + i
        region1 = data[: new_breakpoint - buffer_size]
        region2 = data[
            new_breakpoint + buffer_size : len(data) - new_breakpoint - buffer_size
        ]
        region3 = data[len(data) - new_breakpoint + buffer_size :]

        # calc on smoothed data
        breakpoint_region1 = smoothed_data[
            new_breakpoint - buffer_size : new_breakpoint + buffer_size
        ]
        breakpoint_region2 = smoothed_data[
            -(new_breakpoint + buffer_size) : -(new_breakpoint - buffer_size)
        ]

        mean_diff = abs(np.mean(region1) - np.mean(region2)) + abs(
            np.mean(region2) - np.mean(region3)
        )
        var_sum = np.var(region1) + np.var(region2) + np.var(region3)

        range_at_breakpoint1 = np.max(breakpoint_region1) - np.min(breakpoint_region1)
        range_at_breakpoint2 = np.max(breakpoint_region2) - np.min(breakpoint_region2)

        mean_range_at_breakpoint = (range_at_breakpoint1 + range_at_breakpoint2) / 2

        score = mean_diff - 0.5 * var_sum + mean_range_at_breakpoint

        if score > best_score:
            best_score = score
            best_breakpoint = new_breakpoint

    return best_breakpoint, len(data) - best_breakpoint


def adaptive_smooth(data, initial_window=5, max_window=20):
    smoothed = np.copy(data)
    for i in range(len(data)):
        local_window = initial_window
        while local_window <= max_window:
            window_range = range(
                max(0, i - local_window), min(len(data), i + local_window)
            )
            if np.var(data[window_range]) < 0.05:  # Adjust threshold as needed
                break
            local_window += 2
        smoothed[i] = np.mean(
            data[max(0, i - local_window) : min(len(data), i + local_window)]
        )
    return smoothed


def detect_best_breakpoint(
    data, window_size=20, buffer_size=5, initial_smooth_window=5
):
    """
    Detects the best breakpoint automatically across the entire time series.
    """
    best_breakpoint = None
    best_score = float("-inf")
    smoothed_data = adaptive_smooth(data, initial_smooth_window)

    # Start search across all potential breakpoints
    for new_breakpoint in range(window_size, len(data) - window_size):
        region1 = data[: new_breakpoint - buffer_size]
        region2 = data[
            new_breakpoint + buffer_size : len(data) - new_breakpoint - buffer_size
        ]
        region3 = data[len(data) - new_breakpoint + buffer_size :]

        # Calculate on smoothed data
        breakpoint_region1 = smoothed_data[
            new_breakpoint - buffer_size : new_breakpoint + buffer_size
        ]
        breakpoint_region2 = smoothed_data[
            -(new_breakpoint + buffer_size) : -(new_breakpoint - buffer_size)
        ]

        # Mean and variance difference
        mean_diff = abs(np.mean(region1) - np.mean(region2)) + abs(
            np.mean(region2) - np.mean(region3)
        )
        var_sum = np.var(region1) + np.var(region2) + np.var(region3)

        # Range at breakpoints
        range_at_breakpoint1 = np.max(breakpoint_region1) - np.min(breakpoint_region1)
        range_at_breakpoint2 = np.max(breakpoint_region2) - np.min(breakpoint_region2)
        mean_range_at_breakpoint = (range_at_breakpoint1 + range_at_breakpoint2) / 2

        # Fourier component comparison
        fft1 = np.fft.fft(region1)
        fft2 = np.fft.fft(region2)
        fft3 = np.fft.fft(region3)
        fft_diff = np.sum(np.abs(np.abs(fft1) - np.abs(fft2))) + np.sum(
            np.abs(np.abs(fft2) - np.abs(fft3))
        )

        # Gradient magnitude at the breakpoint
        grad_smoothed = np.gradient(smoothed_data)
        gradient_score = np.abs(
            grad_smoothed[new_breakpoint - buffer_size : new_breakpoint + buffer_size]
        ).mean()

        # Updated score calculation
        score = (
            mean_diff
            - 0.5 * var_sum
            + mean_range_at_breakpoint
            + fft_diff
            - 0.3 * gradient_score
        )

        if score > best_score:
            best_score = score
            best_breakpoint = new_breakpoint

    return best_breakpoint


def phase_detector(signal):

    phase1, phase2 = None, None
    best_drop = 0
    for i in range(50 // 2, 150 // 2):
        t1 = signal[i : i + 20 // 2].max() - signal[i : i + 20 // 2].min()
        if t1 > best_drop:
            phase1 = i + (20 + 5) // 2
            best_drop = t1

    best_drop = 0
    for i in range(200 // 2, 250 // 2):
        t1 = signal[i : i + 20 // 2].max() - signal[i : i + 20 // 2].min()
        if t1 > best_drop:
            phase2 = i - 5 // 2
            best_drop = t1

    return phase1, phase2


def detector_runner(timeseries, detector_method):
    """
    input: # obs * t * freq
    output:
        -- LC phases: # obs * freq * 2
        -- WC phases: # obs * 2
    """

    lc_phases = np.zeros((timeseries.shape[0], timeseries.shape[2], 2), dtype=int)
    wc_phases = np.zeros((timeseries.shape[0], 2), dtype=int)

    # wc_timeseries = timeseries.sum(axis=2)

    for i, observation in enumerate(timeseries):
        for freq in range(len(observation[0])):
            lc_phases[i, freq, 0], lc_phases[i, freq, 1] = detector_method(
                observation[:, freq]
            )
        # print(observation.sum(axis=(1)).shape)
        wc_phases[i, 0], wc_phases[i, 1] = detector_method(observation.sum(axis=(1)))

    return lc_phases, wc_phases
