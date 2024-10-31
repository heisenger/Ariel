import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from numpy.fft import fft, fftfreq
from metrics import simple_score
from scipy.signal import savgol_filter, medfilt, find_peaks
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import pywt
from skimage.restoration import denoise_nl_means, estimate_sigma
import matplotlib.pyplot as plt


def fft_spectrum(signal, sampling_rate):
    """
    Computes the FFT power spectrum of the signal.

    Parameters:
    - signal: The input time series.
    - sampling_rate: The sampling rate to calculate the frequency bins.

    Returns:
    - freqs: The frequency bins.
    - power_spectrum: The power of the FFT at each frequency.
    """
    # Centering the signal by subtracting the mean
    signal_centered = signal - np.mean(signal)
    # Compute the FFT
    fft_result = fft(signal_centered)
    # Compute the power spectrum
    power_spectrum = np.abs(fft_result) ** 2
    # Frequency bins
    freqs = fftfreq(len(fft_result), d=sampling_rate)

    return freqs, power_spectrum


def plot_time_series(time, signal, ax, title="Time Series"):
    """
    Plots the time series on a provided axis.

    Parameters:
    - time: Time points corresponding to the signal.
    - signal: The input time series.
    - ax: The axis on which to plot.
    - title: Title for the plot.
    """
    ax.plot(time, signal)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Signal")
    ax.grid(True)


def plot_fft_spectrum(frequency, fft_values, ax, title="FFT Power Spectrum"):
    """
    Plots the FFT power spectrum on a provided axis.

    Parameters:
    - frequency: Frequency bins from the FFT.
    - fft_values: Power spectrum values from the FFT.
    - ax: The axis on which to plot.
    - title: Title for the plot.
    """
    ax.plot(frequency, fft_values)
    ax.set_title(title)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Power Spectrum")
    ax.grid(True)


def plot_autocorrelation(signal, ax, title="Autocorrelation Function"):
    """
    Plots the autocorrelation function on a provided axis.

    Parameters:
    - signal: The input time series.
    - ax: The axis on which to plot.
    - title: Title for the plot.
    """
    plot_acf(signal, ax=ax, lags=40)
    ax.set_title(title)


def plot_partial_autocorrelation(signal, ax, title="Partial Autocorrelation Function"):
    """
    Plots the partial autocorrelation function on a provided axis.

    Parameters:
    - signal: The input time series.
    - ax: The axis on which to plot.
    - title: Title for the plot.
    """
    plot_pacf(signal, ax=ax, lags=40)
    ax.set_title(title)


def plot_time_series_analysis(signal, sampling_rate):
    """
    Plots a 2x2 grid with the time series, FFT spectrum, autocorrelation,
    and partial autocorrelation functions.

    Parameters:
    - time: Time points corresponding to the signal.
    - signal: The input time series.
    - sampling_rate: The sampling rate to calculate the frequency bins for FFT.
    """
    time = np.arange(len(signal))

    # Compute FFT spectrum
    freqs, power_spectrum = fft_spectrum(signal, sampling_rate)

    # Set up a 2x2 grid of plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot time series
    plot_time_series(time, signal, axes[0, 0], title="Time Series")

    # Plot FFT power spectrum (only positive frequencies)
    positive_freqs = freqs > 0
    plot_fft_spectrum(
        freqs[positive_freqs],
        power_spectrum[positive_freqs],
        axes[0, 1],
        title="FFT Power Spectrum",
    )

    # Plot autocorrelation
    plot_autocorrelation(signal, axes[1, 0], title="Autocorrelation Function")

    # Plot partial autocorrelation
    plot_partial_autocorrelation(
        signal, axes[1, 1], title="Partial Autocorrelation Function"
    )

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_relationships(lc_ratios, wc_ratio, gt):
    """
    Generate plots to visualize the relationships between lc_ratios, wc_ratio, and gt.

    Parameters:
    - lc_ratios: np.ndarray of shape (observations, frequencies)
    - wc_ratio: np.ndarray of shape (observations,)
    - gt: np.ndarray of shape (observations, frequencies)
    """

    # Check if input shapes are consistent
    assert (
        lc_ratios.shape[0] == wc_ratio.shape[0] == gt.shape[0]
    ), "Inconsistent number of observations."
    assert (
        lc_ratios.shape[1] == gt.shape[1]
    ), "Inconsistent number of frequencies between lc_ratios and gt."

    # Calculate means and MSE
    gt_mean = np.mean(gt, axis=1)  # Shape: (observations,)
    lc_mse = np.nanmean(
        (lc_ratios - gt) ** 2, axis=1
    )  # MSE for lc_ratios, shape: (observations,)

    mse_per_freq = np.nanmean(
        (lc_ratios - gt) ** 2, axis=0
    )  # MSE for lc_ratios, shape: (observations,)

    lc_rel_error = np.nanmean(
        (lc_ratios - gt), axis=1
    )  # MSE for lc_ratios, shape: (observations,)

    # Calculate MSE for wc vs gt per observation
    abs_wc_gt = np.abs(wc_ratio - gt.mean(axis=1))  # Shape: (observations,)
    rel_wc_gt = wc_ratio - gt.mean(axis=1)  # Shape: (observations,)

    # Variance of predictions across frequencies
    variance_predictions = np.nanvar(lc_ratios, axis=1)  # Shape: (observations,)

    # Create a 3x2 plot grid
    fig, axs = plt.subplots(3, 2, figsize=(12, 15))

    # Plot 1: gt_mean vs wc_mean
    axs[0, 0].scatter(gt_mean, wc_ratio, color="blue")
    axs[0, 0].set_title("White curve: GT vs Preds")
    axs[0, 0].set_xlabel("GT")
    axs[0, 0].set_ylabel("Pred")
    axs[0, 0].grid(True)

    # Plot 2: MSE across frequency vs gt_mean
    axs[0, 1].scatter(gt_mean, lc_mse, color="red")
    axs[0, 1].set_title("Spectrum MSE vs Mean Flux")
    axs[0, 1].set_xlabel("Mean Flux")
    axs[0, 1].set_ylabel("Specturm MSE")
    axs[0, 1].grid(True)

    # Plot 3: MSE across frequency vs MSE (GT vs WC) per observation
    axs[1, 0].scatter(lc_mse, abs_wc_gt, color="green")
    axs[1, 0].set_title("Spectrum MSE vs WC Abs Error")
    axs[1, 0].set_xlabel("Spectrum MSE")
    axs[1, 0].set_ylabel("WC Abs Error")
    axs[1, 0].grid(True)

    # Plot 4: Variance of predictions across frequency vs MSE across frequency (per observation)
    axs[1, 1].scatter(variance_predictions, lc_mse, color="purple")
    axs[1, 1].set_title("Pred Variance vs Spectrum MSE")
    axs[1, 1].set_xlabel("Pred Variance")
    axs[1, 1].set_ylabel("Spectrum MSE")
    axs[1, 1].grid(True)

    # Plot 5: Check if there is a systematic shift in prediction
    axs[2, 0].scatter(rel_wc_gt, lc_rel_error, color="purple")
    axs[2, 0].set_title("Spectrum vs WC Relative Error")
    axs[2, 0].set_xlabel("WC Signed Error")
    axs[2, 0].set_ylabel("Spectrum Signed Error")
    axs[2, 0].grid(True)

    # Plot 6: Error vs Frequency
    axs[2, 1].plot(mse_per_freq)
    axs[2, 1].set_title("Spectrum MSE vs Frequency")
    axs[2, 1].set_xlabel("Frequency")
    axs[2, 1].set_ylabel("MSE by Frequency")
    axs[2, 1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_score_vs_constant(lc_ratio, gt_selected, constants):
    scores = []

    # Loop through each constant and compute the simple_score
    for const in constants:
        score = simple_score(
            lc_ratio, const * np.ones_like(lc_ratio), gt_selected[:, :-2]
        )
        scores.append(score)

    # Convert scores to numpy array (assuming simple_score returns scalar values)
    scores = np.array(scores)

    # Plot the constant vs score
    plt.figure(figsize=(8, 6))
    plt.plot(constants, scores, label="Score vs Constant", color="blue", marker="o")

    # Add labels and title
    plt.xlabel("Constant")
    plt.ylabel("Score")
    plt.title("Score vs Constant")

    # Show grid and legend
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()


def plot_wavelet_denoised(signal, wavelet="db1", level=1):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    threshold = np.sqrt(2 * np.log(len(signal)))
    new_coeffs = [pywt.threshold(c, threshold, mode="soft") for c in coeffs]
    denoised_signal = pywt.waverec(new_coeffs, wavelet)
    plt.plot(denoised_signal, label="Wavelet Denoised")
    plt.legend()
    plt.show()


def plot_savitzky_golay_filtered(signal, window_length=51, polyorder=3):
    filtered_signal = savgol_filter(signal, window_length, polyorder)
    plt.plot(filtered_signal, label="Savitzky-Golay Filtered")
    plt.legend()
    plt.show()


def plot_kalman_filtered(signal):
    n_iter = len(signal)
    sz = (n_iter,)  # size of array
    Q = 1e-5  # process variance

    # allocate space for arrays
    xhat = np.zeros(sz)  # a posteri estimate of x
    P = np.zeros(sz)  # a posteri error estimate
    xhatminus = np.zeros(sz)  # a priori estimate of x
    Pminus = np.zeros(sz)  # a priori error estimate
    K = np.zeros(sz)  # gain or blending factor

    R = 0.1**2  # estimate of measurement variance, change to see effect

    # intial guesses
    xhat[0] = signal[0]
    P[0] = 1.0

    for k in range(1, n_iter):
        # time update
        xhatminus[k] = xhat[k - 1]
        Pminus[k] = P[k - 1] + Q

        # measurement update
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (signal[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]

    plt.plot(xhat, label="Kalman Filtered")
    plt.legend()
    plt.show()


def plot_median_filtered(signal, kernel_size=5):
    filtered_signal = medfilt(signal, kernel_size)
    plt.plot(filtered_signal, label="Median Filtered")
    plt.legend()
    plt.show()


def plot_non_local_means_denoised(signal, h=1.0, patch_size=5, patch_distance=6):
    sigma_est = np.mean(estimate_sigma(signal))
    denoised_signal = denoise_nl_means(
        signal,
        h=h * sigma_est,
        fast_mode=True,
        patch_size=patch_size,
        patch_distance=patch_distance,
    )  # , multichannel=False)
    plt.plot(denoised_signal, label="Non-Local Means Denoised")
    plt.legend()
    plt.show()


def plot_gaussian_process_regression(signal):
    X = np.arange(len(signal)).reshape(-1, 1)
    y = signal

    kernel = C(1.0, (1e-4, 1e1)) * RBF(1, (1e-4, 1e1))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(X, y)

    y_pred, sigma = gp.predict(X, return_std=True)

    plt.plot(y_pred, label="Gaussian Process Regression")
    plt.fill_between(
        X.flatten(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, alpha=0.2
    )
    plt.legend()
    plt.show()


def plot_spectrogram(time_series, fs=1.0, nfft=128, noverlap=64, cmap="viridis"):
    """
    Plots the spectrogram of a 1D time series data.

    Parameters:
    - time_series: 1D array-like, the time series data to plot.
    - fs: float, the sampling frequency of the time series data.
    - nfft: int, the number of data points used in each block for the FFT.
    - noverlap: int, the number of points of overlap between blocks.
    - cmap: str, the colormap to use for the spectrogram.
    """
    plt.figure(figsize=(10, 6))
    plt.specgram(time_series, NFFT=nfft, Fs=fs, noverlap=noverlap, cmap=cmap)
    plt.title("Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(label="Intensity")
    plt.show()
