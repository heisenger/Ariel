import numpy as np
import preprocess
import pandas as pd
import metrics
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import scipy.signal as signal
from scipy.signal import butter, lfilter
import pywt
import scripts.denoising_models as denoising_models

sys.path.append("../")


star_id = 100468857

data_dir = Path("../data/raw")
target_dir = Path("../data/processed")
train_gt = pd.read_csv(Path(data_dir) / "train_labels.csv")

fgs1_signal, airs_signal = preprocess.preprocess(data_dir, star_id, target_dir)

it = airs_signal[0][70:110, :, 10:20]
oot = np.concatenate(
    [airs_signal[0][0:50, :, 10:20], airs_signal[0][140:, :, 10:20]], axis=0
)


# Simplest prediction
oot_1 = np.mean(np.sum(oot, axis=2), axis=0)
it_1 = np.mean(np.sum(it, axis=2), axis=0)

avg_depth = (oot.sum(axis=(1, 2)).mean() - it.sum(axis=(1, 2)).mean()) / oot.sum(
    axis=(1, 2)
).mean()
print(avg_depth)
oot_1 = np.mean(np.sum(oot, axis=2), axis=0)
it_1 = np.mean(np.sum(it, axis=2), axis=0)

prediction_1 = [None for _ in range(len(oot_1))]

place_holder_count = 0
positive_count = 0
for i in range(len(prediction_1)):
    if np.sum(oot[:, i], axis=1).min() < np.sum(it[:, i], axis=1).max():
        prediction_1[i] = -1
        place_holder_count += 1
    else:
        prediction_1[i] = max(avg_depth, (oot_1[i] - it_1[i]) / (oot_1[i] + 1e-10))
        positive_count += prediction_1[i]

new_placeholder = (avg_depth * len(prediction_1) - positive_count) / place_holder_count

prediction_1 = [
    prediction_1[i] if prediction_1[i] > 0 else new_placeholder
    for i in range(len(prediction_1))
]

# prediction_1 = (oot_1 - it_1) / (oot_1 + 1e-10)


# Remove noise by FFT
cutoff_freq = 0.0005  # Hz
sampling_rate = 1 / (7.5 * 3600 / 185)  # Hz
order = 4

oot_2 = np.sum(oot, axis=2) - np.mean(np.sum(oot, axis=2), axis=0, keepdims=True)
oot_2 = denoising_models.fft_smmoothing(
    oot_2, sampling_rate, order, cutoff_freq
) + np.mean(np.sum(oot, axis=2), axis=0)

oot_2 = np.mean(oot_2, axis=0)

it_2 = np.sum(it, axis=2) - np.mean(np.sum(it, axis=2), axis=0, keepdims=True)
it_2 = denoising_models.fft_smmoothing(
    it_2, sampling_rate, order, cutoff_freq
) + np.mean(np.sum(it, axis=2), axis=0)
it_2 = np.mean(it_2, axis=0)

prediction_2 = (oot_2 - it_2) / (oot_2 + 1e-10)


# Stack wavelet denoising

wavelet_name = "db4"
decomposition_level = 3

oot_3 = np.sum(oot, axis=2) - np.mean(np.sum(oot, axis=2))

oot_3 = denoising_models.wavelet_smoothing(oot_3, 3, wavelet_name)
denoised_oot = np.mean(oot_3, axis=0) + np.mean(np.sum(oot, axis=2))
denoised_oot[:10] = oot_1[:10]

it_3 = np.sum(it, axis=2) - np.mean(np.sum(it, axis=2))

it_3 = denoising_models.wavelet_smoothing(it_3, 3, wavelet_name)
denoised_it = np.mean(it_3, axis=0) + np.mean(np.sum(it, axis=2))
denoised_it[:10] = it_1[:10]

prediction_3 = (denoised_oot - denoised_it) / (denoised_oot + 1e-10)

# Polynomial straightlining


s, x, y, y_new = denoising_models.calibrate_signal(airs_signal[0].sum(axis=(1, 2)))
signal = airs_signal[0].sum(axis=2)
norm_signal = signal - np.mean(signal, axis=0)
signal_cleaned = denoising_models.fft_smmoothing(
    norm_signal, sampling_rate, order, cutoff_freq
) + np.mean(signal, axis=0)

s_2 = denoising_models.calibrate_train(signal_cleaned)
s_2 = [max(0.0058, s_2[i][0]) for i in range(len(s_2))]


def average_closest_five_numpy(arr):
    n = len(arr)
    result = np.zeros(n)
    for i in range(n):
        left = max(0, i - 4)
        right = min(n - 1, i + (8 - i + left))
        window = arr[left : right + 1]
        result[i] = np.mean(window)
    return result


s_2 = average_closest_five_numpy(s_2)

# s, x, y, y_new = denoising.calibrate_signal(signal_poly.sum(axis=(1)))

# plt.plot(airs_signal[0].sum(axis=(1, 2)))
# plt.plot(signal_poly.sum(axis=(1)))
# plt.show()
# print(prediction.shape)
plt.plot(prediction_1, label="No denoising")
# plt.plot(prediction_2, label="FFT Denoising")
# plt.plot(prediction_3, label="FFT + Wavelets")
# plt.plot(s_2, label="Polynomial by frequency")
plt.plot(np.repeat(s, 282), label="Polynomial light curve")

plt.plot(train_gt[train_gt.planet_id == star_id].values[0][:1:-1], label="Ground Truth")
plt.legend()
plt.show()

# plt.plot(oot_2[:, 1])
# plt.plot(np.sum(oot, axis=2)[:, 1])
# plt.show()

planet_gt = pd.DataFrame(train_gt[train_gt.planet_id == star_id].values[0][:1:-1]).T
naive_mean = np.mean(train_gt.values[:, 1:])
naive_sigma = np.std(train_gt.values[:, 1:])

planet_sub = pd.DataFrame(np.hstack((prediction_1, np.array([0.001] * 282)))).T
print(
    "prediction_1:",
    metrics.score(planet_gt, planet_sub, "", naive_mean, naive_sigma, 10**-5),
)

planet_sub = pd.DataFrame(np.hstack((prediction_2, np.array([0.001] * 282)))).T
print(
    "prediction_2:",
    metrics.score(planet_gt, planet_sub, "", naive_mean, naive_sigma, 10**-5),
)

planet_sub = pd.DataFrame(np.hstack((prediction_3, np.array([0.001] * 282)))).T
print(
    "prediction_3:",
    metrics.score(planet_gt, planet_sub, "", naive_mean, naive_sigma, 10**-5),
)

planet_sub = pd.DataFrame(np.hstack((np.array(s_2), np.array([0.001] * 282)))).T
print(
    "prediction_4:",
    metrics.score(planet_gt, planet_sub, "", naive_mean, naive_sigma, 10**-5),
)

planet_sub = pd.DataFrame(np.hstack((np.array([s] * 282), np.array([0.001] * 282)))).T
print(
    "prediction_4:",
    metrics.score(planet_gt, planet_sub, "", naive_mean, naive_sigma, 10**-5),
)
