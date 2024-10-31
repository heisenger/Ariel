import numpy as np
import pandas as pd
from pathlib import Path
import itertools
import os
from astropy.stats import sigma_clip


def ADC_convert(signal, gain, offset):
    signal = signal.astype(np.float64)
    signal /= gain
    signal += offset
    return signal


def mask_hot_dead(signal, dead, dark):
    hot = sigma_clip(dark, sigma=5, maxiters=5).mask
    hot = np.tile(hot, (signal.shape[0], 1, 1))
    dead = np.tile(dead, (signal.shape[0], 1, 1))
    signal = np.ma.masked_where(dead, signal)
    signal = np.ma.masked_where(hot, signal)
    return signal


def apply_linear_corr(linear_corr, clean_signal):
    linear_corr = np.flip(linear_corr, axis=0)
    for x, y in itertools.product(
        range(clean_signal.shape[1]), range(clean_signal.shape[2])
    ):
        poli = np.poly1d(linear_corr[:, x, y])
        clean_signal[:, x, y] = poli(clean_signal[:, x, y])
    return clean_signal


def clean_dark(signal, dead, dark, dt):

    dark = np.ma.masked_where(dead, dark)
    dark = np.tile(dark, (signal.shape[0], 1, 1))

    signal -= dark * dt[:, np.newaxis, np.newaxis]
    return signal


def get_cds(signal):
    cds = signal[:, 1::2, :, :] - signal[:, ::2, :, :]
    return cds


def bin_obs(cds_signal, binning):
    cds_transposed = cds_signal.transpose(0, 1, 3, 2)
    cds_binned = np.zeros(
        (
            cds_transposed.shape[0],
            cds_transposed.shape[1] // binning,
            cds_transposed.shape[2],
            cds_transposed.shape[3],
        )
    )
    for i in range(cds_transposed.shape[1] // binning):
        cds_binned[:, i, :, :] = np.sum(
            cds_transposed[:, i * binning : (i + 1) * binning, :, :], axis=1
        )
    return cds_binned


def correct_flat_field(flat, dead, signal):
    flat = flat.transpose(1, 0)
    dead = dead.transpose(1, 0)
    flat = np.ma.masked_where(dead, flat)
    flat = np.tile(flat, (signal.shape[0], 1, 1))
    signal = signal / flat
    return signal


def read_adjustment_files(data_dir, planet_id, cut_inf, cut_sup):

    folder_dir = Path(data_dir) / "train" / str(planet_id)

    adc = pd.read_csv(Path(data_dir) / "train_adc_info.csv")
    adc_info = adc[adc.planet_id == planet_id]

    axis_df = pd.read_parquet(Path(data_dir) / "axis_info.parquet")

    dt_airs = axis_df["AIRS-CH0-integration_time"].dropna().values
    dt_airs[1::2] += 0.1

    airs_raw = pd.read_parquet(Path(folder_dir) / "AIRS-CH0_signal.parquet")
    fgs1_raw = pd.read_parquet(Path(folder_dir) / "FGS1_signal.parquet")

    flat_airs = (
        pd.read_parquet(Path(folder_dir) / "AIRS-CH0_calibration" / "flat.parquet")
        .values.astype(np.float64)
        .reshape((32, 356))[:, cut_inf:cut_sup]
    )

    flat_fgs1 = (
        pd.read_parquet(Path(folder_dir) / "FGS1_calibration" / "flat.parquet")
        .values.astype(np.float64)
        .reshape((32, 32))
    )
    dark_airs = (
        pd.read_parquet(Path(folder_dir) / "AIRS-CH0_calibration" / "dark.parquet")
        .values.astype(np.float64)
        .reshape((32, 356))[:, cut_inf:cut_sup]
    )
    dark_fgs1 = (
        pd.read_parquet(Path(folder_dir) / "FGS1_calibration" / "dark.parquet")
        .values.astype(np.float64)
        .reshape((32, 32))
    )
    dead_airs = (
        pd.read_parquet(Path(folder_dir) / "AIRS-CH0_calibration" / "dead.parquet")
        .values.astype(np.float64)
        .reshape((32, 356))[:, cut_inf:cut_sup]
    )
    dead_fgs1 = (
        pd.read_parquet(Path(folder_dir) / "FGS1_calibration" / "dead.parquet")
        .values.astype(np.float64)
        .reshape((32, 32))
    )
    linear_corr_airs = (
        pd.read_parquet(
            Path(folder_dir) / "AIRS-CH0_calibration" / "linear_corr.parquet"
        )
        .values.astype(np.float64)
        .reshape((6, 32, 356))[:, :, cut_inf:cut_sup]
    )
    linear_corr_fgs1 = (
        pd.read_parquet(Path(folder_dir) / "FGS1_calibration" / "linear_corr.parquet")
        .values.astype(np.float64)
        .reshape((6, 32, 32))
    )

    return (
        adc_info,
        axis_df,
        flat_airs,
        flat_fgs1,
        dark_airs,
        dark_fgs1,
        dead_airs,
        dead_fgs1,
        linear_corr_airs,
        linear_corr_fgs1,
        dt_airs,
    )


def read_raw_files(data_dir, star_id):
    airs_raw = pd.read_parquet(
        Path(data_dir) / "train" / str(star_id) / "AIRS-CH0_signal.parquet"
    )

    fgs1_raw = pd.read_parquet(
        Path(data_dir) / "train" / str(star_id) / "FGS1_signal.parquet"
    )

    airs_raw = airs_raw.values.astype(np.float64).reshape((airs_raw.shape[0], 32, 356))
    fgs1_raw = fgs1_raw.values.astype(np.float64).reshape((fgs1_raw.shape[0], 32, 32))

    return airs_raw, fgs1_raw


def preprocess(data_dir, star_id, target_dir):

    airs_path = Path(target_dir) / str(str(star_id) + "_airs.npy")
    os.makedirs(os.path.dirname(airs_path), exist_ok=True)

    fgs1_path = Path(target_dir) / str(str(star_id) + "_fgs1.npy")
    os.makedirs(os.path.dirname(fgs1_path), exist_ok=True)

    if os.path.exists(airs_path) and os.path.exists(fgs1_path):
        airs_signal = np.load(airs_path)
        fgs1_signal = np.load(fgs1_path)

    else:
        # cut away signals
        cut_inf, cut_sup = 39, 321

        airs_raw, fgs1_raw = read_raw_files(data_dir, star_id)

        # read the adjustment dfs
        (
            adc_info,
            axis_df,
            flat_airs,
            flat_fgs1,
            dark_airs,
            dark_fgs1,
            dead_airs,
            dead_fgs1,
            linear_corr_airs,
            linear_corr_fgs1,
            dt_airs,
        ) = read_adjustment_files(data_dir, star_id, cut_inf, cut_sup)

        gain, offset = (
            adc_info["AIRS-CH0_adc_gain"].values,
            adc_info["AIRS-CH0_adc_offset"].values,
        )

        fgs1_gain, fgs1_offset = (
            adc_info["FGS1_adc_gain"].values,
            adc_info["FGS1_adc_offset"].values,
        )

        airs_signal = ADC_convert(airs_raw, gain[0], offset[0])
        airs_signal = airs_signal[:, :, cut_inf:cut_sup]

        # mask dead / dark pixels
        airs_signal = mask_hot_dead(airs_signal, dead_airs, dark_airs)
        # print('masked dead / dark pixels', airs_signal[:1])

        # Linearity correction
        airs_signal = apply_linear_corr(linear_corr_airs, airs_signal)
        # print('linearity correction', airs_signal[:1])

        # Remove dark current
        airs_signal = clean_dark(airs_signal, dead_airs, dark_airs, dt_airs)
        # print('dark current removal', airs_signal[:1])

        airs_signal = airs_signal.reshape((1, 11250, 32, 282))
        airs_signal = get_cds(airs_signal)

        airs_signal = bin_obs(airs_signal, binning=30)
        airs_signal = correct_flat_field(flat_airs, dead_airs, airs_signal)

        # get full dynamic range
        fgs1_signal = ADC_convert(fgs1_raw, fgs1_gain[0], fgs1_offset[0])
        dt_fgs1 = np.ones(len(fgs1_signal)) * 0.1
        # print('ADC Converted', fgs1_signal[:1])

        # mask dead / dark pixels
        fgs1_signal = mask_hot_dead(fgs1_signal, dead_fgs1, dark_fgs1)
        # print('masked dead / dark pixels', fgs1_signal[:1])

        # Linearity correction
        fgs1_signal = apply_linear_corr(linear_corr_fgs1, fgs1_signal)
        # print('linearity correction', fgs1_signal[:1])

        # Remove dark current
        fgs1_signal = clean_dark(fgs1_signal, dead_fgs1, dark_fgs1, dt_fgs1)
        # print('dark current removal', fgs1_signal[:1])
        fgs1_signal = fgs1_signal.reshape((1, 135000, 32, 32))
        fgs1_signal = get_cds(fgs1_signal)

        fgs1_signal = bin_obs(fgs1_signal, binning=30 * 12)
        fgs1_signal = correct_flat_field(flat_fgs1, dead_fgs1, fgs1_signal)

        AIRS_CH0_clean = np.zeros(airs_signal.shape)
        FGS1_clean = np.zeros(fgs1_signal.shape)

        print(AIRS_CH0_clean.shape, airs_signal.shape)
        AIRS_CH0_clean[:] = airs_signal[:]
        FGS1_clean[:] = fgs1_signal[:]

        np.save(airs_path, AIRS_CH0_clean)
        np.save(fgs1_path, FGS1_clean)

    return fgs1_signal, airs_signal


def subtract_background(timeseries, edge_pixel=5):
    """
    input: observations * t * freq * spatial
    output: observations * t * freq * spatial

    """
    # compute average of the edge pixel, treat them as background noise
    first_5_rows = timeseries[:, :, :, :edge_pixel]
    last_5_rows = timeseries[:, :, :, -edge_pixel:]

    background_noise = (
        np.mean(first_5_rows, axis=3) + np.mean(last_5_rows, axis=3)
    ) / 2

    print(background_noise.shape)

    # Step 3: Subtract the background noise from all rows along axis 2
    # Use broadcasting to subtract it from each row of the 3rd axis
    data_corrected = timeseries - background_noise[:, :, :, np.newaxis]

    return data_corrected


def spatial_integration(timeseries, range_start=8, range_end=24):
    """
    input: observation * t * freq * spatial
    output: observation * t * freq

    """

    timeseries = timeseries[:, :, :, range_start:range_end].sum(axis=3)

    return timeseries


def bin_frequencies(time_series, k):
    """
    Bin the frequencies of a time series such that each bin has roughly equal intensity.

    Parameters:
    - time_series: numpy.ndarray, the time series data of shape (observations * time_steps, frequency).
    - k: int, the number of bins.

    Returns:
    - binned_series: numpy.ndarray, the binned time series data of shape (time_steps, k).
    """

    full_binned_series = np.zeros_like(time_series)
    # full_binned_series = np.zeros((time_series.shape[0], time_series.shape[1], k))

    for i in range(time_series.shape[0]):
        # Compute the total intensity for each frequency
        total_intensity = np.sum(time_series[i], axis=0)

        # Compute the cumulative intensity
        cumulative_intensity = np.cumsum(total_intensity)

        # Determine the total intensity and the target intensity per bin
        total_intensity_sum = cumulative_intensity[-1]
        target_intensity_per_bin = total_intensity_sum / k

        # Initialize the binned series
        time_steps, freq = time_series[i].shape
        binned_series = np.zeros((time_steps, k))

        # Bin the frequencies
        bin_idx = 0
        bin_intensity = 0
        bin_indices = [0]
        for j in range(freq):
            if (
                bin_intensity + total_intensity[j] > target_intensity_per_bin
                and bin_idx < k - 1
            ):
                bin_indices.append(j)
                bin_idx += 1
                bin_intensity = 0
            binned_series[:, bin_idx] += time_series[i, :, j]
            bin_intensity += total_intensity[j]
        bin_indices.append(j + 1)
        # print(bin_indices)
        # print(binned_series.shape)
        for bin_index in range(len(bin_indices) - 1):
            full_binned_series[i][
                :, bin_indices[bin_index] : bin_indices[bin_index + 1]
            ] = binned_series[:, bin_index].reshape(-1, 1)
    return full_binned_series
