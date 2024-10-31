import sys
import numpy as np
import pandas as pd
from functools import partial
from scipy.optimize import minimize
from pathlib import Path
from scipy.fft import fft, fftfreq
import scipy.signal as signal
from scipy.signal import butter, lfilter, get_window
import pywt
from scripts.phase_detect import optimize_breakpoint
from phase_detect import phase_detector
from preprocess import bin_frequencies


def try_s_with_prior(signal, p1, p2, deg, s, prior):
    out = list(range(p1 - 10)) + list(range(p2 + 10, signal.shape[0]))
    x, y = out, signal[out].tolist()
    x = x + list(range(p1, p2))

    y = y + (signal[p1:p2] * (1 / (1 - s[0]))).tolist()
    z = np.polyfit(x, y, deg)
    p = np.poly1d(z)
    q = np.sqrt(np.mean((p(x) - y) ** 2))

    # Incorporate the prior into the error calculation
    prior_error = np.sqrt(np.mean((p(x) - prior(x)) ** 2))

    return q + prior_error * 0


def calibrate_signal_with_prior(signal, prior):
    """
    input: 1D time series, prior (np.poly1d object)
    output: avg absorption (scalar), time stamp, raw time series, predicted time series
    """

    p1, p2 = phase_detector(signal)

    best_deg, best_score = 1, 1e12
    for deg in range(1, 6):
        f = partial(try_s_with_prior, signal, p1, p2, deg, prior=prior)
        r = minimize(f, [0.001], method="Nelder-Mead")
        s = r.x[0]

        out = list(range(p1 - 10)) + list(range(p2 + 10, signal.shape[0]))
        x, y = out, signal[out].tolist()
        x = x + list(range(p1, p2))
        y = y + (signal[p1:p2] * (1 / (1 - s))).tolist()

        z = np.polyfit(x, y, deg)
        p = np.poly1d(z)
        q = np.sqrt(np.mean((p(x) - y) ** 2))

        if q < best_score:
            best_score = q
            best_deg = deg
            best_s = s

    y = y[: len(out)] + (signal[p1:p2] * (1 / (1 - best_s))).tolist()
    z = np.polyfit(x, y, best_deg)
    p = np.poly1d(z)

    return best_s, x, y, p(x), p


# Polynomial full length calibration
def try_s(signal, p1, p2, deg, s):
    out = np.concatenate((np.arange(p1 - 10), np.arange(p2 + 10, signal.shape[0])))
    x = np.concatenate((out, np.arange(p1, p2)))
    y = np.concatenate((signal[out], signal[p1:p2] * (1 / (1 - s[0]))))

    z = np.polyfit(x, y, deg)
    p = np.poly1d(z)
    q = np.sqrt(np.mean((p(x) - y) ** 2))

    if s < 1e-4:
        return q + 1e3

    return q


def calibrate_signal(signal, ingress, egress, full_output=False):
    """
    input: 1D time series
    output: avg absorption (scalar), time stamp, raw time series, predicted time series
    """

    p1, p2 = ingress, egress

    best_deg, best_score, best_s = 1, 1e12, None
    out = np.concatenate((np.arange(p1 - 10), np.arange(p2 + 10, signal.shape[0])))
    x_out = out
    x_in = np.arange(p1, p2)
    x = np.concatenate((x_out, x_in))

    for deg in range(1, 6):
        f = partial(try_s, signal, p1, p2, deg)
        r = minimize(f, [0.001], method="Nelder-Mead")
        s = r.x[0]

        y = np.concatenate((signal[out], signal[p1:p2] * (1 / (1 - s))))

        z = np.polyfit(x, y, deg)
        p = np.poly1d(z)
        q = np.sqrt(np.mean((p(x) - y) ** 2))

        if q < best_score:
            best_score = q
            best_deg = deg
            best_s = s

    # x = np.concatenate((x_out, x_in))
    y = np.concatenate((signal[out], signal[p1:p2] * (1 / (1 - best_s))))
    z = np.polyfit(x, y, best_deg)
    p = np.poly1d(z)

    if full_output:
        return best_s, x, y, p(x), p
    else:
        return signal - p(np.arange(signal.shape[0])) + signal[0]


def try_s_old(signal, p1, p2, deg, s):
    out = list(range(p1 - 10)) + list(range(p2 + 10, signal.shape[0]))
    x, y = out, signal[out].tolist()
    x = x + list(range(p1, p2))

    y = y + (signal[p1:p2] * (1 / (1 - s[0]))).tolist()
    z = np.polyfit(x, y, deg)
    p = np.poly1d(z)
    q = np.sqrt(np.mean((p(x) - y) ** 2))

    if s < 1e-2:
        return q + 1e3

    return q


def calibrate_signal_old(signal, ingress, egress, full_output=False):
    """
    input: 1D time series
    output: avg absorption (scalar), time stamp, raw time series, predicted time series
    """

    p1, p2 = ingress, egress

    best_deg, best_score = 1, 1e12
    for deg in range(1, 6):
        f = partial(try_s, signal, p1, p2, deg)
        r = minimize(f, [0.001], method="Nelder-Mead")
        s = r.x[0]

        out = list(range(p1 - 10)) + list(range(p2 + 10, signal.shape[0]))
        x, y = out, signal[out].tolist()
        x = x + list(range(p1, p2))
        y = y + (signal[p1:p2] * (1 / (1 - s))).tolist()

        z = np.polyfit(x, y, deg)
        p = np.poly1d(z)
        q = np.sqrt(np.mean((p(x) - y) ** 2))  # np.abs(p(x) - y).mean()

        if q < best_score:
            best_score = q
            best_deg = deg
            best_s = s

    y = y[: len(out)] + (signal[p1:p2] * (1 / (1 - best_s))).tolist()
    z = np.polyfit(x, y, best_deg)
    p = np.poly1d(z)

    if full_output:
        return best_s, x, y, p(x), p

    else:
        return signal - p(np.arange(signal.shape[0])) + signal[0]


def calibrate_train(signal_cube):
    """Calibrates each time series (t, m) in the signal_cube (t, k, m) separately.
    input: t (time) * k (frequency)
    output: absorption, time stamp, original signal, fitted signal"""
    results = []

    if signal_cube.ndim == 1:
        signal_cube = signal_cube[:, np.newaxis]

    for k in range(signal_cube.shape[1]):  # Iterate over the 'k' dimension
        signal = signal_cube[:, k]
        # p1, p2 = (
        #     70,
        #     125,
        # )  # Placeholder, you can replace it with phase_detector(signal) if necessary.

        p1 = optimize_breakpoint(signal, 40)
        p2 = len(signal_cube) - p1

        p1, p2 = phase_detector(signal)
        best_deg, best_score = 1, 1e12
        best_s, best_p1, best_p2 = None, None, None

        # Define the 'out' portion, which is outside the [p1:p2] range
        out = list(range(p1 - 5)) + list(range(p2 + 5, signal.shape[0]))
        max_out_value = max(signal[out])  # Maximum value in the out portion

        for deg in range(1, 6):
            # Modify the objective function to ensure that the signal[p1:p2] is less than or equal to max_out_value
            def constrained_try_s(s):
                y_scaled = signal[p1:p2] * (1 + s)
                if np.any(
                    y_scaled > max_out_value
                ):  # Ensure the scaled signal stays below the max_out_value
                    return 1e6  # Return a large penalty to discourage this solution

                out_points = list(range(p1 - 30)) + list(
                    range(p2 + 30, signal.shape[0])
                )
                x, y = out_points, signal[out_points].tolist()
                x = x + list(range(p1, p2))
                y = y + y_scaled.tolist()

                z = np.polyfit(x, y, deg)
                p = np.poly1d(z)
                q = np.abs(p(x) - y).mean()

                return q

            # Minimize with the constrained objective function
            r = minimize(constrained_try_s, [0.0001], method="Nelder-Mead")
            s = r.x[0]

            # Verify if the solution is the best so far
            if r.fun < best_score:
                best_score = r.fun
                best_deg = deg
                best_s = s
                best_p1, best_p2 = p1, p2

        # After finding the best_s, apply it and compute the final interpolation
        x = list(range(best_p1 - 30)) + list(range(best_p2 + 30, signal.shape[0]))
        y = signal[x].tolist()
        y = y + (signal[best_p1:best_p2] * (1 + best_s)).tolist()

        z = np.polyfit(x + list(range(best_p1, best_p2)), y, best_deg)
        p = np.poly1d(z)

        results.append(
            (
                best_s,
                x + list(range(best_p1, best_p2)),
                y,
                p(x + list(range(best_p1, best_p2))),
            )
        )

    return results


# polyfit model
def polyfit_model(time_series, break_point_1, break_point_2, buffer=5, degree=5):
    """
    input: 1D time series, breakpoint_1, breakpoint_2
    output: 1D cleaned time series, absorption prdcition
    """

    # separate the three segments of our time series
    oot_1_flux = time_series[: break_point_1 - buffer]
    transit_flux = time_series[break_point_1 + buffer : break_point_2 - buffer]
    oot_2_flux = time_series[break_point_2 + buffer :]

    time_series = [oot_1_flux, transit_flux, oot_2_flux]
    coeff_list = []
    poly_trend_list = []

    # fit the polynomial on the three segments
    for i, flux in enumerate(time_series):
        time_list = np.linspace(0, len(flux), len(flux))
        coeff_list.append(np.polyfit(time_list, flux, deg=degree))
        poly_trend_list.append(np.polyval(coeff_list[i], time_list))

    oot_1_flux_cleaned = (
        oot_1_flux - poly_trend_list[0] + np.mean(poly_trend_list[0][:5])
    )

    transit_flux_cleaned = (
        transit_flux
        - poly_trend_list[1]
        + oot_1_flux_cleaned[-1]
        - (np.mean(oot_1_flux) - np.mean(transit_flux)) * 0
        - (oot_1_flux[-1] - transit_flux[-1])
    )

    oot_2_flux_cleaned = (oot_2_flux - poly_trend_list[2] + poly_trend_list[2][0]) * (
        oot_1_flux_cleaned[0] / poly_trend_list[2][0]
    )

    # combine cleaned data:
    fitted_time_series = np.hstack(
        (
            oot_1_flux_cleaned,
            np.repeat(oot_1_flux_cleaned[-1], buffer),
            np.repeat(transit_flux_cleaned[0], buffer),
            transit_flux_cleaned,
            np.repeat(transit_flux_cleaned[-1], buffer),
            np.repeat(oot_2_flux_cleaned[0], buffer),
            oot_2_flux_cleaned,
        )
    )

    flux_ratio = 1 - (
        np.mean(transit_flux_cleaned)
        / np.mean(np.concatenate((oot_1_flux_cleaned, oot_2_flux_cleaned)))
    )

    return fitted_time_series, flux_ratio


def model_runner(timeseries, lc_phases, wc_phases, method):
    """
    input:
    -- timeseries: obs * t * freq
    -- lc_phases: obs * freq * 2
    -- wc_phases: obs * 2

    output:
    -- fitted_lc: obs * t * freq
    -- fitted_wc: obs * t
    -- lc_flux_ratio: obs * freq
    -- wc_flux_ratio: obs
    """

    fitted_lc = np.zeros(timeseries.shape, dtype=int)
    fitted_wc = np.zeros((timeseries.shape[0], timeseries.shape[1]), dtype=int)

    for i, observation in enumerate(timeseries):
        for j in range(timeseries.shape[2]):
            lc_phase_1 = lc_phases[i, j, 0]
            lc_phase_2 = lc_phases[i, j, 1]

            # print("testing", lc_phase_1, lc_phase_2)
            # print(timeseries[i, :, j])
            fitted_lc[i, :, j] = method(
                timeseries[i, :, j], lc_phase_1 + 5, lc_phase_2 - 5
            )  # include buffer here
        wc_phase_1 = wc_phases[i, 0] + 5  # include buffer here
        wc_phase_2 = wc_phases[i, 1] - 5  # include buffer here

        fitted_wc[i] = method(
            signal.medfilt(timeseries[i].sum(axis=1), 5), wc_phase_1, wc_phase_2
        )

    return fitted_lc, fitted_wc


def fft_smmoothing(signal, sampling_rate, order, cutoff_freq, normalized=False):
    # window = get_window("hamming", signal.shape[0])
    # signal_windowed = signal * window[:, np.newaxis]

    if signal.ndim == 1:
        signal = signal[:, np.newaxis]

    norm_factor = 0
    if normalized:
        norm_factor = np.mean(signal, axis=0)
        signal = signal - norm_factor

    padding = 10  # Adjust the amount of padding
    signal_padded = np.pad(signal, ((padding, padding), (0, 0)), mode="reflect")

    fft_result = fft(signal_padded, axis=0)

    b, a = butter(order, cutoff_freq, fs=sampling_rate, btype="lowpass")

    filtered_signal = np.zeros_like(signal_padded)

    # Apply the filter on the real part of the IFFT
    for i in range(signal.shape[1]):  # Loop over the k time series
        filtered_signal[:, i] = lfilter(b, a, np.real(np.fft.ifft(fft_result[:, i])))

    return filtered_signal[padding:-padding] + norm_factor
    # filtered_signal = lfilter(
    #     b, a, np.real(np.fft.ifft(fft_result))
    # )  # Apply inverse FFT and take real part
    # return filtered_signal


def wavelet_smoothing(signal, decomposition_level, wavelet_name="db4"):
    """
    input: observations * t * freq (20, 187, 282)
    output: observations * t * freq (20, 187, 282)
    """
    # Center the data by subtracting the mean
    mean_value = np.mean(
        signal, axis=(1), keepdims=True
    )  # Mean across time and frequency
    centered_signal = signal - mean_value

    # Apply wavelet decomposition along the frequency axis (axis=2)
    coeffs = pywt.wavedec(
        centered_signal, wavelet_name, level=decomposition_level, axis=2
    )

    # Calculate the threshold based on the last coefficient
    threshold = (
        0.5 * np.median(np.abs(coeffs[-1])) * np.sqrt(2 * np.log(signal.shape[2]))
    )

    # Apply soft thresholding to wavelet coefficients
    coeffs = [pywt.threshold(c, threshold, mode="soft") for c in coeffs]

    # Reconstruct the signal after thresholding along the frequency axis
    smoothed_signal = pywt.waverec(coeffs, wavelet_name, axis=2)

    # Ensure the output shape matches the input shape
    smoothed_signal = smoothed_signal[:, :, : signal.shape[2]]

    # Add the mean back to the smoothed signal
    smoothed_signal += mean_value

    return smoothed_signal


def polyfit_model_2(
    time_series, break_point_1, break_point_2, buffer=5, degree_transit=5, degree_oot=2
):
    """
    input: 1D time series, break_point_1, break_point_2
    output: 1D cleaned time series, absorption prediction
    """

    # Normalize the time series by dividing by its median (out-of-transit portion)
    median_flux = np.median(
        np.concatenate((time_series[:break_point_1], time_series[break_point_2:]))
    )
    time_series_normalized = time_series / median_flux

    # Separate the three segments of the time series
    oot_1_flux = time_series_normalized[: break_point_1 - buffer]
    transit_flux = time_series_normalized[
        break_point_1 + buffer : break_point_2 - buffer
    ]
    oot_2_flux = time_series_normalized[break_point_2 + buffer :]

    # Fit the polynomial to each segment (lower degree for out-of-transit flux, higher for transit)
    time_list_oot_1 = np.linspace(0, len(oot_1_flux) - 1, len(oot_1_flux))
    time_list_transit = np.linspace(0, len(transit_flux) - 1, len(transit_flux))
    time_list_oot_2 = np.linspace(0, len(oot_2_flux) - 1, len(oot_2_flux))

    # Polynomial fitting with different degrees for transit and out-of-transit
    oot_1_poly_coeff = np.polyfit(time_list_oot_1, oot_1_flux, deg=degree_oot)
    transit_poly_coeff = np.polyfit(time_list_transit, transit_flux, deg=degree_transit)
    oot_2_poly_coeff = np.polyfit(time_list_oot_2, oot_2_flux, deg=degree_oot)

    # Polynomial trend lines for each segment
    oot_1_trend = np.polyval(oot_1_poly_coeff, time_list_oot_1)
    transit_trend = np.polyval(transit_poly_coeff, time_list_transit)
    oot_2_trend = np.polyval(oot_2_poly_coeff, time_list_oot_2)

    # Remove the trend from each segment to get cleaned flux
    oot_1_flux_cleaned = oot_1_flux - oot_1_trend + np.mean(oot_1_trend[:5])
    transit_flux_cleaned = transit_flux - transit_trend + oot_1_flux_cleaned[-1]
    oot_2_flux_cleaned = oot_2_flux - oot_2_trend + oot_1_flux_cleaned[0]

    # Smooth transition between oot_1 and transit, and between transit and oot_2
    def smooth_transition(flux1, flux2, overlap=5):
        transition = np.linspace(0, 1, overlap)
        smoothed_flux = (1 - transition) * flux1[-overlap:] + transition * flux2[
            :overlap
        ]
        return smoothed_flux  # np.hstack((flux1[:-overlap], smoothed_flux, flux2[overlap:]))

    # Ensure buffer overlap doesn't exceed segment lengths
    buffer = min(
        buffer,
        len(oot_1_flux_cleaned),
        len(oot_2_flux_cleaned),
        len(transit_flux_cleaned),
    )

    # Combine the cleaned data with smooth transitions
    fitted_time_series = np.hstack(
        (
            oot_1_flux_cleaned,
            smooth_transition(
                oot_1_flux_cleaned, transit_flux_cleaned, overlap=buffer * 2
            ),
            transit_flux_cleaned,
            smooth_transition(
                transit_flux_cleaned, oot_2_flux_cleaned, overlap=buffer * 2
            ),
            oot_2_flux_cleaned,
        )
    )

    # print(len(fitted_time_series))

    # Calculate flux ratio (depth of transit)
    flux_ratio = 1 - (
        np.mean(transit_flux_cleaned)
        / np.mean(np.concatenate((oot_1_flux_cleaned, oot_2_flux_cleaned)))
    )

    return fitted_time_series, flux_ratio


def median_denoising(time_series, window_size):
    """
    input: observations * t * freq (20, 187, 282)
    output: observations * t * freq (20, 187, 282)
    """
    # Apply median filter to the time series
    cleaned_time_series = signal.medfilt(time_series, window_size)

    return cleaned_time_series
