import numpy as np
import pandas as pd


def ratio_from_timeseries(signal, phase_0, phase_1, buffer=5, est_mean=False):
    """
    input: 1D time series; phase_0 (start of eclipse); phase_1 (end of eclipse)
    output: predicted absorption ratio

    """

    # out of transit average flux intensiy
    oot_flux = np.concat((signal[: phase_0 - buffer], signal[phase_1 + buffer :]))
    it_flux = signal[phase_0 + buffer : phase_1 - buffer]

    # avg intensity
    oot_avg = np.mean(oot_flux)
    it_avg = np.mean(it_flux)

    # estimated noise
    oot_var = np.std(oot_flux)
    it_var = np.std(it_flux)

    # calculate weighing
    oot_weight = 1 / np.max((oot_var, 1e-5))
    it_weight = 1 / np.max((it_var, 1e-5))
    obs_weight = np.mean((oot_weight, it_weight))

    # if np.isnan(oot_weight).sum() + np.isnan(it_weight).sum() > 0:
    # print("error", oot_weight, it_weight, it_flux, phase_0, phase_1, signal[])
    # print("ok")
    obs_ratio = np.clip((oot_avg - it_avg) / (it_avg + 1e-6), 0, None)

    # print(oot_avg, it_avg, obs_ratio)
    if est_mean:
        est_ratio = obs_weight * obs_ratio + (1 - obs_weight) * est_mean
        return est_ratio

    else:
        est_ratio = obs_ratio
        return est_ratio


def calculate_spectrum(lc, wc, lc_phases, wc_phases):
    """
    input:
    -- lc: obs * t * freq
    -- wc: obs * t
    -- lc_phases: obs * freq * 2 (1 for start 1 for end of eclipse)
    -- wc_phases: obs * 2 (1 for start 1 for end of eclipse)

    output
    -- lc_ratio: obs * freq
    -- wc_ratio: obs * freq (flat line across all freq; this shows the avg pred)
    """

    lc_ratio = np.zeros((lc.shape[0], lc.shape[2]))
    wc_ratio = np.zeros((wc.shape[0], lc.shape[2]))

    for i in range(len(lc)):

        for f in range(lc.shape[2]):
            freq_timeseries = lc[i, :, f]
            freq_ratio = ratio_from_timeseries(
                freq_timeseries, lc_phases[i][f][0], lc_phases[i][f][1]
            )
            lc_ratio[i, f] = freq_ratio

        wc_ratio[i] = ratio_from_timeseries(wc[i], wc_phases[i][0], wc_phases[i][1])

    return lc_ratio, wc_ratio
