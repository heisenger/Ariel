import numpy as np
import matplotlib.pyplot as plt


def idealized_light_curve(
    timeseries,
    absorption_fraction=0.01,
    ingress_instant=0.3,
    egress_instant=0.7,
    buffer=0.05,
):
    """
    Draws an idealized light curve with smooth transitions around ingress and egress instants,
    with buffer regions for gradual dips and rises.

    Parameters:
    - time: 1D numpy array representing time (e.g., phase or time during observation)
    - absorption_fraction: The fraction of light absorbed during the transit (default 1%).
    - ingress_instant: Time at which the ingress begins.
    - egress_instant: Time at which the egress ends.
    - buffer: Time buffer before and after ingress/egress for gradual transitions (default is 0.05).

    Returns:
    - flux: 1D numpy array representing the normalized flux values for the light curve.
    """
    time = np.linspace(0, timeseries.shape[0], timeseries.shape[0])

    # Initialize the flux to 1 (normalized out-of-transit)
    flux = np.ones_like(time)

    # Define the buffer regions for smooth transitions
    ingress_start = ingress_instant - buffer
    ingress_end = ingress_instant + buffer
    egress_start = egress_instant - buffer
    egress_end = egress_instant + buffer

    # Ingress: Flux gradually dips down
    ingress_mask = (time >= ingress_start) & (time <= ingress_end)
    flux[ingress_mask] = 1 - (
        absorption_fraction * (time[ingress_mask] - ingress_start) / (2 * buffer)
    )

    # Egress: Flux gradually rises back up
    egress_mask = (time >= egress_start) & (time <= egress_end)
    flux[egress_mask] = 1 - (
        absorption_fraction * (egress_end - time[egress_mask]) / (2 * buffer)
    )

    # In-transit: Constant dip (between end of ingress and start of egress)
    in_transit_mask = (time > ingress_end) & (time < egress_start)
    flux[in_transit_mask] = 1 - absorption_fraction

    return flux


def generate_ideal_wc(gt, phases, fitted_wc, airs_selected):
    generated_wc = []
    for i in range(len(gt)):
        absorption_fraction = gt[i][:-1].mean()
        ingress_instant = phases[i][0]  # Example ingress instant (30% of the time)
        egress_instant = phases[i][1]  # Example egress instant (70% of the time)
        buffer = 5  # Buffer before and after ingress/egress

        flux = idealized_light_curve(
            airs_selected[i],
            absorption_fraction,
            ingress_instant,
            egress_instant,
            buffer,
        )

        flux = flux * fitted_wc[i][:10].mean()
        # print(flux.shape, fitted_wc[i].shape
        flux = np.hstack(
            [
                flux[: ingress_instant - buffer],
                fitted_wc[i][ingress_instant - buffer : ingress_instant + buffer],
                flux[ingress_instant + buffer : egress_instant - buffer],
                fitted_wc[i][egress_instant - buffer : egress_instant + buffer],
                flux[egress_instant + buffer :],
            ]
        )

        generated_wc.append(flux)
    return np.array(generated_wc)
