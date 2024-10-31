import numpy as np
import pandas as pd
import pandas.api.types
import scipy.stats
from sklearn.metrics import mean_squared_error


class ParticipantVisibleError(Exception):
    pass


def mse(prediction, gt):
    return mean_squared_error(gt, prediction)


def simple_score(
    preds_mean,
    preds_var,
    gt,
    naive_mean=0.002551948412224277,
    naive_var=0.0017262180193330474,
):
    """
    input:
    - preds_mean: observations * freq
    - preds_var: observations * freq
    - gt: observations * freq
    - naive_mean: scalar, result of np.mean(gt_df.values[:,2:])
    - naive_var: scalar, result of np.std(gt_df.values[:,2:])
    """

    GLL_pred = np.sum(scipy.stats.norm.logpdf(gt, loc=preds_mean, scale=preds_var))
    GLL_true = np.sum(
        scipy.stats.norm.logpdf(gt, loc=gt, scale=(10**-5) * np.ones_like(gt))
    )
    GLL_mean = np.sum(
        scipy.stats.norm.logpdf(gt, loc=naive_mean, scale=naive_var * np.ones_like(gt))
    )
    submit_score = (GLL_pred - GLL_mean) / (GLL_true - GLL_mean)

    return float(np.clip(submit_score, 0.0, 1.0))


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    naive_mean: float,
    naive_sigma: float,
    sigma_true: float,
    by_freq=False,
) -> float:
    """
    This is a Gaussian Log Likelihood based metric. For a submission, which contains the predicted mean (x_hat) and variance (x_hat_std),
    we calculate the Gaussian Log-likelihood (GLL) value to the provided ground truth (x). We treat each pair of x_hat,
    x_hat_std as a 1D gaussian, meaning there will be 283 1D gaussian distributions, hence 283 values for each test spectrum,
    the GLL value for one spectrum is the sum of all of them.

    Inputs:
        - solution: Ground Truth spectra (from test set)
            - shape: (nsamples, n_wavelengths)
        - submission: Predicted spectra and errors (from participants)
            - shape: (nsamples, n_wavelengths*2)
        naive_mean: (float) mean from the train set.
        naive_sigma: (float) standard deviation from the train set.
        sigma_true: (float) essentially sets the scale of the outputs.
    """

    # del solution[row_id_column_name]
    # del submission[row_id_column_name]

    if submission.min().min() < 0:
        raise ParticipantVisibleError("Negative values in the submission")
    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f"Submission column {col} must be a number")

    n_wavelengths = len(solution.columns)
    if len(submission.columns) != n_wavelengths * 2:
        raise ParticipantVisibleError("Wrong number of columns in the submission")

    y_pred = submission.iloc[:, :n_wavelengths].values
    # Set a non-zero minimum sigma pred to prevent division by zero errors.
    sigma_pred = np.clip(
        submission.iloc[:, n_wavelengths:].values, a_min=10**-15, a_max=None
    )
    y_true = solution.values

    if by_freq:
        GLL_pred = scipy.stats.norm.logpdf(y_true, loc=y_pred, scale=sigma_pred)
        GLL_true = scipy.stats.norm.logpdf(
            y_true, loc=y_true, scale=sigma_true * np.ones_like(y_true)
        )

        GLL_mean = scipy.stats.norm.logpdf(
            y_true,
            loc=naive_mean * np.ones_like(y_true),
            scale=naive_sigma * np.ones_like(y_true),
        )
        submit_score = (GLL_pred - GLL_mean) / (GLL_true - GLL_mean)
        return np.clip(submit_score, 0.0, 1.0)

    else:

        GLL_pred = np.sum(scipy.stats.norm.logpdf(y_true, loc=y_pred, scale=sigma_pred))
        # print(GLL_pred)
        GLL_true = np.sum(
            scipy.stats.norm.logpdf(
                y_true, loc=y_true, scale=sigma_true * np.ones_like(y_true)
            )
        )
        GLL_mean = np.sum(
            scipy.stats.norm.logpdf(
                y_true,
                loc=naive_mean * np.ones_like(y_true),
                scale=naive_sigma * np.ones_like(y_true),
            )
        )

        submit_score = (GLL_pred - GLL_mean) / (GLL_true - GLL_mean)
        return float(np.clip(submit_score, 0.0, 1.0))
