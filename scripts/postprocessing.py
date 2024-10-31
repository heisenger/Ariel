import numpy as np


def scale_and_shift_to_target(numbers, target_mean):
    """shifts data series so that it hovers around a certain mean"""
    # Convert to numpy array for easier manipulation
    numbers = np.array(numbers)

    # Calculate the current mean
    current_mean = np.mean(numbers)

    # Shift the values so the new mean becomes the target
    shifted_numbers = numbers - current_mean + target_mean

    return list(shifted_numbers)
