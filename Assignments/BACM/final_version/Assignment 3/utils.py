import numpy as np
from scipy.signal import savgol_filter


def smooth(y, window, poly):
    y = np.asarray(y)

    # Ensure valid window
    if len(y) < window:
        return y

    if window % 2 == 0:
        window += 1  # must be odd

    window = min(window, len(y) if len(y) % 2 == 1 else len(y) - 1)

    if window < 3:
        return y

    return np.array(savgol_filter(y, window, poly))
