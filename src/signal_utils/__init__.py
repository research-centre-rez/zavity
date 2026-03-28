import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.ndimage import gaussian_filter1d


def erf_step(x, a, b, x0, sigma):
    """
    a     : left-side level
    b     : step amplitude (can be positive or negative)
    x0    : sub-sample edge position
    sigma : blur width in samples
    """
    z = (x - x0) / (np.sqrt(2) * sigma)
    return a + 0.5 * b * (1.0 + erf(z))


def find_step_edge_subsample(
    y,
    search=None,
    fit_half_width=20,
    sigma_bounds=(0.3, 20.0),
    presmooth=0.0
):
    """
    Estimate sub-sample location of a single step edge in 1D noisy data.

    Parameters
    ----------
    y : array-like
        1D signal.
    search : tuple(int, int) or None
        Optional coarse search interval [start, end).
    fit_half_width : int
        Half-width of local fitting window around coarse edge.
    sigma_bounds : tuple(float, float)
        Allowed blur-width interval.
    presmooth : float
        Optional light Gaussian smoothing before coarse detection only.
        Usually keep small (0..2). Fit is still done to original y.

    Returns
    -------
    result : dict
        {
            'x0': fitted sub-sample edge position,
            'a': left level,
            'b': step amplitude,
            'sigma': blur width,
            'coarse_idx': coarse estimate,
            'window': (lo, hi),
            'fitted_curve': y_fit_in_window,
            'x_window': x_window
        }
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    x = np.arange(n, dtype=float)

    # ----- coarse detection -----
    ys = y.copy()
    if presmooth > 0:
        ys = gaussian_filter1d(ys, presmooth, mode='nearest')

    if search is None:
        s0, s1 = 0, n
    else:
        s0, s1 = max(0, search[0]), min(n, search[1])

    # derivative-based coarse estimate only
    g = gaussian_filter1d(ys, sigma=2.0 if presmooth == 0 else max(1.0, presmooth), order=1, mode='nearest')
    coarse_idx = s0 + np.argmax(np.abs(g[s0:s1]))

    # ----- local fitting window -----
    lo = max(0, coarse_idx - fit_half_width)
    hi = min(n, coarse_idx + fit_half_width + 1)

    xw = x[lo:hi]
    yw = y[lo:hi]

    # Initial parameter guesses
    k = max(3, len(yw) // 4)
    left_med = np.median(yw[:k])
    right_med = np.median(yw[-k:])
    b0 = right_med - left_med
    a0 = left_med
    x0_0 = float(coarse_idx)
    sigma0 = 2.0

    # If step polarity is opposite locally, that's fine; b can be negative.
    p0 = [a0, b0, x0_0, sigma0]

    # Bounds
    # a unrestricted, b unrestricted, x0 inside local region, sigma positive
    lower = [-np.inf, -np.inf, lo - 1.0, sigma_bounds[0]]
    upper = [ np.inf,  np.inf, hi + 1.0, sigma_bounds[1]]

    popt, pcov = curve_fit(
        erf_step, xw, yw,
        p0=p0,
        bounds=(lower, upper),
        maxfev=20000
    )

    a, b, x0, sigma = popt
    yfit = erf_step(xw, *popt)

    return {
        'x0': float(x0),
        'a': float(a),
        'b': float(b),
        'sigma': float(sigma),
        'coarse_idx': int(coarse_idx),
        'window': (int(lo), int(hi)),
        'fitted_curve': yfit,
        'x_window': xw,
        'cov': pcov,
    }

def circular_signal(angles, min=-90, max=90, shift=0):
    out = []
    for ader in angles:
        if ader > 0:
            out.append(ader + min + shift)
        else:
            out.append(max + ader + shift)
    return np.array(out)


def circular_signal_derivative(angles, sampling, period=180):
    """
    Computes the derivative of a circular signal, compensating for circularity.
    period define range of values i.e. 180 is (-90, 90)
    """
    angle_derivative = angles[sampling:] - angles[:-sampling]  # increase angle diff to distinguish noise and angle change
    # compensate circularity
    derivatives = []
    for ader in angle_derivative:
        if np.abs(ader) < (period // 2):
            derivatives.append(ader)
        elif ader < 0:
            derivatives.append(period + ader)
        else: # ader >=0
            derivatives.append(-np.abs(period - ader))
    return np.array(derivatives)


import numpy as np
from scipy.optimize import curve_fit


def broken_line(x, a, b, c, x0):
    """
    Continuous piecewise linear function with a slope change at x0.
    slope left  = b
    slope right = b + c
    """
    x = np.asarray(x, dtype=float)
    return a + b * x + c * np.maximum(0.0, x - x0)


def find_slope_change_broken_line(y, search=None, fit_half_width=None):
    """
    Estimate sub-sample position where slope changes.

    Parameters
    ----------
    y : array-like
        1D noisy signal
    search : (start, end) or None
        optional interval where the breakpoint is expected
    fit_half_width : int or None
        optional local fitting half-window around coarse estimate

    Returns
    -------
    dict with x0 and fitted parameters
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    x = np.arange(n, dtype=float)

    # optional search interval
    if search is None:
        s0, s1 = 0, n
    else:
        s0, s1 = max(0, search[0]), min(n, search[1])

    # coarse guess:
    # use difference of local slopes from linear fits on left/right windows
    w = max(5, min(20, (s1 - s0) // 10))
    scores = np.full(n, -np.inf)

    for k in range(max(s0 + w, w), min(s1 - w, n - w)):
        xl = x[k-w:k]
        yl = y[k-w:k]
        xr = x[k:k+w]
        yr = y[k:k+w]

        bl = np.polyfit(xl, yl, 1)[0]
        br = np.polyfit(xr, yr, 1)[0]
        scores[k] = abs(br - bl)

    k0 = int(np.argmax(scores[s0:s1]) + s0)

    # optional local window
    if fit_half_width is None:
        lo, hi = s0, s1
    else:
        lo = max(0, k0 - fit_half_width)
        hi = min(n, k0 + fit_half_width + 1)

    xw = x[lo:hi]
    yw = y[lo:hi]

    # initial guess
    p0 = [
        float(np.median(yw)),   # a
        0.0,                    # b
        0.0,                    # c
        float(k0)               # x0
    ]

    bounds = (
        [-np.inf, -np.inf, -np.inf, lo - 1.0],
        [ np.inf,  np.inf,  np.inf, hi + 1.0]
    )

    popt, pcov = curve_fit(
        broken_line,
        xw,
        yw,
        p0=p0,
        bounds=bounds,
        maxfev=20000
    )

    a, b, c, x0 = popt

    return {
        "x0": float(x0),
        "a": float(a),
        "b_left": float(b),
        "b_right": float(b + c),
        "slope_jump": float(c),
        "coarse_idx": int(k0),
        "window": (int(lo), int(hi)),
        "cov": pcov,
        "x_window": xw,
        "y_fit": broken_line(xw, *popt),
    }