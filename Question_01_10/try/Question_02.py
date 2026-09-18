"""Question 2: convert a BGR image to grayscale without OpenCV."""

import numpy as np


def BGR2GRAY(img: np.ndarray) -> np.ndarray:
    """Return the luminance image calculated from a BGR image.

    The input uses the BGR channel ordering used by the exercise image.  The
    result is an unsigned 8-bit, two-dimensional grayscale image.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("img must have shape (height, width, 3) in BGR order")

    b, g, r = img.astype(np.float64, copy=False).transpose(2, 0, 1)
    gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
    # The small epsilon prevents values such as 254.99999999999997 for white
    # from being truncated to 254 by the conversion to uint8.
    return np.clip(np.floor(gray + 1e-10), 0, 255).astype(np.uint8)
