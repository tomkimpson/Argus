"""Module which holds all functions which are related to the properties of the gravitational wave."""

import numpy as np


def pairwise_angular_separation(ra_rad, dec_rad):
    """Compute the pairwise angular separations for a set of celestial coordinates in radians.

    This function takes arrays of right ascension (RA) and declination (Dec), both in radians,
    and returns an NxN matrix of angular separations, where N is the length of the input arrays.
    Each entry (i, j) in the output is the angular separation between the coordinate pair
    (ra_rad[i], dec_rad[i]) and (ra_rad[j], dec_rad[j]).

    Parameters
    ----------
    ra_rad : numpy.ndarray
        1D array of right ascensions in radians, of length N.
    dec_rad : numpy.ndarray
        1D array of declinations in radians, of length N.

    Returns
    -------
    sep_rad : numpy.ndarray
        NxN matrix (2D array) of pairwise angular separations in radians.

    Notes
    -----
    The spherical distance formula used is:

        cos(theta) = sin(dec1) * sin(dec2)
                    + cos(dec1) * cos(dec2) * cos(ra1 - ra2)

    where (ra1, dec1) and (ra2, dec2) are coordinate pairs in radians.

    """
    # Reshape for broadcasting
    ra1 = ra_rad[:, None]
    ra2 = ra_rad[None, :]
    dec1 = dec_rad[:, None]
    dec2 = dec_rad[None, :]

    # Spherical distance formula:
    #   cos(theta) = sin(dec1)*sin(dec2) + cos(dec1)*cos(dec2)*cos(ra1 - ra2)
    cos_sep = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(
        ra1 - ra2
    )

    # Clip values to avoid floating-point errors outside [-1, 1] when taking arccos
    cos_sep = np.clip(cos_sep, -1.0, 1.0)

    # Compute separation in radians
    sep_rad = np.arccos(cos_sep)

    return sep_rad


def hellings_downs(θ):
    """Compute the Hellings–Downs function for an angle θ (in radians).

    Parameters
    ----------
    θ : np.ndarray or float
        Angular separation between pulsars in radians

    Returns
    -------
    np.ndarray or float
        Hellings-Downs correlation values
    """
    # Handle the vector case first
    if isinstance(θ, np.ndarray):
        mask = np.isclose(θ, 0.0)
        x = np.zeros_like(θ)
        # Only compute (1-cos(θ))/2 for non-zero angles
        x[~mask] = (1 - np.cos(θ[~mask])) / 2.0

        result = np.zeros_like(θ)
        result[mask] = 1.0
        # Only compute HD function for non-zero angles
        result[~mask] = (3 / 2) * x[~mask] * np.log(x[~mask]) - x[~mask] / 4 + 0.5

        return result
    else:
        # Handle scalar input
        # Special case for θ = 0 (autocorrelation): x = 0 leads to 0 * log(0) = nan
        # The correct limit as θ → 0 is HD(0) = 1
        if np.isclose(θ, 0.0):
            return 1.0
        x = (1 - np.cos(θ)) / 2.0
        return (3 / 2) * x * np.log(x) - x / 4 + 0.5
