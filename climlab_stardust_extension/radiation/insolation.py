"""Extended insolation utilities.

Provides the ``daily_avg_of_x`` function for computing insolation-weighted
daily averages of quantities that depend on the instantaneous cosine of the
solar zenith angle.  This is used by the aerosol optical-depth table code
to average directional optical properties over the diurnal cycle.
"""

import numpy as np
import xarray as xr
from numpy import sin, cos, tan, arcsin, arccos, deg2rad, pi

from climlab import constants as const
from climlab.solar.insolation import (
    _standardize_inputs,
    solar_longitude,
)


def daily_avg_of_x(lat, day, mu_vect, x_mat, orb=const.orb_present,
                   day_type=1, days_per_year=const.days_per_year, nh=201):
    """Compute the insolation-weighted daily average of a quantity.

    Similar to ``daily_insolation``, but instead of returning insolation
    itself, it returns the daily average of a user-supplied function of
    the instantaneous cosine of the solar zenith angle (mu).

    Parameters
    ----------
    lat : array_like
        Latitude(s) in degrees.
    day : array_like
        Calendar day(s) or solar longitude(s).
    mu_vect : 1-D array
        Cosine-zenith-angle grid on which *x_mat* is defined.
    x_mat : 2-D array, shape (n, len(mu_vect))
        Values to average; each row is a separate quantity.
    orb : dict, optional
        Orbital parameters (default: present-day).
    day_type : {1, 2}
        1 = calendar days, 2 = solar longitude in degrees.
    days_per_year : float, optional
        Length of the year in days.
    nh : int, optional
        Number of hour-angle quadrature points (default 201).

    Returns
    -------
    x_mat_avg : ndarray, shape (n, len(lat))
        Daily-averaged values for each latitude.
    Ho : ndarray
        Hour angle at sunrise/sunset.
    """
    phi, day, ecc, long_peri, obliquity, input_is_xarray, _ignored = \
        _standardize_inputs(lat, day, orb)

    if day_type == 1:
        lambda_long = deg2rad(solar_longitude(day, orb, days_per_year))
    elif day_type == 2:
        lambda_long = deg2rad(day)
    else:
        raise ValueError('Invalid day_type.')

    # Declination angle
    delta = arcsin(sin(obliquity) * sin(lambda_long))
    # Hour angle at sunrise/sunset (Berger 1978)
    Ho = xr.where(
        abs(delta) - pi / 2 + abs(phi) < 0.,
        arccos(-tan(phi) * tan(delta)),
        xr.where(phi * delta > 0., pi, 0.)
    )

    # Compute insolation-averaged quantity
    h = (np.array(Ho)[..., np.newaxis]
         * np.linspace(-1, 1, nh)[np.newaxis, np.newaxis, :])
    hmid = 0.5 * (h[..., :-1] + h[..., 1:])
    dh = np.diff(h, axis=2)
    mu_h = (np.array(sin(delta) * sin(phi))[..., np.newaxis]
            + np.array(cos(delta) * cos(phi))[..., np.newaxis] * cos(hmid))
    mu_h = np.where(mu_h > 0.0, mu_h, 0.0)

    n = x_mat.shape[0]
    x_mat_avg = np.zeros((n, len(phi)))
    for k in range(n):
        x_mat_avg[k, :] = np.mean(
            np.sum(np.interp(mu_h, mu_vect, x_mat[k, :]) * dh, axis=2)
            / (2 * np.pi),
            axis=0,
        )
    if not input_is_xarray:
        Ho = Ho.transpose().values
    return x_mat_avg, Ho
