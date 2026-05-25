"""Weak Temperature Gradient relaxation process.

Relaxes tropical virtual temperature toward its area-weighted mean on an
isobaric surface, mimicking the effect of gravity waves that enforce weak
horizontal temperature gradients in the tropics.

Provenance
----------
Ported from ``climlab_stardust/climlab/convection/large_scale_condensation.py``.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import numpy as np
from climlab.utils import constants as const
from climlab.process.time_dependent_process import TimeDependentProcess


# ---------------------------------------------------------------------------
# WeakTemperatureGradient
# ---------------------------------------------------------------------------

class WeakTemperatureGradient(TimeDependentProcess):
    """Weak Temperature Gradient (WTG) relaxation process.

    Relaxes tropical virtual temperature toward its area-weighted mean,
    mimicking the effect of gravity waves that enforce weak temperature
    gradients in the tropics.

    Parameters
    ----------
    relaxation_time : float
        Relaxation timescale in seconds (default: 4 hours).
    lat0 : float
        Latitude boundary for tropical region in degrees (default: 20).
    small : float
        Small parameter (default: 1e-2).
    pmin : float
        Minimum pressure level in hPa (default: 150).
    pmax : float
        Maximum pressure level in hPa (default: 850).
    ps : float
        Surface pressure in hPa (default: 1000).
    scale_func : callable, optional
        Scaling function.
    """

    def __init__(self,
                 relaxation_time=4 * const.seconds_per_hour,
                 lat0=20.0, small=1e-2, **kwargs):
        super().__init__(**kwargs)

        assert 'Tatm' in self.state, (
            f'Tatm should be a state parameter of {self.name}'
        )
        assert 'q' in self.state, (
            f'q should be a state parameter of {self.name}'
        )

        # --- Store parameters ------------------------------------------------
        self.relaxation_time = relaxation_time
        self.small = small
        self.lat0 = lat0
        self.pmin = kwargs.get('pmin', 150.0)
        self.pmax = kwargs.get('pmax', 850.0)
        self.ps = kwargs.get('ps', 1000.0)

        # --- Latitude weighting for tropical average -------------------------
        sin_lat_bound = np.sin(
            np.pi / 180.0 * self.state['Tatm'].domain.lat.bounds
        )
        w = np.where(
            np.abs(self.lat) <= self.lat0,
            np.diff(sin_lat_bound), 0.0,
        )
        self.weight = (w / np.sum(w))[:, np.newaxis]

        self.scale_func = kwargs.get('scale_func', lambda x: np.exp(-x))

    # -----------------------------------------------------------------
    # _compute
    # -----------------------------------------------------------------

    def _compute(self):
        timestep = getattr(self, 'timestep_in_seconds', self.timestep)

        # --- Virtual temperature ---------------------------------------------
        fac = 1.0 + const.eps * self.state['q']
        T_virtual = self.state['Tatm'] * fac

        # --- Pressure coordinates (Pa) --------------------------------------
        p = 1e2 * self.lev[np.newaxis, :]
        p_bounds = 1e2 * self.state['Tatm'].domain.lev.bounds[np.newaxis, :]
        dp = np.diff(p_bounds, axis=-1)

        # --- Lapse rate and Brunt-Vaisala frequency --------------------------
        lapse_d = const.g / const.cp
        dz_dp = -const.Rd * self.state['Tatm'] / const.g / p

        # Extrapolate T to domain boundaries
        Tsurf = (
            (self.Tatm[..., -1] - self.Tatm[..., -2])
            / (p[..., -1] - p[..., -2])
            * (p_bounds[..., -1] - p[..., -1])
            + self.Tatm[..., -1]
        )[:, np.newaxis]
        Ttoa = (
            (self.Tatm[..., 0] - self.Tatm[..., 1])
            / (p[..., 0] - p[..., 1])
            * (p_bounds[..., 0] - p[..., 0])
            + self.Tatm[..., 0]
        )[:, np.newaxis]

        T_bound = np.concatenate(
            (Ttoa, 0.5 * (self.Tatm[..., 1:] + self.Tatm[..., :-1]), Tsurf),
            axis=-1,
        )
        dT_dp = np.diff(T_bound, axis=-1) / dp
        lapse = -dT_dp / dz_dp
        N2 = const.Rd * self.Tatm / p * (lapse_d - lapse)
        N2 = np.where(N2 > 0.0, N2, 0.0)

        # --- Relaxation rate (latitude-dependent) ----------------------------
        tau_relaxation_inv = np.where(
            np.abs(self.lat) <= self.lat0,
            1.0 / self.relaxation_time
            * np.cos(self.lat / self.lat0 * np.pi / 2) ** 2,
            0.0,
        )[:, np.newaxis]

        # --- Area-weighted tropical reference --------------------------------
        T_virtual_ref = np.sum(
            self.weight * T_virtual, axis=0,
        )[np.newaxis, :]

        # --- Temperature tendency --------------------------------------------
        T_tend = (
            (T_virtual_ref - T_virtual)
            * (1.0 - np.exp(-tau_relaxation_inv * timestep))
            / timestep / fac
        )
        T_tend = np.where(self.lev >= self.pmin, T_tend, 0.0)
        T_tend = np.where(self.lev <= self.pmax, T_tend, 0.0)

        return {'Tatm': T_tend}
