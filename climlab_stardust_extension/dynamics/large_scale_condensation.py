"""Extended large-scale condensation process.

Inherits from ``climlab.dynamics.large_scale_condensation.LargeScaleCondensation``
and adds features from the ``climlab_stardust`` fork:

* Array-valued ``RH_ref`` (latitude- and level-dependent threshold)
* Custom moisture sink function (``sink_func``)
* Pressure cutoff (``pmin``) to suppress condensation in the stratosphere
* ``qsat_extended`` for ERA5 / simplified saturation options
* Exponential relaxation ``(1 - exp(-dt/tau))/dt`` instead of ``1/tau``
* Multi-latitude precipitation handling
* Additional diagnostic: ``precipitation_2D`` (per-level condensation rate)

The base class lives at ``climlab.dynamics.large_scale_condensation`` (official
climlab). This extension follows the project convention of ``_extended`` suffixed
classes that add stardust functionality on top of the official API.

Provenance
----------
Ported from ``climlab_stardust/climlab/convection/large_scale_condensation.py``.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import numpy as np
from climlab.utils import constants as const
from climlab.dynamics.large_scale_condensation import (
    LargeScaleCondensation as _BaseLSC,
)

from climlab_stardust_extension.utils.thermo import qsat_extended


# ---------------------------------------------------------------------------
# LargeScaleCondensation_extended
# ---------------------------------------------------------------------------

class LargeScaleCondensation_extended(_BaseLSC):
    """Extended large-scale condensation with stardust features.

    Parameters
    ----------
    condensation_time : float
        Relaxation timescale in seconds (default: 4 hours).
    RH_ref : float or ndarray
        Reference relative humidity threshold (default: 0.9).  If an ndarray
        is passed it is broadcast to the shape of ``q``.
    sink_func : callable, optional
        Custom moisture sink function ``f(state) -> tendency``.  When given,
        the standard relaxation formula is bypassed entirely.
    pmin : float
        Minimum pressure level in hPa; condensation tendencies are zeroed
        above this level (default: 10.0).
    rh_small : float, optional
        Regularisation parameter forwarded to ``qsat_extended``.
    do_era5 : bool, optional
        Use ERA5-compatible saturation formula.
    do_simplified : bool, optional
        Use simplified saturation formula.

    Diagnostics
    -----------
    latent_heating : ndarray
        Latent heating rate per grid cell (W m-2).
    precipitation : ndarray
        Column-integrated precipitation rate (kg m-2 s-1).
    precipitation_2D : ndarray
        Per-level precipitation rate (kg m-2 s-1).
    """

    def __init__(self,
                 condensation_time=4 * const.seconds_per_hour,
                 RH_ref=0.9,
                 **kwargs):
        # --- Extract extension-specific kwargs before calling super ----------
        # These are not recognised by the base class, so pop them now.
        sink_func = kwargs.pop('sink_func', None)
        pmin = kwargs.pop('pmin', 10.0)
        small_dict = (
            {'small': kwargs.pop('rh_small')} if 'rh_small' in kwargs else {}
        )
        do_era5_dict = (
            {'do_era5': kwargs.pop('do_era5')} if 'do_era5' in kwargs else {}
        )
        do_simplified_dict = (
            {'do_simplified': kwargs.pop('do_simplified')}
            if 'do_simplified' in kwargs else {}
        )

        # --- Initialise the base class ---------------------------------------
        super().__init__(
            condensation_time=condensation_time,
            RH_ref=RH_ref,
            **kwargs,
        )

        # --- Override RH_ref to support array broadcasting -------------------
        if isinstance(RH_ref, np.ndarray):
            self.RH_ref = RH_ref + 0.0 * self.state['q']
        else:
            self.RH_ref = RH_ref * np.ones_like(self.state['q'])

        # --- Store extension attributes --------------------------------------
        self.sink_func = sink_func
        self.do_sink_func = sink_func is not None
        self.pmin = pmin
        self.small_dict = small_dict
        self.do_era5_dict = do_era5_dict
        self.do_simplified_dict = do_simplified_dict

        # --- Register additional diagnostic ----------------------------------
        # The base class already registers ``latent_heating`` and
        # ``precipitation``.  We add a per-level field.
        self.add_diagnostic('precipitation_2D', 0.0 * self.Tatm)

    # -----------------------------------------------------------------
    # _compute -- full override with extended physics
    # -----------------------------------------------------------------

    def _compute(self):
        timestep = getattr(self, 'timestep_in_seconds', self.timestep)

        # --- Saturation specific humidity (extended formula) -----------------
        qsaturation = qsat_extended(
            self.Tatm, self.lev,
            **self.small_dict, **self.do_era5_dict, **self.do_simplified_dict,
        )

        # --- Multi-latitude flag ---------------------------------------------
        if len(self.Ts) > 1:
            lev_mat, _ = np.meshgrid(self.lev, self.lat)
            has_lat = True
        else:
            lev_mat = self.lev
            has_lat = False

        # --- Moisture tendency -----------------------------------------------
        if self.do_sink_func:
            # Custom sink function provides the tendency directly
            qtendency = -self.sink_func(self.state)
            tendency_max = -self.q / timestep
            qtendency = np.where(
                qtendency < tendency_max, tendency_max, qtendency,
            )
        else:
            # Exponential relaxation toward RH_ref * qsat
            qtendency = (
                -(self.q - self.RH_ref * qsaturation)
                * (1.0 - np.exp(-timestep / self.condensation_time))
                / timestep
            )
            # Only allow condensation (negative tendency), not evaporation
            qtendency = np.where(qtendency > 0.0, 0.0, qtendency)

        # --- Suppress condensation above pmin --------------------------------
        qtendency = np.where(lev_mat <= self.pmin, 0.0, qtendency)

        # --- Temperature tendency from latent heating ------------------------
        tendencies = {}
        tendencies['q'] = qtendency * 1.0
        tendencies['Tatm'] = -const.Lhvap / const.cp * tendencies['q']

        # --- Update diagnostics ----------------------------------------------
        self.latent_heating[:] = (
            tendencies['Tatm'] * self.Tatm.domain.heat_capacity
        )
        self.precipitation_2D[:] = self.latent_heating / const.Lhvap

        if has_lat:
            self.precipitation[:, 0] = (
                np.sum(self.latent_heating, axis=-1) / const.Lhvap
            )
        else:
            self.precipitation[:] = (
                np.sum(self.latent_heating) / const.Lhvap
            )

        return tendencies
