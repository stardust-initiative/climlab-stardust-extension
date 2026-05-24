"""Extended RRTMG longwave radiation.

Extends climlab's ``RRTMG_LW`` with:

* Monte-Carlo ensemble averaging via ``climlab_rrtmg_lw_ensemble``
  (provided by the ``climlab-rrtmg-stardust`` Fortran package).
* Column-by-column and seed-permutation options for McICA.
* ``iaer`` flag passed through to the Fortran driver.
* Spectral flux diagnostics (up/down/clr per band).
"""

import numpy as np
import warnings
from climlab import constants as const
from climlab.radiation.rrtm.rrtmg_lw import RRTMG_LW
from climlab.radiation.rrtm.utils import (
    _prepare_general_arguments,
    _climlab_to_rrtm,
    _rrtm_to_climlab,
)
from climlab.domain import Axis, domain

# Values from the Fortran extension
nbndlw = 1
ngptlw = 1
try:
    from climlab_rrtmg import rrtmg_lw as _rrtmg_lw
    nbndlw = int(_rrtmg_lw.parrrtm.nbndlw)
    ngptlw = int(_rrtmg_lw.parrrtm.ngptlw)
except Exception:
    warnings.warn(
        'Cannot import climlab_rrtmg (stardust). '
        'RRTMG_LW_extended will not be functional.'
    )

# Longwave spectral band limits (wavenumbers in cm^-1)
wavenum_bounds = np.array([
    10., 350., 500., 630., 700., 820.,
    980., 1080., 1180., 1390., 1480., 1800.,
    2080., 2250., 2380., 2600., 3250.
])
wavenum_delta = np.diff(wavenum_bounds)
wavenum_ax = Axis(axis_type='abstract', bounds=wavenum_bounds)


class RRTMG_LW_extended(RRTMG_LW):
    """Extended RRTMG longwave with ensemble averaging and aerosol support.

    Adds over ``RRTMG_LW``:

    * ``do_col_by_col`` : bool — process columns independently.
    * ``do_seed_permutation`` : bool — advance McICA seed each timestep.
    * ``n_rrtmg_repeat`` : int — number of ensemble members.
    * ``iaer`` : int — aerosol flag (0 = none, 10 = input optical props).
    * Calls ``climlab_rrtmg_lw_ensemble`` instead of separate McICA + RRTMG.
    * Outputs spectral flux diagnostics when ``return_spectral_olr=True``.

    Parameters
    ----------
    iaer : int
        Aerosol option flag (default 0).
    do_col_by_col : bool
        Column-by-column mode (default False).
    do_seed_permutation : bool
        Advance McICA seed each step (default False).
    n_rrtmg_repeat : int
        Number of ensemble members to average (default 1).
    **kwargs
        Passed to ``RRTMG_LW.__init__``.
    """

    def __init__(self, iaer=0, do_col_by_col=False,
                 do_seed_permutation=False, n_rrtmg_repeat=1, **kwargs):
        super(RRTMG_LW_extended, self).__init__(**kwargs)
        self.iaer = iaer
        self.do_col_by_col = do_col_by_col
        self.do_seed_permutation = do_seed_permutation
        self.n_rrtmg_repeat = n_rrtmg_repeat

    def _spectral_field(self, field):
        """Return tauaer-like spectral field without wrapping in a Field domain.

        This avoids a bug in the original implementation where separate
        process updates would corrupt the domain-wrapped spectral field.
        """
        if isinstance(field, np.ndarray):
            s = tuple([nbndlw] + list(self.Tatm.shape))
            if np.all(field.shape == s):
                return field
        return field * np.repeat(
            np.ones_like(self.Tatm[np.newaxis, ...]), nbndlw, axis=0
        )

    def _compute_heating_rates(self):
        """Call ``climlab_rrtmg_lw_ensemble`` and compute heating rates."""
        (ncol, nlay, icld, ispec, permuteseed, irng, idrv, cp,
         play, plev, tlay, tlev, tsfc,
         h2ovmr, o3vmr, co2vmr, ch4vmr, n2ovmr, o2vmr,
         cfc11vmr, cfc12vmr, cfc22vmr, ccl4vmr, emis,
         inflglw, iceflglw, liqflglw,
         cldfrac, ciwp, clwp, reic, relq, tauc,
         tauaer) = self._prepare_lw_arguments()

        n_rrtmg_repeat = (
            self.n_rrtmg_repeat
            if (icld > 0 and self.do_seed_permutation)
            else 1
        )
        col_by_col = 1 if self.do_col_by_col else 0
        do_seed_perm = 1 if self.do_seed_permutation else 0
        if self.do_seed_permutation:
            seed = (permuteseed
                    + n_rrtmg_repeat * self.time['steps'] * ngptlw)
        else:
            seed = permuteseed

        (olr_sr, uflx, dflx, hr, uflxc, dflxc, hrc,
         duflx_dt, duflxc_dt,
         uflxspec, dflxspec, uflxcspec, dflxcspec) = \
            _rrtmg_lw.climlab_rrtmg_lw_ensemble(
                ncol, nlay,
                seed, irng, n_rrtmg_repeat, col_by_col, do_seed_perm,
                icld, ispec, idrv,
                play, plev, tlay, tlev, tsfc,
                cldfrac, ciwp, clwp, reic, relq, tauc,
                h2ovmr, o3vmr, co2vmr, ch4vmr, n2ovmr, o2vmr,
                cfc11vmr, cfc12vmr, cfc22vmr, ccl4vmr, emis,
                inflglw, iceflglw, liqflglw, tauaer,
            )

        # Assign flux diagnostics
        self.LW_flux_up = _rrtm_to_climlab(uflx) + 0. * self.LW_flux_up
        self.LW_flux_down = _rrtm_to_climlab(dflx) + 0. * self.LW_flux_down
        self.LW_flux_up_clr = (
            _rrtm_to_climlab(uflxc) + 0. * self.LW_flux_up_clr
        )
        self.LW_flux_down_clr = (
            _rrtm_to_climlab(dflxc) + 0. * self.LW_flux_down_clr
        )
        self._compute_LW_flux_diagnostics()

        # Spectral diagnostics
        if self.return_spectral_olr:
            self.OLR_spectral = (
                np.squeeze(olr_sr)[..., np.newaxis, :]
                + 0. * self.OLR_spectral
            )
            self.LW_flux_up_spectral = _rrtm_to_climlab(
                uflxspec.transpose(0, 2, 1)
            ).transpose(0, 2, 1)
            self.LW_flux_down_spectral = _rrtm_to_climlab(
                dflxspec.transpose(0, 2, 1)
            ).transpose(0, 2, 1)
            self.LW_flux_up_clr_spectral = _rrtm_to_climlab(
                uflxcspec.transpose(0, 2, 1)
            ).transpose(0, 2, 1)
            self.LW_flux_down_clr_spectral = _rrtm_to_climlab(
                dflxcspec.transpose(0, 2, 1)
            ).transpose(0, 2, 1)

        # Heating rates
        LWheating_Wm2 = (
            np.array(np.diff(self.LW_flux_net, axis=-1)) + 0. * self.Tatm
        )
        LWheating_clr_Wm2 = (
            np.array(np.diff(self.LW_flux_net_clr, axis=-1)) + 0. * self.Tatm
        )
        self.heating_rate['Ts'] = (
            np.array(-self.LW_flux_net[..., -1, np.newaxis]) + 0. * self.Ts
        )
        self.heating_rate['Tatm'] = LWheating_Wm2

        Catm = self.Tatm.domain.heat_capacity
        self.TdotLW = LWheating_Wm2 / Catm * const.seconds_per_day
        self.TdotLW_clr = LWheating_clr_Wm2 / Catm * const.seconds_per_day

        if self.return_spectral_olr:
            LWheating_Wm2_spectral = np.array(np.diff(
                self.LW_flux_up_spectral - self.LW_flux_down_spectral, axis=1
            ))
            LWheating_clr_Wm2_spectral = np.array(np.diff(
                self.LW_flux_up_clr_spectral - self.LW_flux_down_clr_spectral,
                axis=1,
            ))
            self.TdotLW_spectral = (
                LWheating_Wm2_spectral
                / Catm[np.newaxis, :, np.newaxis]
                * const.seconds_per_day
            )
            self.TdotLW_clr_spectral = (
                LWheating_clr_Wm2_spectral
                / Catm[np.newaxis, :, np.newaxis]
                * const.seconds_per_day
            )
