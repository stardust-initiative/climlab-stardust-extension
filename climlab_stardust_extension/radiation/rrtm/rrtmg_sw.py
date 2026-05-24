"""Extended RRTMG shortwave radiation.

Extends climlab's ``RRTMG_SW`` with:

* Monte-Carlo ensemble averaging via ``climlab_rrtmg_sw_ensemble``
  (provided by the ``climlab-rrtmg-stardust`` Fortran package).
* Column-by-column and seed-permutation options for McICA.
* Spectrally-resolved ASR diagnostics.
* Aerosol layer parameters (``add_aero_layer``, ``r_mu``, etc.) for
  advanced scattering treatment.
* ``kmodts`` control parameter.
"""

import numpy as np
import warnings
from climlab import constants as const
from climlab.radiation.rrtm.rrtmg_sw import RRTMG_SW
from climlab.radiation.rrtm.utils import (
    _prepare_general_arguments,
    _climlab_to_rrtm,
    _climlab_to_rrtm_sfc,
    _rrtm_to_climlab,
)
from climlab.domain import Field, Axis, domain

# Values from the Fortran extension
nbndsw = 1
naerec = 1
ngptsw = 1
try:
    from climlab_rrtmg import rrtmg_sw as _rrtmg_sw
    nbndsw = int(_rrtmg_sw.parrrsw.nbndsw)
    naerec = int(_rrtmg_sw.parrrsw.naerec)
    ngptsw = int(_rrtmg_sw.parrrsw.ngptsw)
except Exception:
    warnings.warn(
        'Cannot import climlab_rrtmg (stardust). '
        'RRTMG_SW_extended will not be functional.'
    )

# Shortwave spectral band limits (wavenumbers in cm^-1)
band_numbers = np.array([29, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])
wavenum_bounds = np.array([
    820., 2600., 3250., 4000., 4650., 5150., 6150., 7700., 8050.,
    12850., 16000., 22650., 29000., 38000., 50000.,
])
wavenum_delta = np.diff(wavenum_bounds)
wavenum_ax = Axis(axis_type='abstract', bounds=wavenum_bounds)


class RRTMG_SW_extended(RRTMG_SW):
    """Extended RRTMG shortwave with ensemble averaging and aerosol layer.

    Adds over ``RRTMG_SW``:

    * ``do_col_by_col`` : bool — process columns independently.
    * ``do_seed_permutation`` : bool — advance McICA seed each timestep.
    * ``n_rrtmg_repeat`` : int — number of ensemble members.
    * ``return_spectral_asr`` : bool — spectrally-decomposed ASR output.
    * ``kmodts`` : int — control flag for adding aerosol layer.
    * ``add_aero_layer`` : int — flag to enable aerosol layer treatment.
    * ``r_mu, t_mu, r_bar, t_bar`` : aerosol layer optical parameters.
    * Calls ``climlab_rrtmg_sw_ensemble`` instead of separate McICA + RRTMG.

    Parameters
    ----------
    do_col_by_col : bool
        Column-by-column mode (default False).
    do_seed_permutation : bool
        Advance McICA seed each step (default False).
    n_rrtmg_repeat : int
        Number of ensemble members (default 1).
    return_spectral_asr : bool
        Return spectrally-resolved ASR (default False).
    kmodts : int
        Control flag (default 2).
    add_aero_layer : int
        Enable aerosol layer (0 or 1, default 0).
    r_mu, t_mu, r_bar, t_bar : float or ndarray
        Aerosol layer optical parameters (default 0/0/0/0).
    **kwargs
        Passed to ``RRTMG_SW.__init__``.
    """

    def __init__(self, do_col_by_col=False, do_seed_permutation=False,
                 n_rrtmg_repeat=1, return_spectral_asr=False,
                 kmodts=2, add_aero_layer=0,
                 r_mu=0.0, t_mu=0.0, r_bar=0.0, t_bar=0.0,
                 **kwargs):
        super(RRTMG_SW_extended, self).__init__(**kwargs)
        self.do_col_by_col = do_col_by_col
        self.do_seed_permutation = do_seed_permutation
        self.n_rrtmg_repeat = n_rrtmg_repeat
        self.add_input('return_spectral_asr', return_spectral_asr)
        self.add_input('kmodts', kmodts)
        self.add_input('add_aero_layer', add_aero_layer)
        if add_aero_layer == 1:
            self.add_input('r_mu', r_mu)
            self.add_input('t_mu', t_mu)
            self.add_input('r_bar', r_bar)
            self.add_input('t_bar', t_bar)

        # Spectral ASR diagnostics
        if self.return_spectral_asr:
            self._ispec = 1
            spectral_axes = {**self.ASR.domain.axes, 'wavenumber': wavenum_ax}
            spectral_domain = domain._Domain(axes=spectral_axes)
            shape = list(self.ASR.shape)
            shape.append(wavenum_ax.num_points)
            spectral_domain.shape = tuple(shape)
            spectral_domain.axis_index = {
                **self.ASR.domain.axis_index,
                'wavenumber': len(shape) - 1,
            }
        else:
            self._ispec = (0,)

    def _spectral_field(self, field):
        """Return spectral field without domain wrapping."""
        return Field(
            field * np.repeat(
                np.ones_like(self.Tatm[np.newaxis, ...]), nbndsw, axis=0
            ),
            domain=domain._Domain(
                axes={**self.Tatm.domain.axes, 'wavenumber': wavenum_ax}
            ),
        )

    def _prepare_sw_arguments(self):
        """Prepare arguments including extended aerosol-layer fields."""
        icld = self.icld
        ispec = self._ispec
        irng = self.irng
        permuteseed = self.permuteseed
        inflgsw = self.inflgsw
        iceflgsw = self.iceflgsw
        liqflgsw = self.liqflgsw
        dyofyr = self.dyofyr
        isolvar = self.isolvar
        solcycfrac = self.solcycfrac
        iaer = self.iaer
        scon = self.S0
        indsolvar = self.indsolvar
        bndsolvar = self.bndsolvar
        kmodts = self.kmodts

        (ncol, nlay, play, plev, tlay, tlev, tsfc,
         h2ovmr, o3vmr, co2vmr, ch4vmr, n2ovmr, o2vmr, cfc11vmr,
         cfc12vmr, cfc12vmr, cfc22vmr, ccl4vmr,
         cldfrac, ciwp, clwp, relq, reic) = _prepare_general_arguments(self)

        aldif = _climlab_to_rrtm_sfc(self.aldif, self.Ts)
        aldir = _climlab_to_rrtm_sfc(self.aldir, self.Ts)
        asdif = _climlab_to_rrtm_sfc(self.asdif, self.Ts)
        asdir = _climlab_to_rrtm_sfc(self.asdir, self.Ts)
        coszen = _climlab_to_rrtm_sfc(self.coszen, self.Ts)
        adjes = _climlab_to_rrtm_sfc(self.irradiance_factor, self.Ts)

        tauc = _climlab_to_rrtm(
            self.tauc * np.ones_like(self.Tatm)
        ) * np.ones([nbndsw, ncol, nlay])
        ssac = _climlab_to_rrtm(
            self.ssac * np.ones_like(self.Tatm)
        ) * np.ones([nbndsw, ncol, nlay])
        asmc = _climlab_to_rrtm(
            self.asmc * np.ones_like(self.Tatm)
        ) * np.ones([nbndsw, ncol, nlay])
        fsfc = _climlab_to_rrtm(
            self.fsfc * np.ones_like(self.Tatm)
        ) * np.ones([nbndsw, ncol, nlay])

        tauaer = _climlab_to_rrtm(self.tauaer, spectral_axis=True)
        ssaaer = _climlab_to_rrtm(self.ssaaer, spectral_axis=True)
        asmaer = _climlab_to_rrtm(self.asmaer, spectral_axis=True)
        ecaer = _climlab_to_rrtm(self.ecaer, spectral_axis=True)

        add_aero_layer = self.add_aero_layer
        if self.add_aero_layer == 1:
            r_mu_arr = _climlab_to_rrtm(
                self.r_mu, spectral_axis=True, do_lev_flip=False
            )
            t_mu_arr = _climlab_to_rrtm(
                self.t_mu, spectral_axis=True, do_lev_flip=False
            )
            r_bar_arr = _climlab_to_rrtm(
                self.r_bar, spectral_axis=True, do_lev_flip=False
            )
            t_bar_arr = _climlab_to_rrtm(
                self.t_bar, spectral_axis=True, do_lev_flip=False
            )
        else:
            s = tauaer.shape
            r_mu_arr = np.zeros(s)
            t_mu_arr = np.ones(s)
            r_bar_arr = np.zeros(s)
            t_bar_arr = np.ones(s)

        args = [
            ncol, nlay, icld, ispec, iaer, permuteseed, irng,
            play, plev, tlay, tlev, tsfc,
            h2ovmr, o3vmr, co2vmr, ch4vmr, n2ovmr, o2vmr,
            aldif, aldir, asdif, asdir, coszen, adjes, dyofyr, scon, isolvar,
            indsolvar, bndsolvar, solcycfrac,
            inflgsw, iceflgsw, liqflgsw,
            cldfrac, ciwp, clwp, reic, relq, tauc, ssac, asmc, fsfc,
            tauaer, ssaaer, asmaer, ecaer, kmodts,
            add_aero_layer, r_mu_arr, t_mu_arr, r_bar_arr, t_bar_arr,
        ]
        return args

    def _compute_heating_rates(self):
        """Call ``climlab_rrtmg_sw_ensemble`` and compute heating rates."""
        (ncol, nlay, icld, ispec, iaer, permuteseed, irng,
         play, plev, tlay, tlev, tsfc,
         h2ovmr, o3vmr, co2vmr, ch4vmr, n2ovmr, o2vmr,
         aldif, aldir, asdif, asdir, coszen, adjes, dyofyr, scon, isolvar,
         indsolvar, bndsolvar, solcycfrac,
         inflgsw, iceflgsw, liqflgsw,
         cldfrac, ciwp, clwp, reic, relq, tauc, ssac, asmc, fsfc,
         tauaer, ssaaer, asmaer, ecaer, kmodts,
         add_aero_layer, r_mu, t_mu, r_bar, t_bar) = \
            self._prepare_sw_arguments()

        n_rrtmg_repeat = (
            self.n_rrtmg_repeat
            if (icld > 0 and self.do_seed_permutation)
            else 1
        )
        col_by_col = 1 if self.do_col_by_col else 0
        do_seed_perm = 1 if self.do_seed_permutation else 0
        if self.do_seed_permutation:
            seed = (permuteseed
                    + n_rrtmg_repeat * self.time['steps'] * ngptsw)
        else:
            seed = permuteseed

        (swuflx, swdflx, swhr, swuflxc, swdflxc, swhrc,
         swuflxspec, swdflxspec, swuflxcspec, swdflxcspec) = \
            _rrtmg_sw.climlab_rrtmg_sw_ensemble(
                ncol, nlay,
                seed, irng, n_rrtmg_repeat, col_by_col, do_seed_perm,
                icld, ispec, iaer, play, plev, tlay, tlev, tsfc,
                cldfrac, ciwp, clwp, reic, relq, tauc, ssac, asmc, fsfc,
                h2ovmr, o3vmr, co2vmr, ch4vmr, n2ovmr, o2vmr,
                asdir, asdif, aldir, aldif,
                kmodts, coszen, adjes, dyofyr, scon, isolvar,
                inflgsw, iceflgsw, liqflgsw,
                tauaer, ssaaer, asmaer, ecaer,
                bndsolvar, indsolvar, solcycfrac,
                add_aero_layer, r_mu, t_mu, r_bar, t_bar,
            )

        # Flux diagnostics
        self.SW_flux_up = _rrtm_to_climlab(swuflx) + 0. * self.SW_flux_up
        self.SW_flux_down = _rrtm_to_climlab(swdflx) + 0. * self.SW_flux_down
        self.SW_flux_up_clr = (
            _rrtm_to_climlab(swuflxc) + 0. * self.SW_flux_up_clr
        )
        self.SW_flux_down_clr = (
            _rrtm_to_climlab(swdflxc) + 0. * self.SW_flux_down_clr
        )

        # Spectral ASR
        if self.return_spectral_asr:
            self.SW_flux_up_spectral = _rrtm_to_climlab(
                swuflxspec.transpose(0, 2, 1)
            ).transpose(0, 2, 1)
            self.SW_flux_down_spectral = _rrtm_to_climlab(
                swdflxspec.transpose(0, 2, 1)
            ).transpose(0, 2, 1)
            self.SW_flux_up_clr_spectral = _rrtm_to_climlab(
                swuflxcspec.transpose(0, 2, 1)
            ).transpose(0, 2, 1)
            self.SW_flux_down_clr_spectral = _rrtm_to_climlab(
                swdflxcspec.transpose(0, 2, 1)
            ).transpose(0, 2, 1)
            self.ASR_spectral = (
                self.SW_flux_down_spectral[:, 0, :]
                - self.SW_flux_up_spectral[:, 0, :]
            )
            s = self.ASR_spectral.shape
            self.ASR_spectral = self.ASR_spectral.reshape(s[0], 1, s[1])

        # Derived flux diagnostics (ASR, etc.)
        self._compute_SW_flux_diagnostics()

        # Heating rates
        SWheating_Wm2 = (
            np.array(-np.diff(self.SW_flux_net, axis=-1)) + 0. * self.Tatm
        )
        SWheating_clr_Wm2 = (
            np.array(-np.diff(self.SW_flux_net_clr, axis=-1)) + 0. * self.Tatm
        )
        self.heating_rate['Ts'] = (
            np.array(self.SW_flux_net[..., -1, np.newaxis]) + 0. * self.Ts
        )
        self.heating_rate['Tatm'] = SWheating_Wm2

        Catm = self.Tatm.domain.heat_capacity
        self.TdotSW = SWheating_Wm2 / Catm * const.seconds_per_day
        self.TdotSW_clr = SWheating_clr_Wm2 / Catm * const.seconds_per_day
