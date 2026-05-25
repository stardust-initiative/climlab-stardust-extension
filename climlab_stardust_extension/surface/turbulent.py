"""Extended surface turbulent heat and moisture flux processes.

Extends climlab's SensibleHeatFlux and LatentHeatFlux with:
- Multi-level heat distribution via a turbulent layer profile (p_turb_layer)
- Analytic implicit coupling option (do_analytic)
- External flux prescription (do_external)
- ERA5-compatible qsat for latent heat flux
"""

import numpy as np
from climlab import constants as const
from climlab.surface.turbulent import SensibleHeatFlux, LatentHeatFlux
from climlab.domain.field import Field

from climlab_stardust_extension.utils.thermo import (
    qsat_extended, dqsat_dT,
)


class SensibleHeatFlux_extended(SensibleHeatFlux):
    """Extended sensible heat flux with multi-level distribution.

    Extends :class:`climlab.surface.SensibleHeatFlux` with:
    - ``p_turb_layer``: distribute flux across a turbulent boundary layer
      (exponential weight profile) instead of only the lowest model level.
    - ``do_analytic``: analytic implicit coupling for numerical stability.
    - ``do_external``: prescribe an external SHF value.

    Parameters
    ----------
    Cd : float
        Drag coefficient (default: 3E-3)
    p_turb_layer : float
        Pressure scale for turbulent layer distribution in hPa (default: 0).
        If 0, all flux goes to the lowest level (same as original).
    do_analytic : bool
        Use analytic implicit coupling (default: False)
    do_external : bool
        Use externally prescribed SHF (default: False)
    SHF_external : ndarray
        External SHF value (required if do_external=True)
    """
    def __init__(self, Cd=3E-3, p_turb_layer=0.0, do_analytic=False,
                 **kwargs):
        # Set subclass-specific attributes explicitly BEFORE calling
        # super().__init__.  The target base class is conda-forge
        # upstream climlab's ``SensibleHeatFlux``, which does not
        # accept ``p_turb_layer`` / ``do_analytic`` / ``do_external``
        # as kwargs and will not populate them on ``self``.  Popping
        # these kwargs here also prevents them from reaching
        # ``Process.__init__`` (where they would otherwise be stored
        # silently in ``self.param`` without being set as attributes).
        self.p_turb_layer = p_turb_layer
        self.do_analytic = do_analytic
        self.do_external = kwargs.pop('do_external', False)
        if self.do_external:
            assert 'SHF_external' in kwargs, (
                'if do_external is set, SHF_external must be provided'
            )
            self.SHF_external = kwargs.pop('SHF_external')
        super(SensibleHeatFlux_extended, self).__init__(Cd=Cd, **kwargs)
        # Set up vertical weight profile
        if self.p_turb_layer > 0.0:
            w_bounds = (
                (1.0 - np.exp(-(self.ps - self.lev_bounds) / self.p_turb_layer))
                / (1.0 - np.exp(-self.ps / self.p_turb_layer))
            )
            self.weight = -np.diff(w_bounds)
        else:
            self.weight = np.zeros_like(self.lev)
            self.weight[-1] = 1.0
        while len(self.weight.shape) < len(self.Tatm.shape):
            self.weight = self.weight[np.newaxis, ...]
        assert np.all(self.weight[..., -1] > 0.0), (
            f'self.weight[...,-1] must be finite: weight={self.weight}'
        )

    def _compute_heating_rates(self):
        self._compute_flux()
        self.heating_rate['Ts'] = -np.sum(self._flux, axis=-1)[..., np.newaxis]
        self.heating_rate['Tatm'] = self._flux

    def _compute_flux(self):
        Ta = Field(self.Tatm[..., -1, np.newaxis], domain=self.Ts.domain)
        Ts = self.Ts
        rho = self._air_density(Ta)
        alpha = self.resistance * const.cp * rho * self.Cd * self.U
        DeltaT = Ts - Ta
        SHF = alpha * DeltaT
        if self.do_external:
            SHF = self.SHF_external
        if self.do_analytic:
            cp_s = self.Ts.domain.heat_capacity
            cp0_tilde = (
                self.Tatm.domain.heat_capacity[-1] / self.weight[..., -1]
            )
            cp_avg = 1 / (1 / cp_s + 1 / cp0_tilde)
            gamma = alpha * getattr(self, 'timestep_in_seconds', self.timestep) / cp_avg
            self._flux = self.weight * SHF * (1.0 - np.exp(-gamma)) / gamma
        else:
            self._flux = self.weight * SHF
        self.SHF[:] = SHF


class LatentHeatFlux_extended(LatentHeatFlux):
    """Extended latent heat flux with multi-level distribution.

    Extends :class:`climlab.surface.LatentHeatFlux` with:
    - ``p_turb_layer``: distribute flux across a turbulent boundary layer.
    - ``do_analytic``: analytic implicit coupling for numerical stability.
    - ``do_external``: prescribe an external LHF value.
    - ERA5-compatible qsat formula support.

    Parameters
    ----------
    Cd : float
        Drag coefficient (default: 3E-3)
    p_turb_layer : float
        Pressure scale for turbulent layer distribution in hPa (default: 0).
    do_analytic : bool
        Use analytic implicit coupling (default: False)
    do_external : bool
        Use externally prescribed LHF (default: False)
    LHF_external : ndarray
        External LHF value (required if do_external=True)
    qsat_param_dict : dict
        Parameters for qsat_extended (do_era5, small, do_simplified)
    """
    def __init__(self, Cd=3E-3, p_turb_layer=0.0, do_analytic=False,
                 **kwargs):
        # See ``SensibleHeatFlux_extended.__init__`` for the rationale:
        # the target base class is conda-forge upstream climlab's
        # ``LatentHeatFlux``, which does not know about any of the
        # extended kwargs.  Set them on ``self`` explicitly before
        # super() so ``Process.__init__`` doesn't silently swallow
        # them into ``self.param``.
        self.p_turb_layer = p_turb_layer
        self.do_analytic = do_analytic
        self.do_external = kwargs.pop('do_external', False)
        if self.do_external:
            assert 'LHF_external' in kwargs, (
                'if do_external is set, LHF_external must be provided'
            )
            self.LHF_external = kwargs.pop('LHF_external')
        self.qsat_param_dict = kwargs.pop('qsat_param_dict', {
            'do_era5': False, 'small': 0.0, 'do_simplified': False,
        })
        super(LatentHeatFlux_extended, self).__init__(Cd=Cd, **kwargs)
        # Set up vertical weight profile
        if self.p_turb_layer > 0.0:
            w_bounds = (
                (1.0 - np.exp(-(self.ps - self.lev_bounds) / self.p_turb_layer))
                / (1.0 - np.exp(-self.ps / self.p_turb_layer))
            )
            self.weight = -np.diff(w_bounds)
        else:
            self.weight = np.zeros_like(self.lev)
            self.weight[-1] = 1.0
        while len(self.weight.shape) < len(self.Tatm.shape):
            self.weight = self.weight[np.newaxis, ...]
        assert np.all(self.weight[..., -1] > 0.0), (
            f'self.weight[...,-1] must be finite: weight={self.weight}'
        )

    def _compute_heating_rates(self):
        self._compute_flux()
        self.heating_rate['Ts'] = -np.sum(self._flux, axis=-1)[..., np.newaxis]
        self.heating_rate['Tatm'] = self._flux

    def _compute_flux(self):
        q = Field(self.q[..., -1, np.newaxis], domain=self.Ts.domain)
        Ta = Field(self.Tatm[..., -1, np.newaxis], domain=self.Ts.domain)
        qs = qsat_extended(self.Ts, self.ps, **self.qsat_param_dict)
        Deltaq = Field(qs - q, domain=self.Ts.domain)
        rho = self._air_density(Ta)
        alpha = self.resistance * const.Lhvap * rho * self.Cd * self.U
        LHF = alpha * Deltaq
        if self.do_external:
            LHF = self.LHF_external
        if self.do_analytic:
            dqs_dT_val = dqsat_dT(
                self.Ts, self.ps, **self.qsat_param_dict
            )
            lp_s = self.Ts.domain.heat_capacity / dqs_dT_val
            lp0_tilde = (
                const.Lhvap / const.cp
                * self.Tatm.domain.heat_capacity[-1] / self.weight[..., -1]
            )
            lp_avg = 1 / (1 / lp_s + 1 / lp0_tilde)
            gamma = alpha * getattr(self, 'timestep_in_seconds', self.timestep) / lp_avg
            self._flux = self.weight * LHF * (1.0 - np.exp(-gamma)) / gamma
        else:
            self._flux = self.weight * LHF
        self.LHF[:] = LHF
        self.evaporation[:] = self.LHF / const.Lhvap

    def _compute(self):
        tendencies = self._temperature_tendencies()
        if 'q' in self.state:
            tendencies['Tatm'] *= 0.
            tendencies['q'] = const.cp / const.Lhvap * tendencies['Tatm']
            # Recompute: the Tatm tendency from heating_rate is already
            # distributed across levels by _compute_heating_rates
            # We need the q tendency from the moisture flux
            tendencies['q'] = (
                self._flux / const.Lhvap
                / self.Tatm.domain.heat_capacity * const.cp
            )
        return tendencies
