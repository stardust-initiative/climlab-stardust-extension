"""Extended meridional moist diffusion and moist static energy transport.

Provides new process classes for:
- MeridionalMoistDiffusionAtm: moist diffusion on atmospheric levels
- SpecificEnthalpy: diagnostic moist static energy
- FixedGeoPotentialFlux: geopotential flux with fixed geopotential
- GeoPotentialFlux: geopotential flux computed from temperature
- MoistMeridionalAdvectionDiffusion: combined MSE + q transport
- moist_amplification_factor_extended: extended moist amplification factor
"""

import numpy as np
from climlab.utils import constants as const
from climlab.process.time_dependent_process import TimeDependentProcess
from climlab.process.diagnostic import DiagnosticProcess
from climlab.process import couple
from climlab.dynamics.meridional_advection_diffusion import (
    MeridionalAdvectionDiffusion,
)
from climlab.dynamics.meridional_heat_diffusion import (
    MeridionalHeatDiffusion,
    MeridionalDiffusion,
)
from climlab_stardust_extension.utils.thermo import dqsat_dT
from climlab_stardust_extension.dynamics.two_dimensional_advection_diffusion import (
    TwoDimensionalAdvectionDiffusion,
)


def moist_amplification_factor_extended(Tkelvin, relative_humidity=0.8,
                                        p=1000.0, do_simplified=False,
                                        do_era5=False, small=0.0):
    """Compute the moist amplification factor for heat diffusion.

    Extends climlab's moist_amplification_factor with ERA5-compatible
    qsat derivative and pressure/formula options.

    Parameters
    ----------
    Tkelvin : ndarray
        Temperature in Kelvin
    relative_humidity : float
        Relative humidity (default: 0.8)
    p : float or ndarray
        Pressure in hPa (default: 1000)
    do_simplified : bool
        Use simplified qsat formula
    do_era5 : bool
        Use ERA5-compatible formula
    small : float
        Regularization parameter

    Returns
    -------
    m : ndarray
        Moist amplification factor (dimensionless)
    """
    dqsdTs = dqsat_dT(
        Tkelvin, p,
        do_simplified=do_simplified, do_era5=do_era5, small=small
    )
    m = const.Lhvap / const.cp * relative_humidity * dqsdTs
    return m


class MeridionalMoistDiffusionAtm(MeridionalHeatDiffusion):
    """Meridional moist diffusion on atmospheric levels.

    Extends MeridionalHeatDiffusion with a moist amplification factor
    computed at each level.

    Parameters
    ----------
    D : float
        Diffusivity parameter (default: 0.24)
    relative_humidity : float
        Reference relative humidity for moist factor (default: 0.8)
    do_moist_fac : bool
        Whether to include moist amplification (default: True)
    """
    def __init__(self, D=0.24, relative_humidity=0.8, do_moist_fac=True,
                 **kwargs):
        self.relative_humidity = relative_humidity
        self.do_moist_fac = do_moist_fac
        super(MeridionalMoistDiffusionAtm, self).__init__(D=D, **kwargs)
        self._update_diffusivity()

    def _update_diffusivity(self):
        lat_bounds = self.lat_bounds
        n_lev = self.Tatm.domain.lev.num_points
        n_lat = self.Tatm.domain.lat.num_points
        lat = self.Tatm.domain.lat.points
        lev = self.Tatm.domain.lev.points[np.newaxis, :]
        is_kelvin = np.mean(self.Tatm) > 140.0
        dT = 0.0 if is_kelvin else const.tempCtoK
        Tkelvin = self.Tatm + dT
        if self.do_moist_fac:
            m_amp_fac = moist_amplification_factor_extended(
                Tkelvin, self.relative_humidity, p=lev, do_simplified=True
            )
        else:
            m_amp_fac = np.zeros_like(Tkelvin)
        heat_capacity = self.Tatm.domain.heat_capacity[np.newaxis, :]
        K_fac = const.a**2 * (1.0 + m_amp_fac) / heat_capacity
        K_fac_interp = np.zeros((n_lev, n_lat + 1))
        for k in range(n_lev):
            K_fac_interp[k, :] = np.interp(lat_bounds, lat, K_fac[:, k])
        self.K = self.D * K_fac_interp

    def _implicit_solver(self):
        self._update_diffusivity()
        return super(MeridionalMoistDiffusionAtm, self)._implicit_solver()


class SpecificEnthalpy(DiagnosticProcess):
    """Diagnostic process computing specific enthalpy (moist static energy).

    se = cp * Tatm + Lhvap * q
    """
    def __init__(self, **kwargs):
        super(SpecificEnthalpy, self).__init__(**kwargs)
        assert hasattr(self.state, 'q'), (
            'q must be a state parameter when using SpecificEnthalpy'
        )
        assert hasattr(self.state, 'Tatm'), (
            'Tatm must be a state parameter when using SpecificEnthalpy'
        )
        self._se = 0.0 * self.Tatm
        self._compute_se()

    def _compute_se(self):
        self._se[:] = const.cp * self.Tatm + const.Lhvap * self.q

    def get_T_part(self, se_part, q_part):
        """Extract temperature tendency from MSE and q tendencies."""
        return (se_part - const.Lhvap * q_part) / const.cp

    @property
    def se(self):
        self._compute_se()
        return self._se

    def _compute(self):
        return {}


class FixedGeoPotentialFlux(TimeDependentProcess):
    """Geopotential flux tendency using a prescribed geopotential field.

    Parameters
    ----------
    U : ndarray
        Meridional velocity at lat boundaries, shape (nlat+1, nlev)
    W : ndarray
        Vertical velocity at lev boundaries, shape (nlat, nlev+1)
    geopotential : ndarray
        Geopotential field, shape (nlat+1, nlev+1)
    """
    def __init__(self, U=0., W=0., **kwargs):
        super(FixedGeoPotentialFlux, self).__init__(**kwargs)
        assert hasattr(self.state, 'se'), (
            'se must be a state parameter when using FixedGeoPotentialFlux'
        )
        self.geopotential = kwargs.get('geopotential', 0.0)
        self._latbounds = (
            np.sin(np.pi / 180 * self.se.domain.lat.bounds) * const.a
        )
        self._dlatbounds = np.diff(self._latbounds)
        self._levbounds = self.state['se'].domain.axes['lev'].bounds * 1e2
        self._dlevbounds = np.diff(self._levbounds)
        self._levpoints = 0.5 * (self._levbounds[1:] + self._levbounds[:-1])
        self.W = W
        self.U = U
        nlat = self.state['se'].domain.lat.points.shape[0]
        nlev = self.state['se'].domain.lev.points.shape[0]
        assert nlat + 1 == U.shape[0]
        assert nlev == U.shape[1]
        assert nlat == W.shape[0]
        assert nlev + 1 == W.shape[1]
        assert nlat + 1 == self.geopotential.shape[0]
        assert nlev + 1 == self.geopotential.shape[1]

    @property
    def U(self):
        return self._U

    @U.setter
    def U(self, Uvalue):
        self._U = Uvalue

    @property
    def W(self):
        return self._W

    @W.setter
    def W(self, Wvalue):
        self._W = Wvalue

    @property
    def geopotential(self):
        return self._geopotential

    @geopotential.setter
    def geopotential(self, GPvalue):
        self._geopotential = GPvalue

    def _compute(self):
        gp_mid_lat = 0.5 * (
            self.geopotential[1:, :] + self.geopotential[:-1, :]
        )
        gp_mid_lev = 0.5 * (
            self.geopotential[:, 1:] + self.geopotential[:, :-1]
        )
        self.gp_source_z = (
            0.5 * (self.W[:, :-1] + self.W[:, 1:])
            * (gp_mid_lat[:, :-1] - gp_mid_lat[:, 1:])
            / self._dlevbounds[None, :]
        )
        self.gp_source_y = (
            0.5 * (self.U[:-1, :] + self.U[1:, :])
            * (gp_mid_lev[:-1, :] - gp_mid_lev[1:, :])
            / self._dlatbounds[:, None]
        )
        return {'se': self.gp_source_y + self.gp_source_z}


class GeoPotentialFlux(TimeDependentProcess):
    """Geopotential flux tendency computed from temperature.

    Parameters
    ----------
    U : ndarray
        Meridional velocity, shape (nlat+1, nlev)
    W : ndarray
        Vertical velocity, shape (nlat, nlev+1)
    T : ndarray
        Temperature field, shape (nlat, nlev)
    """
    def __init__(self, U, W, T, **kwargs):
        super(GeoPotentialFlux, self).__init__(**kwargs)
        assert hasattr(self.state, 'se'), (
            'se must be a state parameter when using GeoPotentialFlux'
        )
        self._ybounds = (
            np.sin(np.pi / 180 * self.se.domain.lat.bounds) * const.a
        )
        self._y = (
            np.sin(np.pi / 180 * self.se.domain.lat.points) * const.a
        )
        self._dy = np.diff(self._ybounds)
        self._pbounds = self.state['se'].domain.axes['lev'].bounds * 1e2
        self._dp = np.diff(self._pbounds)
        self._p = self.state['se'].domain.axes['lev'].points * 1e2

        nlat = self.state['se'].domain.lat.points.shape[0]
        nlev = self.state['se'].domain.lev.points.shape[0]
        assert nlat + 1 == U.shape[0]
        assert nlev == U.shape[1]
        assert nlat == W.shape[0]
        assert nlev + 1 == W.shape[1]
        assert nlat == T.shape[0]
        assert nlev == T.shape[1]

        self.W = W
        self.U = U
        self.T = T

    @property
    def U(self):
        return self._U

    @U.setter
    def U(self, Uvalue):
        self._U = Uvalue

    @property
    def W(self):
        return self._W

    @W.setter
    def W(self, Wvalue):
        self._W = Wvalue

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, Tvalue):
        self._T = Tvalue

    def _compute(self):
        dGPdp = const.Rd * self.T / self._p[np.newaxis, :]

        T_SP = (
            (self.T[1, ...] - self.T[0, ...])
            / (self._y[1] - self._y[0])
            * (self._ybounds[0] - self._y[..., 0])
            + self.T[0, ...]
        )[np.newaxis, :]
        T_mid = (
            np.diff(self.T, axis=0) / np.diff(self._y)[:, np.newaxis]
            * (self._ybounds[1:-1, np.newaxis] - self._y[:-1, np.newaxis])
            + self.T[:-1, :]
        )
        T_NP = (
            (self.T[-1, ...] - self.T[-2, ...])
            / (self._y[-1] - self._y[-2])
            * (self._ybounds[-1] - self._y[..., -1])
            + self.T[-1, ...]
        )[np.newaxis, :]
        Textend = np.concatenate((T_SP, T_mid, T_NP), axis=0)

        dTdy = np.diff(Textend, axis=0) / self._dy[:, np.newaxis]
        dGPdy = np.cumsum(
            (const.Rd * dTdy * (self._dp / self._p)[np.newaxis, :])
            [:, ::-1], axis=1
        )[:, ::-1]
        W_mid = (
            np.diff(self.W, axis=1) / self._dp[np.newaxis, :]
            * (self._p[np.newaxis, :] - self._pbounds[np.newaxis, :-1])
            + self.W[:, :-1]
        )
        U_mid = (
            np.diff(self.U, axis=0) / self._dy[:, np.newaxis]
            * (self._y[:, np.newaxis] - self._ybounds[:-1, np.newaxis])
            + self.U[:-1, :]
        )
        self.gp_source_z = W_mid * dGPdp
        self.gp_source_y = U_mid * dGPdy
        return {'se': self.gp_source_y + self.gp_source_z}


class MoistMeridionalAdvectionDiffusion(TimeDependentProcess):
    """Combined moist static energy and moisture transport.

    Couples MSE transport (advection-diffusion) with moisture transport,
    allowing either 1D meridional diffusion or 2D advection-diffusion
    for each component.

    Parameters
    ----------
    K : ndarray or tuple
        MSE diffusivity. For 2D: (Kyy, Kzz, Kyz)
    U : ndarray or tuple
        MSE velocity. For 2D: (U_lat, W_lev)
    Kq : ndarray or tuple
        Moisture diffusivity. For 2D: (Kyy, Kzz, Kyz)
    Uq : ndarray or tuple
        Moisture velocity. For 2D: (U_lat, W_lev)
    q_transport_2d : bool
        Use 2D transport for moisture (default: False)
    mse_transport_2d : bool
        Use 2D transport for MSE (default: False)
    do_analytic_gp : bool
        Compute geopotential from temperature (default: False)
    geopotential : ndarray, optional
        Fixed geopotential field (for FixedGeoPotentialFlux)
    """
    def __init__(self, K=0., U=0., Kq=0., Uq=0., **kwargs):
        super(MoistMeridionalAdvectionDiffusion, self).__init__(**kwargs)
        assert hasattr(self.state, 'q'), (
            'q must be a state parameter'
        )
        assert hasattr(self.state, 'Tatm'), (
            'Tatm must be a state parameter'
        )
        self.geopotential = kwargs.get('geopotential', 0.0)
        self.se_obj = SpecificEnthalpy(state=self.state)
        q_transport_2d = kwargs.get('q_transport_2d', False)
        mse_transport_2d = kwargs.get('mse_transport_2d', False)
        self.do_analytic_gp = kwargs.get('do_analytic_gp', False)
        dU = kwargs.get('dU', (0.0, 0.0))

        if q_transport_2d:
            q_source_param_dict = kwargs.get('q_source_param_dict', {})
            q_source_func_dict = {}
            for k in ['fy_source_func', 'fz_source_func', 'source_func']:
                if f'q_{k}' in kwargs:
                    q_source_func_dict[k] = kwargs[f'q_{k}']
            rho = (
                self.state['Tatm'].domain.lev.points[None, :]
                * const.mb_to_Pa / const.Rd / self.state['Tatm']
            )
            self.diff_q = TwoDimensionalAdvectionDiffusion(
                name='q Transport 2D',
                state={'q': self.state['q']},
                timestep=self.timestep, U=Uq[0], W=Uq[1],
                Kyy=Kq[0], Kzz=Kq[1], Kyz=Kq[2], rho=rho,
                source_param_dict=q_source_param_dict,
                **q_source_func_dict
            )
        else:
            self.diff_q = MeridionalAdvectionDiffusion(
                name='q Diffusion',
                state={'q': self.state['q']},
                K=Kq.T, U=Uq.T, timestep=self.timestep
            )

        if mse_transport_2d:
            mse_source_param_dict = kwargs.get('mse_source_param_dict', {})
            mse_source_func_dict = {}
            for k in ['fy_source_func', 'fz_source_func', 'source_func']:
                if f'mse_{k}' in kwargs:
                    mse_source_func_dict[k] = kwargs[f'mse_{k}']
            rho = (
                self.state['Tatm'].domain.lev.points[None, :]
                * const.mb_to_Pa / const.Rd / self.state['Tatm']
            )
            if self.do_analytic_gp:
                geopotential_obj = GeoPotentialFlux(
                    U=1 * U[0], W=1 * U[1], T=self.state['Tatm'],
                    name='Geopotential Flux',
                    state={'se': self.se_obj.se},
                    timestep=self.timestep
                )
            else:
                geopotential_obj = FixedGeoPotentialFlux(
                    name='Geopotential Flux',
                    state={'se': self.se_obj.se},
                    timestep=self.timestep,
                    U=1 * U[0], W=1 * U[1],
                    geopotential=self.geopotential
                )
            self.diff_mse = couple([
                TwoDimensionalAdvectionDiffusion(
                    name='SE Transport 2D',
                    state={'se': self.se_obj.se},
                    timestep=self.timestep,
                    U=1 * U[0] + dU[0], W=1 * U[1] + dU[1],
                    Kyy=K[0], Kzz=K[1], Kyz=K[2], rho=rho,
                    source_param_dict=mse_source_param_dict,
                    **mse_source_func_dict
                ),
                geopotential_obj
            ], name='MSE transport 2D')
            self.diff_mse.compute()
        else:
            self.diff_mse = MeridionalDiffusion(
                name='SE Diffusion',
                state={'se': self.se_obj.se},
                K=K, timestep=self.timestep
            )

    def _compute(self):
        self.diff_q.timestep = self.timestep
        self.diff_mse.timestep = self.timestep
        self.se_obj._compute_se()
        dict_q = self.diff_q._compute()
        dict_se = self.diff_mse.compute()
        dict_t = {
            'Tatm': self.se_obj.get_T_part(dict_se['se'], dict_q['q'])
        }
        return {'q': dict_q['q'], 'Tatm': dict_t['Tatm']}
