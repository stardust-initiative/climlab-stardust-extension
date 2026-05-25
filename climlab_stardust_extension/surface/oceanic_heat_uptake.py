"""Oceanic heat uptake process with a deep ocean temperature layer."""

import numpy as np
from climlab.process.time_dependent_process import TimeDependentProcess
from climlab.utils import constants as const


class OceanicHeatUptake(TimeDependentProcess):
    """Two-layer ocean heat uptake process.

    Couples surface temperature (Ts) with a deep ocean temperature (Td)
    via a heat uptake efficiency parameter. Supports analytic time-jumping
    for the deep ocean layer.

    Parameters
    ----------
    gamma_u : float or ndarray
        Heat uptake efficiency (W/m2/K) (default: 0.0)
    tau_d : float or ndarray
        Deep ocean relaxation timescale in years (default: 1000)
    max_time_jumps : int
        Maximum number of Td time-jumps to log (default: 100)
    """
    def __init__(self, gamma_u=0.0, tau_d=1000.0, max_time_jumps=100,
                 **kwargs):
        super(OceanicHeatUptake, self).__init__(**kwargs)
        assert 'Ts' in self.state.keys(), (
            f'Ts is required for {__name__} module'
        )
        assert 'Td' in self.state.keys(), (
            f'Td is required for {__name__} module'
        )
        if isinstance(gamma_u, np.ndarray):
            self.gamma_u = gamma_u + 0.0 * self.state.Ts
        else:
            self.gamma_u = gamma_u * np.ones_like(self.state.Ts.domain)
        if isinstance(tau_d, np.ndarray):
            self.tau_d = tau_d + 0.0 * self.state.Td
        else:
            self.tau_d = tau_d * np.ones_like(self.state.Td.domain)
        self.add_diagnostic(
            'oceanic_heat_uptake', 0. * self.state.Ts
        )
        self.add_diagnostic(
            'log_Td_time_jumps', -np.ones((max_time_jumps, 2))
        )
        self._compute()

    def _compute(self):
        self.oceanic_heat_uptake[:] = (
            self.gamma_u * (self.state.Ts - self.state.Td)
        )
        tendencies = {}
        tendencies['Ts'] = (
            -self.oceanic_heat_uptake / self.Ts.domain.heat_capacity
        )
        tendencies['Td'] = (
            (self.state.Ts - self.state.Td)
            / (const.seconds_per_year * self.tau_d)
        )
        return tendencies

    def Td_time_jump(self, delta_t_years):
        """Jump the deep ocean temperature forward analytically.

        Assumes Ts is approximately fixed, and propagates Td by
        delta_t_years analytically.

        Parameters
        ----------
        delta_t_years : float
            Time to jump forward in years
        """
        self.state.Td[:] = (
            self.state.Td[:]
            + (self.state.Ts[:] - self.state.Td[:])
            * np.exp(-delta_t_years / self.tau_d)
        )
        k = 0
        while self.log_Td_time_jumps[k, 0] != -1:
            k += 1
        self.log_Td_time_jumps[k, :] = np.array(
            [self.time['steps'], delta_t_years]
        )
