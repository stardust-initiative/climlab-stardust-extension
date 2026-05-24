"""Tests for surface modules: SensibleHeatFlux_extended, LatentHeatFlux_extended, OceanicHeatUptake."""

import numpy as np
import pytest
import climlab
from climlab.domain.field import Field


class TestSensibleHeatFluxExtended:
    @pytest.fixture
    def column_state(self):
        state = climlab.column_state(num_lev=30)
        return state

    def test_instantiation_default(self, column_state):
        from climlab_stardust_extension.surface.turbulent import (
            SensibleHeatFlux_extended,
        )
        proc = SensibleHeatFlux_extended(state=column_state)
        assert proc.p_turb_layer == 0.0
        assert proc.do_analytic is False
        # Weight should be 1 at lowest level, 0 elsewhere
        assert proc.weight[..., -1] == 1.0

    def test_turb_layer(self, column_state):
        from climlab_stardust_extension.surface.turbulent import (
            SensibleHeatFlux_extended,
        )
        proc = SensibleHeatFlux_extended(
            state=column_state, p_turb_layer=100.0,
        )
        assert proc.p_turb_layer == 100.0
        # Weight should sum to approximately 1
        assert proc.weight.sum() == pytest.approx(1.0, abs=0.01)
        # Regression: the exponential profile must actually spread the
        # flux across multiple levels. Prior to the kwargs-plumbing fix,
        # p_turb_layer was silently lost in super().__init__ and
        # self.weight collapsed to [0,...,0,1] (bottom level only),
        # even though self.weight.sum() stayed at 1.0.
        assert proc.weight[..., -1] < 0.9
        assert (proc.weight > 1e-6).sum() >= 2

    def test_do_analytic_preserved(self, column_state):
        from climlab_stardust_extension.surface.turbulent import (
            SensibleHeatFlux_extended,
        )
        proc = SensibleHeatFlux_extended(
            state=column_state, do_analytic=True,
        )
        assert proc.do_analytic is True

    def test_do_external_flux_value(self, column_state):
        """do_external=True must route SHF_external through to the flux."""
        from climlab_stardust_extension.surface.turbulent import (
            SensibleHeatFlux_extended,
        )
        SHF_ext = 42.0 * np.ones_like(column_state['Ts'])
        proc = SensibleHeatFlux_extended(
            state=column_state,
            do_external=True,
            SHF_external=SHF_ext,
        )
        assert proc.do_external is True
        proc.step_forward()
        np.testing.assert_allclose(
            np.asarray(proc.SHF), SHF_ext, rtol=1e-12,
        )

    def test_compute(self, column_state):
        from climlab_stardust_extension.surface.turbulent import (
            SensibleHeatFlux_extended,
        )
        proc = SensibleHeatFlux_extended(state=column_state)
        proc.step_forward()
        # After a step, SHF diagnostic should exist
        assert hasattr(proc, 'SHF')


class TestLatentHeatFluxExtended:
    @pytest.fixture
    def column_state_with_q(self):
        state = climlab.column_state(num_lev=30)
        q = 5e-3 * np.ones_like(state.Tatm)
        state['q'] = q
        return state

    def test_instantiation(self, column_state_with_q):
        from climlab_stardust_extension.surface.turbulent import (
            LatentHeatFlux_extended,
        )
        proc = LatentHeatFlux_extended(state=column_state_with_q)
        assert proc.p_turb_layer == 0.0
        assert proc.do_analytic is False

    def test_turb_layer(self, column_state_with_q):
        from climlab_stardust_extension.surface.turbulent import (
            LatentHeatFlux_extended,
        )
        proc = LatentHeatFlux_extended(
            state=column_state_with_q, p_turb_layer=100.0,
        )
        assert proc.p_turb_layer == 100.0
        assert proc.weight.sum() == pytest.approx(1.0, abs=0.01)
        # Regression: p_turb_layer must actually produce a spread profile.
        # Prior to the fix the weight collapsed to [0,...,0,1].
        assert proc.weight[..., -1] < 0.9
        assert (proc.weight > 1e-6).sum() >= 2

    def test_do_analytic_preserved(self, column_state_with_q):
        from climlab_stardust_extension.surface.turbulent import (
            LatentHeatFlux_extended,
        )
        proc = LatentHeatFlux_extended(
            state=column_state_with_q, do_analytic=True,
        )
        assert proc.do_analytic is True

    def test_do_external_flux_value(self, column_state_with_q):
        """do_external=True must route LHF_external through to the flux."""
        from climlab_stardust_extension.surface.turbulent import (
            LatentHeatFlux_extended,
        )
        LHF_ext = 123.0 * np.ones_like(column_state_with_q['Ts'])
        proc = LatentHeatFlux_extended(
            state=column_state_with_q,
            do_external=True,
            LHF_external=LHF_ext,
        )
        assert proc.do_external is True
        proc.step_forward()
        np.testing.assert_allclose(
            np.asarray(proc.LHF), LHF_ext, rtol=1e-12,
        )

    def test_qsat_params(self, column_state_with_q):
        from climlab_stardust_extension.surface.turbulent import (
            LatentHeatFlux_extended,
        )
        proc = LatentHeatFlux_extended(
            state=column_state_with_q,
            qsat_param_dict={'do_era5': True, 'small': 0.0, 'do_simplified': False},
        )
        assert proc.qsat_param_dict['do_era5'] is True


class TestOceanicHeatUptake:
    def test_instantiation(self):
        from climlab_stardust_extension.surface.oceanic_heat_uptake import (
            OceanicHeatUptake,
        )
        state = climlab.column_state(num_lev=30)
        # Add deep ocean temperature
        Td = Field(np.array([280.0]), domain=state.Ts.domain)
        state['Td'] = Td
        proc = OceanicHeatUptake(state=state, gamma_u=0.7, tau_d=500.0)
        assert hasattr(proc, 'oceanic_heat_uptake')
        assert hasattr(proc, 'log_Td_time_jumps')

    def test_compute_returns_tendencies(self):
        from climlab_stardust_extension.surface.oceanic_heat_uptake import (
            OceanicHeatUptake,
        )
        state = climlab.column_state(num_lev=30)
        Td = Field(np.array([280.0]), domain=state.Ts.domain)
        state['Td'] = Td
        proc = OceanicHeatUptake(state=state, gamma_u=0.7)
        tend = proc._compute()
        assert 'Ts' in tend
        assert 'Td' in tend

    def test_td_time_jump(self):
        from climlab_stardust_extension.surface.oceanic_heat_uptake import (
            OceanicHeatUptake,
        )
        state = climlab.column_state(num_lev=30)
        Td = Field(np.array([280.0]), domain=state.Ts.domain)
        state['Td'] = Td
        # Set Ts different from Td
        state['Ts'][:] = 290.0
        proc = OceanicHeatUptake(state=state, gamma_u=0.7, tau_d=100.0)
        Td_before = proc.state.Td.copy()
        proc.Td_time_jump(100.0)
        # Td should move toward Ts
        assert np.all(proc.state.Td > Td_before)
