"""Tests for convection modules: LargeScaleCondensation_extended, WeakTemperatureGradient,
SimplifiedBettsMiller_extended."""

import numpy as np
import pytest
import climlab


class TestLargeScaleCondensationExtended:
    @pytest.fixture
    def column_state(self):
        """Single-column state with Tatm, Ts, q."""
        state = climlab.column_state(num_lev=30)
        q = 1e-3 * np.ones_like(state.Tatm)
        state['q'] = q
        return state

    def test_instantiation(self, column_state):
        from climlab_stardust_extension.dynamics.large_scale_condensation import (
            LargeScaleCondensation_extended,
        )
        proc = LargeScaleCondensation_extended(state=column_state)
        assert hasattr(proc, 'Tatm')
        assert hasattr(proc, 'q')

    def test_inherits_from_climlab_base(self, column_state):
        from climlab.dynamics.large_scale_condensation import (
            LargeScaleCondensation,
        )
        from climlab_stardust_extension.dynamics.large_scale_condensation import (
            LargeScaleCondensation_extended,
        )
        proc = LargeScaleCondensation_extended(state=column_state)
        assert isinstance(proc, LargeScaleCondensation)

    def test_compute_returns_tendencies(self, column_state):
        from climlab_stardust_extension.dynamics.large_scale_condensation import (
            LargeScaleCondensation_extended,
        )
        proc = LargeScaleCondensation_extended(state=column_state)
        tend = proc._compute()
        assert 'Tatm' in tend
        assert 'q' in tend

    def test_condensation_reduces_q(self, column_state):
        """When q > qsat, condensation should produce negative q tendency."""
        from climlab_stardust_extension.dynamics.large_scale_condensation import (
            LargeScaleCondensation_extended,
        )
        # Set very high q to guarantee supersaturation
        column_state['q'][:] = 0.1
        proc = LargeScaleCondensation_extended(state=column_state)
        tend = proc._compute()
        # At least some levels should have negative q tendency
        assert np.any(tend['q'] < 0)

    def test_diagnostics_registered(self, column_state):
        """Check that all three diagnostics exist with correct names."""
        from climlab_stardust_extension.dynamics.large_scale_condensation import (
            LargeScaleCondensation_extended,
        )
        proc = LargeScaleCondensation_extended(state=column_state)
        assert 'latent_heating' in proc.diagnostics
        assert 'precipitation' in proc.diagnostics
        assert 'precipitation_2D' in proc.diagnostics

    def test_pmin_suppresses_upper_levels(self, column_state):
        """Condensation tendencies should be zero above pmin."""
        from climlab_stardust_extension.dynamics.large_scale_condensation import (
            LargeScaleCondensation_extended,
        )
        column_state['q'][:] = 0.1  # supersaturated everywhere
        pmin = 200.0
        proc = LargeScaleCondensation_extended(
            state=column_state, pmin=pmin,
        )
        tend = proc._compute()
        lev = proc.Tatm.domain.lev.points
        upper_idx = np.where(lev <= pmin)[0]
        if len(upper_idx) > 0:
            np.testing.assert_array_equal(tend['q'][upper_idx], 0.0)

    def test_convection_init_import(self):
        """The extended class should be importable via convection.__init__."""
        from climlab_stardust_extension.convection import (
            LargeScaleCondensation_extended,
        )
        assert LargeScaleCondensation_extended is not None


class TestWeakTemperatureGradient:
    def test_instantiation(self):
        from climlab_stardust_extension.convection.large_scale_convection import (
            WeakTemperatureGradient,
        )
        # WTG needs multi-latitude state with 'q'
        state = climlab.column_state(num_lev=30, num_lat=10)
        q = 1e-3 * np.ones_like(state.Tatm)
        state['q'] = q
        proc = WeakTemperatureGradient(
            state=state, relaxation_time=24*3600.0,
        )
        assert hasattr(proc, 'relaxation_time')

    def test_compute_returns_tendency(self):
        from climlab_stardust_extension.convection.large_scale_convection import (
            WeakTemperatureGradient,
        )
        state = climlab.column_state(num_lev=30, num_lat=10)
        q = 1e-3 * np.ones_like(state.Tatm)
        state['q'] = q
        proc = WeakTemperatureGradient(
            state=state, relaxation_time=24*3600.0,
        )
        tend = proc._compute()
        assert 'Tatm' in tend

    def test_convection_init_import(self):
        """WeakTemperatureGradient should be importable via convection.__init__."""
        from climlab_stardust_extension.convection import (
            WeakTemperatureGradient,
        )
        assert WeakTemperatureGradient is not None


class TestSimplifiedBettsMillerExtended:
    def test_instantiation(self):
        from climlab_stardust_extension.convection.simplified_betts_miller import (
            SimplifiedBettsMiller_extended,
        )
        state = climlab.column_state(num_lev=30)
        q = 1e-3 * np.ones_like(state.Tatm)
        state['q'] = q
        proc = SimplifiedBettsMiller_extended(state=state, sp=1013.0, pmin=10.0)
        assert proc.pmin == 10.0

    def test_pmin_zeroes_upper_tendencies(self):
        """Tendencies above pmin should be zero."""
        from climlab_stardust_extension.convection.simplified_betts_miller import (
            SimplifiedBettsMiller_extended,
        )
        state = climlab.column_state(num_lev=30)
        q = 5e-3 * np.ones_like(state.Tatm)
        state['q'] = q
        pmin = 100.0  # zero tendencies above 100 hPa
        proc = SimplifiedBettsMiller_extended(state=state, pmin=pmin)
        tend = proc._compute()
        lev = proc.Tatm.domain.lev.points
        upper_idx = np.where(lev <= pmin)[0]
        if len(upper_idx) > 0:
            np.testing.assert_array_equal(tend['Tatm'][upper_idx], 0.0)
            np.testing.assert_array_equal(tend['q'][upper_idx], 0.0)

    def test_convection_triggers_on_unstable_profile(self):
        """Regression test for the es0-normalization bug.

        es0 is an internal normalization constant of the `lcltabl` lookup
        table in climlab_sbm_convection's Fortran betts_miller: the table
        is calibrated assuming es0 = 1. An earlier version of this wrapper
        passed es0 = 610.78 (the physical saturation vapor pressure at 0°C)
        which silently shifted the lookup argument by ln(610.78) ≈ 6.4
        and returned wildly wrong LCL temperatures, silencing convective
        triggering over the entire column.

        This test gives the scheme a convectively unstable warm/moist
        tropical profile and asserts that it actually produces nonzero
        rain and nonzero temperature tendencies. With es0 = 610.78 the
        scheme returned zero everywhere and this test would fail.
        """
        from climlab_stardust_extension.convection.simplified_betts_miller import (
            SimplifiedBettsMiller_extended,
        )
        # Build a single-column warm-moist profile that should trigger
        # deep convection in any reasonable SBM setup.
        nlev = 30
        state = climlab.column_state(num_lev=nlev)
        lev = state['Tatm'].domain.lev.points  # hPa, TOA first
        # Moist adiabatic-ish troposphere, cold stratosphere
        Tatm = np.where(
            lev > 100.0,
            300.0 - 6.5e-3 * 8000.0 * np.log(1000.0 / np.maximum(lev, 1.0)),
            215.0,
        )
        state['Tatm'][:] = Tatm
        state['Ts'][:] = 300.0
        # Exponential q profile peaking near the surface, ~tropics
        q_profile = 0.02 * np.exp(-(1000.0 - lev) / 2000.0)
        q_profile = np.clip(q_profile, 1e-6, 0.025)
        # Multiplying by ones_like(Tatm) inherits the Field wrapper so
        # climlab accepts it as a state variable.
        state['q'] = q_profile * np.ones_like(state.Tatm)

        proc = SimplifiedBettsMiller_extended(
            state=state, sp=1013.0, pmin=10.0,
            tau_bm=7200.0, do_envsat=True,
            do_shallower=True, do_changeqref=True,
        )
        tend = proc._compute()

        # Core invariant: in an unstable tropical column, SBM must fire.
        rain_max = float(np.max(np.abs(proc.precipitation)))
        tdel_max = float(np.max(np.abs(tend['Tatm'])))
        qdel_max = float(np.max(np.abs(tend['q'])))
        assert rain_max > 0.0, (
            f'precipitation should be nonzero in a warm/moist tropical '
            f'column; got {rain_max}. This likely indicates the es0 '
            f'lookup-table normalization has regressed away from 1.0.'
        )
        assert tdel_max > 0.0, (
            f'Tatm tendency should be nonzero; got {tdel_max}'
        )
        assert qdel_max > 0.0, (
            f'q tendency should be nonzero; got {qdel_max}'
        )
