"""Tests for dynamics modules: meridional moist diffusion.

Advection-diffusion numerics and the 2-D transport solver are covered by
test_two_d_adv_diff_numerics.py and test_two_d_advdiff_solver.py.
"""

import numpy as np
import pytest
import climlab


class TestMeridionalMoistDiffusion:
    def test_moist_amplification_factor(self):
        """Moist amplification should be >= 1 at warm temperatures."""
        from climlab_stardust_extension.dynamics.meridional_moist_diffusion import (
            moist_amplification_factor_extended,
        )
        from climlab.utils.thermo import qsat
        T = np.array([280., 290., 300.])
        p = 1000.
        Mf = moist_amplification_factor_extended(T, p)
        assert np.all(Mf >= 1.0)
        # Should increase with temperature
        assert Mf[2] > Mf[0]

    def test_moist_amplification_era5(self):
        """ERA5 mode should also produce valid amplification factors."""
        from climlab_stardust_extension.dynamics.meridional_moist_diffusion import (
            moist_amplification_factor_extended,
        )
        T = np.array([280., 290., 300.])
        p = 1000.
        Mf = moist_amplification_factor_extended(T, p, do_era5=True)
        assert np.all(Mf >= 1.0)
        assert np.all(np.isfinite(Mf))

    def test_specific_enthalpy(self):
        """SpecificEnthalpy diagnostic should compute se = cp*T + Lhvap*q."""
        from climlab_stardust_extension.dynamics.meridional_moist_diffusion import (
            SpecificEnthalpy,
        )
        from climlab import constants as const
        state = climlab.column_state(num_lev=20)
        q = 5e-3 * np.ones_like(state.Tatm)
        state['q'] = q
        proc = SpecificEnthalpy(state=state)
        proc._compute()
        expected = const.cp * state.Tatm + const.Lhvap * q
        np.testing.assert_allclose(proc.se, expected, rtol=1e-10)
