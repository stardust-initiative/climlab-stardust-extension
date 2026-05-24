"""Tests for radiation modules.

The extended RRTMG classes require the climlab-rrtmg-stardust Fortran
package.  Tests that call the Fortran are marked with ``@pytest.mark.rrtmg``
and can be skipped if the extension is not installed.
"""

import numpy as np
import pytest
import climlab


# Check if the stardust RRTMG Fortran extension is available
try:
    from climlab_rrtmg import rrtmg_lw as _rrtmg_lw
    _rrtmg_lw.climlab_rrtmg_lw_ensemble  # check for ensemble function
    HAS_RRTMG_STARDUST = True
except (ImportError, AttributeError):
    HAS_RRTMG_STARDUST = False

rrtmg = pytest.mark.skipif(
    not HAS_RRTMG_STARDUST,
    reason='climlab-rrtmg-stardust not installed',
)


class TestDailyAvgOfX:
    def test_basic_call(self):
        from climlab_stardust_extension.radiation.insolation import daily_avg_of_x
        lat = np.array([0., 30., 60., 90.])
        day = np.arange(365)
        mu_vect = np.linspace(0, 1, 50)
        # x = mu (identity function): result should be close to daily-average coszen
        x_mat = mu_vect[np.newaxis, :]
        result, Ho = daily_avg_of_x(lat, day, mu_vect, x_mat)
        assert result.shape == (1, len(lat))
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)

    def test_multi_row(self):
        from climlab_stardust_extension.radiation.insolation import daily_avg_of_x
        lat = np.array([0., 45.])
        day = np.array([172])  # summer solstice
        mu_vect = np.linspace(0, 1, 20)
        x_mat = np.vstack([mu_vect, mu_vect**2])
        result, Ho = daily_avg_of_x(lat, day, mu_vect, x_mat)
        assert result.shape == (2, len(lat))


@rrtmg
class TestRRTMG_LW_extended:
    def test_instantiation(self):
        from climlab_stardust_extension.radiation.rrtm.rrtmg_lw import (
            RRTMG_LW_extended,
        )
        state = climlab.column_state(num_lev=30)
        lw = RRTMG_LW_extended(state=state)
        assert hasattr(lw, 'do_col_by_col')
        assert hasattr(lw, 'do_seed_permutation')
        assert hasattr(lw, 'n_rrtmg_repeat')
        assert lw.iaer == 0

    def test_clearsky(self):
        from climlab_stardust_extension.radiation.rrtm.rrtmg_lw import (
            RRTMG_LW_extended,
        )
        state = climlab.column_state(num_lev=30)
        lw = RRTMG_LW_extended(state=state, icld=0)
        lw.step_forward()
        assert np.all(np.isfinite(lw.OLR))
        assert np.all(lw.OLR > 0)


@rrtmg
class TestRRTMG_SW_extended:
    def test_instantiation(self):
        from climlab_stardust_extension.radiation.rrtm.rrtmg_sw import (
            RRTMG_SW_extended,
        )
        state = climlab.column_state(num_lev=30)
        sw = RRTMG_SW_extended(state=state)
        assert hasattr(sw, 'do_col_by_col')
        assert sw.kmodts == 2

    def test_clearsky(self):
        from climlab_stardust_extension.radiation.rrtm.rrtmg_sw import (
            RRTMG_SW_extended,
        )
        state = climlab.column_state(num_lev=30)
        sw = RRTMG_SW_extended(state=state, icld=0)
        sw.step_forward()
        assert np.all(np.isfinite(sw.ASR))


@rrtmg
class TestRRTMG_extended:
    def test_instantiation(self):
        from climlab_stardust_extension.radiation.rrtm.rrtmg import (
            RRTMG_extended,
        )
        state = climlab.column_state(num_lev=30)
        rad = RRTMG_extended(state=state)
        # Should have extended LW and SW subprocesses
        assert 'LW' in rad.subprocess
        assert 'SW' in rad.subprocess
        from climlab_stardust_extension.radiation.rrtm.rrtmg_lw import RRTMG_LW_extended
        from climlab_stardust_extension.radiation.rrtm.rrtmg_sw import RRTMG_SW_extended
        assert isinstance(rad.subprocess['LW'], RRTMG_LW_extended)
        assert isinstance(rad.subprocess['SW'], RRTMG_SW_extended)

    def test_timestep_scaling(self):
        from climlab_stardust_extension.radiation.rrtm.rrtmg import (
            RRTMG_extended,
        )
        from climlab import constants as const
        state = climlab.column_state(num_lev=30)
        dt = const.seconds_per_day
        rad = RRTMG_extended(
            state=state,
            n_timestep_sw=2,
            n_timestep_lw=3,
            timestep=dt,
        )
        assert rad.subprocess['SW'].timestep == pytest.approx(2 * dt)
        assert rad.subprocess['LW'].timestep == pytest.approx(3 * dt)

    def test_clearsky_step(self):
        from climlab_stardust_extension.radiation.rrtm.rrtmg import (
            RRTMG_extended,
        )
        state = climlab.column_state(num_lev=30)
        rad = RRTMG_extended(state=state, icld=0)
        rad.step_forward()
        assert np.all(np.isfinite(rad.OLR))
        assert np.all(np.isfinite(rad.ASR))
