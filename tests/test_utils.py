"""Tests for utility modules: constants, thermo, file_handling."""

import numpy as np
import pytest


class TestConstants:
    def test_n_avogadro(self):
        from climlab_stardust_extension.utils.constants import n_avogadro
        assert n_avogadro == pytest.approx(6.02214067e23, rel=1e-6)


class TestThermo:
    def test_tetens_at_273(self):
        """Teten's formula should give ~611 Pa at 273.16 K."""
        from climlab_stardust_extension.utils.thermo import _tetens
        # _tetens requires (T, a1, a3, a4, T0); use ERA5 water coefficients
        a1, a3, a4, T0 = 611.21, 17.502, 32.19, 273.16
        es = _tetens(273.16, a1, a3, a4, T0)
        assert es == pytest.approx(611.21, rel=0.05)

    def test_clausius_clapeyron_extended_default(self):
        """Default (non-ERA5) should match the original climlab CC."""
        from climlab_stardust_extension.utils.thermo import clausius_clapeyron_extended
        from climlab.utils.thermo import clausius_clapeyron
        T = np.array([250., 273.15, 300.])
        es_ext = clausius_clapeyron_extended(T, do_era5=False)
        es_orig = clausius_clapeyron(T)
        np.testing.assert_allclose(es_ext, es_orig, rtol=1e-10)

    def test_clausius_clapeyron_extended_era5(self):
        """ERA5 mode should return finite positive values."""
        from climlab_stardust_extension.utils.thermo import clausius_clapeyron_extended
        T = np.array([200., 250., 273.15, 300., 320.])
        es = clausius_clapeyron_extended(T, do_era5=True)
        assert np.all(es > 0)
        assert np.all(np.isfinite(es))

    def test_qsat_extended_default(self):
        """qsat should return values in [0, 1] range for reasonable T, p."""
        from climlab_stardust_extension.utils.thermo import qsat_extended
        T = np.array([250., 273.15, 300.])
        p = 1000.  # hPa
        q = qsat_extended(T, p)
        assert np.all(q > 0)
        assert np.all(q < 1)

    def test_qsat_extended_with_small(self):
        """Adding 'small' regularisation should not break qsat."""
        from climlab_stardust_extension.utils.thermo import qsat_extended
        T = np.array([280.])
        p = 1000.
        q0 = qsat_extended(T, p, small=0.0)
        q1 = qsat_extended(T, p, small=1e-3)
        # small is a regularisation term; at normal p the effect is tiny
        np.testing.assert_allclose(q0, q1, rtol=1e-6)

    def test_dqsat_dT_positive(self):
        """Derivative of qsat w.r.t. T should be positive."""
        from climlab_stardust_extension.utils.thermo import dqsat_dT
        T = np.array([250., 273.15, 300.])
        p = 1000.
        dq = dqsat_dT(T, p)
        assert np.all(dq > 0)

    def test_clausius_clapeyron_T_deriv(self):
        """Derivative should be consistent with finite differences."""
        from climlab_stardust_extension.utils.thermo import (
            clausius_clapeyron_extended,
            clausius_clapeyron_T_deriv,
        )
        T = 280.0
        dT = 0.01
        numerical = (clausius_clapeyron_extended(T + dT) - clausius_clapeyron_extended(T - dT)) / (2 * dT)
        analytic = clausius_clapeyron_T_deriv(T)
        assert analytic == pytest.approx(numerical, rel=1e-4)
