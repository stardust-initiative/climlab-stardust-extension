"""Tests for aerosol optical depth tables module.

Uses mock NetCDF fixtures to avoid network calls and external data
dependencies.  The ``load_repo_table`` and ``load_config`` functions
are patched so that AerosolsOptDepTables reads from local temp files.
"""

import numpy as np
import pytest
import climlab
from unittest.mock import patch, MagicMock

from climlab_stardust_extension.radiation.optical_depth_tables_aerosols import (
    AerosolsOptDepTables,
    aerosol_instance,
    construct_uni_layer_vmr_p_based,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_config():
    """Return a mock config dict matching what load_config returns."""
    return {
        'aerosols_opt_tables_http': 'https://mock.example.com/repo/abc',
        'aerosols_tables_dict': {
            'TestMaterial': {'mat_path': 'tables/test.nc'},
        },
        'aerosols_token': 'mock_token',
        'proj_name': 'test_project',
    }


def _make_aerosol(nc_path, material='TestMaterial', r_m=0.5e-6, nlev=20):
    """Build an AerosolsOptDepTables instance backed by *nc_path*.

    Patches load_repo_table and load_config so no network calls occur.
    """
    state = climlab.column_state(num_lev=nlev)
    domain = state.Tatm.domain
    vmr = 1e-9 * np.ones(nlev)
    coszen = np.array([0.5])

    aero_list = [aerosol_instance(material, r_m, vmr)]
    config = _mock_config()
    config['aerosols_tables_dict'] = {material: {'mat_path': 'tables/test.nc'}}

    with patch(
        'climlab_stardust_extension.radiation.optical_depth_tables_aerosols.load_config',
        return_value=config,
    ), patch(
        'climlab_stardust_extension.radiation.optical_depth_tables_aerosols.load_repo_table',
        return_value=(nc_path, False),
    ):
        aero = AerosolsOptDepTables(
            domain=domain,
            aerosol_instance_list=aero_list,
            coszen=coszen,
        )
    return aero


# ---------------------------------------------------------------------------
# Static method tests (no mocking needed)
# ---------------------------------------------------------------------------

class TestGetInterpParam:
    """Tests for AerosolsOptDepTables.get_interp_param."""

    def test_exact_match_interior(self):
        x_vect = np.array([0.0, 1.0, 2.0, 3.0])
        ix, fac = AerosolsOptDepTables.get_interp_param(2.0, x_vect)
        # Exact match at index 2 — fac should be 1.0 (weight on ix)
        # or 0.0 (weight on ix+1).  Either way, result is exact.
        assert 0 <= ix <= len(x_vect) - 1
        assert 0.0 <= fac <= 1.0

    def test_below_minimum(self):
        x_vect = np.array([1.0, 2.0, 3.0])
        ix, fac = AerosolsOptDepTables.get_interp_param(0.0, x_vect)
        assert ix == 0
        assert fac == 1.0  # clamp to lower bound

    def test_above_maximum(self):
        x_vect = np.array([1.0, 2.0, 3.0])
        ix, fac = AerosolsOptDepTables.get_interp_param(5.0, x_vect)
        # Should clamp to upper bound
        assert fac == 0.0

    def test_midpoint(self):
        x_vect = np.array([0.0, 2.0, 4.0])
        ix, fac = AerosolsOptDepTables.get_interp_param(1.0, x_vect)
        # 1.0 is between x_vect[0]=0 and x_vect[1]=2
        assert ix == 0
        assert fac == pytest.approx(0.5)


class TestToGray:
    def test_uniform_unchanged(self):
        arr = 3.0 * np.ones((5, 14))
        gray = AerosolsOptDepTables.to_gray(arr)
        np.testing.assert_allclose(gray, 3.0)

    def test_shape_preserved(self):
        arr = np.random.rand(5, 14)
        gray = AerosolsOptDepTables.to_gray(arr)
        assert gray.shape == arr.shape

    def test_all_bands_equal(self):
        arr = np.random.rand(5, 14)
        gray = AerosolsOptDepTables.to_gray(arr)
        # All columns in each row should be the same
        for row in gray:
            np.testing.assert_allclose(row, row[0])


# ---------------------------------------------------------------------------
# Initialization tests (require mock NetCDF)
# ---------------------------------------------------------------------------

class TestAerosolsOptDepTablesInit:
    def test_basic_init_no_coszen(self, mock_aerosol_nc_no_coszen):
        aero = _make_aerosol(mock_aerosol_nc_no_coszen)
        assert aero.n_mat == 1
        assert len(aero.babs_lw_dict) == 1
        assert len(aero.bext_sw_dict) == 1
        assert aero.has_mu is False

    def test_basic_init_with_coszen(self, mock_aerosol_nc_with_coszen):
        # _compute_optical_depths assumes 1D spectral arrays per material;
        # with coszen, bext_sw_dict entries are 2D (nsw, n_coszen), so we
        # patch _compute_optical_depths to test the table-loading path only.
        with patch.object(AerosolsOptDepTables, '_compute_optical_depths'):
            aero = _make_aerosol(mock_aerosol_nc_with_coszen)
        assert aero.n_mat == 1
        assert aero.has_mu is True
        key = list(aero.mu_dict.keys())[0]
        assert len(aero.mu_dict[key]) > 0
        # bext_sw should retain the coszen dimension
        assert aero.bext_sw_dict[key].ndim == 2


class TestOpticalDepthComputation:
    def test_shapes_single_column(self, mock_aerosol_nc_no_coszen):
        nlev = 20
        aero = _make_aerosol(mock_aerosol_nc_no_coszen, nlev=nlev)
        # tauaer_lw: (nlw, nlev), tauaer_sw: (nsw, nlev)
        assert aero.tauaer_lw.shape == (16, nlev)
        assert aero.tauaer_sw.shape == (14, nlev)
        assert aero.ssaaer_sw.shape == (14, nlev)
        assert aero.asmaer_sw.shape == (14, nlev)

    def test_values_positive(self, mock_aerosol_nc_no_coszen):
        aero = _make_aerosol(mock_aerosol_nc_no_coszen)
        assert np.all(aero.tauaer_lw >= 0)
        assert np.all(aero.tauaer_sw >= 0)

    def test_ssa_in_valid_range(self, mock_aerosol_nc_no_coszen):
        aero = _make_aerosol(mock_aerosol_nc_no_coszen)
        # Single-scattering albedo should be in [0, 1]
        assert np.all(aero.ssaaer_sw >= 0)
        assert np.all(aero.ssaaer_sw <= 1.0 + 1e-10)

    def test_tau_scales_with_vmr(self, mock_aerosol_nc_no_coszen):
        """Doubling VMR should approximately double optical depth."""
        aero1 = _make_aerosol(mock_aerosol_nc_no_coszen)
        tau1 = aero1.tauaer_sw.copy()

        # Update VMR to double
        key = list(aero1.vmr_dict.keys())[0]
        new_vmr_dict = {key: aero1.vmr_dict[key] * 2.0}
        aero1.vmr_dict = new_vmr_dict
        tau2 = aero1.tauaer_sw.copy()

        # Should be approximately 2x (exact for linear dependence)
        ratio = tau2 / (tau1 + 1e-30)
        np.testing.assert_allclose(ratio[tau1 > 1e-20], 2.0, rtol=1e-10)


class TestVMRUpdate:
    def test_update_vmr_changes_tau(self, mock_aerosol_nc_no_coszen):
        aero = _make_aerosol(mock_aerosol_nc_no_coszen)
        tau_before = aero.tauaer_sw.copy()

        key = list(aero.vmr_dict.keys())[0]
        new_vmr = aero.vmr_dict[key] * 3.0
        new_list = [aerosol_instance(key[0], key[1], new_vmr)]
        aero.update_vmr(new_list)

        tau_after = aero.tauaer_sw.copy()
        assert not np.allclose(tau_before, tau_after)

    def test_update_vmr_wrong_keys_raises(self, mock_aerosol_nc_no_coszen):
        aero = _make_aerosol(mock_aerosol_nc_no_coszen)
        wrong_list = [aerosol_instance('WrongMaterial', 1e-6, np.ones(20))]
        with pytest.raises(AssertionError):
            aero.update_vmr(wrong_list)


# ---------------------------------------------------------------------------
# VMR construction helpers
# ---------------------------------------------------------------------------

class TestConstructUniLayerVmrPBased:
    def test_basic_shape(self):
        state = climlab.column_state(num_lev=30)
        rho_particle = 2200.0  # kg/m^3 (silica)
        density = 1e-3  # kg/m^2
        vmr = construct_uni_layer_vmr_p_based(
            rho_particle, density,
            p_min=100.0, p_max=200.0,
            r_m=250e-9, state=state,
        )
        assert vmr.shape[-1] == 30  # nlev

    def test_vmr_nonnegative(self):
        state = climlab.column_state(num_lev=30)
        vmr = construct_uni_layer_vmr_p_based(
            2200.0, 1e-3,
            p_min=100.0, p_max=200.0,
            r_m=250e-9, state=state,
        )
        assert np.all(vmr >= 0)

    def test_vmr_localised(self):
        """VMR should be non-zero only between p_min and p_max."""
        state = climlab.column_state(num_lev=50)
        p_min, p_max = 200.0, 400.0
        vmr = construct_uni_layer_vmr_p_based(
            2200.0, 1e-3,
            p_min=p_min, p_max=p_max,
            r_m=250e-9, state=state,
        )
        vmr = np.squeeze(vmr)
        lev = state.Tatm.domain.lev.points
        # Levels well outside the layer should have zero VMR
        outside = (lev < p_min - 50) | (lev > p_max + 50)
        if np.any(outside):
            assert np.allclose(vmr[outside], 0.0, atol=1e-20)

    def test_2d_state(self):
        """With a multi-latitude state, VMR should have 2D shape."""
        state = climlab.column_state(num_lev=30, num_lat=5)
        vmr = construct_uni_layer_vmr_p_based(
            2200.0, 1e-3 * np.ones(5),
            p_min=100.0, p_max=200.0,
            r_m=250e-9, state=state,
        )
        assert vmr.shape == (5, 30)


# ---------------------------------------------------------------------------
# _compute method
# ---------------------------------------------------------------------------

class TestComputeMethod:
    def test_compute_returns_empty_tendencies(self, mock_aerosol_nc_no_coszen):
        aero = _make_aerosol(mock_aerosol_nc_no_coszen)
        result = aero._compute()
        assert isinstance(result, dict)
        assert len(result) == 0
