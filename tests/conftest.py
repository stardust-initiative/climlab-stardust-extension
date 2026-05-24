"""Shared pytest fixtures for climlab-stardust-extension tests."""

import os
import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def tmp_netcdf_dir(tmp_path):
    """Temporary directory for mock NetCDF files."""
    return tmp_path


def _create_mock_aerosol_nc(path, n_mu=5, n_lw=16, n_sw=14,
                            include_coszen=False, n_coszen=10, seed=42):
    """Create a mock Mie-theory optical-property NetCDF file.

    Returns the file path.
    """
    rng = np.random.RandomState(seed)
    mu_samples = np.linspace(-5.0, 2.5, n_mu)

    babs_lw = rng.uniform(0.1, 5.0, (n_mu, n_lw))
    bet_bar_bsca_sw = rng.uniform(0.01, 0.5, (n_mu, n_sw))

    if include_coszen:
        bext_sw = rng.uniform(10.0, 50.0, (n_mu, n_sw, n_coszen))
        bsca_sw = bext_sw * rng.uniform(0.1, 0.9, (n_mu, n_sw, n_coszen))
        basc_sw = bsca_sw * rng.uniform(0.1, 0.9, (n_mu, n_sw, n_coszen))
        bet_mu_bsca_sw = rng.uniform(0.01, 0.3, (n_mu, n_sw, n_coszen))
        coszen_vals = np.linspace(0.02, 1.0, n_coszen)
        data_vars = {
            'mu_samples': (['mu_dim'], mu_samples),
            'babs_lw': (['mu_dim', 'nlw'], babs_lw),
            'bext_sw': (['mu_dim', 'nsw', 'coszen_dim'], bext_sw),
            'bsca_sw': (['mu_dim', 'nsw', 'coszen_dim'], bsca_sw),
            'basc_sw': (['mu_dim', 'nsw', 'coszen_dim'], basc_sw),
            'bet_mu_bsca_sw': (['mu_dim', 'nsw', 'coszen_dim'], bet_mu_bsca_sw),
            'bet_bar_bsca_sw': (['mu_dim', 'nsw'], bet_bar_bsca_sw),
            'coszen': (['coszen_dim'], coszen_vals),
        }
    else:
        bext_sw = rng.uniform(10.0, 50.0, (n_mu, n_sw))
        bsca_sw = bext_sw * rng.uniform(0.1, 0.9, (n_mu, n_sw))
        basc_sw = bsca_sw * rng.uniform(0.1, 0.9, (n_mu, n_sw))
        bet_mu_bsca_sw = rng.uniform(0.01, 0.3, (n_mu, n_sw))
        data_vars = {
            'mu_samples': (['mu_dim'], mu_samples),
            'babs_lw': (['mu_dim', 'nlw'], babs_lw),
            'bext_sw': (['mu_dim', 'nsw'], bext_sw),
            'bsca_sw': (['mu_dim', 'nsw'], bsca_sw),
            'basc_sw': (['mu_dim', 'nsw'], basc_sw),
            'bet_mu_bsca_sw': (['mu_dim', 'nsw'], bet_mu_bsca_sw),
            'bet_bar_bsca_sw': (['mu_dim', 'nsw'], bet_bar_bsca_sw),
        }

    ds = xr.Dataset(data_vars)
    fpath = str(path)
    ds.to_netcdf(fpath)
    return fpath


@pytest.fixture
def mock_aerosol_nc_no_coszen(tmp_netcdf_dir):
    """Mock aerosol NetCDF *without* coszen dimension."""
    return _create_mock_aerosol_nc(
        tmp_netcdf_dir / 'mock_aerosol_no_coszen.nc',
        include_coszen=False,
    )


@pytest.fixture
def mock_aerosol_nc_with_coszen(tmp_netcdf_dir):
    """Mock aerosol NetCDF *with* coszen dimension."""
    return _create_mock_aerosol_nc(
        tmp_netcdf_dir / 'mock_aerosol_with_coszen.nc',
        include_coszen=True,
    )
