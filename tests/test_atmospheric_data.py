r"""Tests for the AtmosphericData time/space-interpolation process.

AtmosphericData regrids time-varying atmospheric fields (winds, diffusivities,
tropopause) onto the simulation grid and interpolates them in time. The tests
run against stock upstream climlab.
"""
import numpy as np
import pytest
import xarray as xr
from datetime import datetime
from climlab.domain.domain import zonal_mean_column
from climlab.domain.field import Field

from climlab_stardust_extension.dynamics.atmospheric_data import AtmosphericData


def _monthly_dataarray(values_per_month, num_src_lat=5, num_src_lev=5):
    """A (month, lat, level) DataArray; each month's field is spatially uniform."""
    months = np.arange(1, 13)
    lat = np.linspace(-90, 90, num_src_lat)
    lev = np.linspace(1, 1010, num_src_lev)
    data = np.empty((12, num_src_lat, num_src_lev))
    for i, value in enumerate(values_per_month):
        data[i] = value
    return xr.DataArray(data, coords={'month': months, 'lat': lat, 'level': lev},
                        dims=['month', 'lat', 'level'])


def make_atm_data(values_per_month, time_type=1, num_lat=8, num_lev=6):
    _, atm = zonal_mean_column(num_lat=num_lat, num_lev=num_lev)
    state = {'dummy': Field(np.zeros((num_lat, num_lev)), domain=atm)}
    da = _monthly_dataarray(values_per_month)
    return AtmosphericData(
        name='atm', state=state, timestep=86400.0,
        param_configs={'field': {'data': da, 'method': 'linear',
                                 'grid_type': 'centers*centers'}},
        t_0=datetime(2024, 1, 1), time_type=time_type,
    )


class TestAtmosphericData:
    def test_construction(self):
        proc = make_atm_data([5.0] * 12)
        assert 'field' in proc.available_params()

    def test_grid_interpolation_shape(self):
        """Source data is regridded onto the simulation grid, dims (lat, level)."""
        proc = make_atm_data([5.0] * 12, num_lat=8, num_lev=6)
        field = proc.get_param('field')
        assert field.dims == ('lat', 'level')
        assert field.shape == (8, 6)

    def test_constant_field_interpolates_exactly(self):
        """A field uniform in month and space interpolates to that constant."""
        proc = make_atm_data([5.0] * 12)
        np.testing.assert_allclose(proc.get_param('field').values, 5.0,
                                   rtol=1e-10)

    def test_monthly_cycle_varies_in_time(self):
        """With a month-dependent field, stepping forward changes the value."""
        proc = make_atm_data(list(np.arange(1.0, 13.0)), time_type=1)
        v0 = float(proc.get_param('field').mean())
        for _ in range(60):  # advance ~2 months
            proc.step_forward()
        v1 = float(proc.get_param('field').mean())
        assert not np.isclose(v0, v1)
        assert 1.0 <= v1 <= 12.0

    def test_constant_time_type_uses_time_mean(self):
        """time_type=0 returns the mean over the month axis."""
        proc = make_atm_data(list(np.arange(1.0, 13.0)), time_type=0)
        np.testing.assert_allclose(proc.get_param('field').values,
                                   np.mean(np.arange(1.0, 13.0)), rtol=1e-10)

    def test_compute_advances_time(self):
        """_compute advances current_time by the timestep and yields no tendencies."""
        proc = make_atm_data([5.0] * 12)
        t0 = proc.get_current_time()
        tend = proc._compute()
        assert tend == {}
        assert proc.get_current_time() > t0

    def test_remove_param(self):
        proc = make_atm_data([5.0] * 12)
        proc.remove_param('field')
        assert 'field' not in proc.available_params()
