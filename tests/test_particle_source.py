r"""Tests for the aerosol ParticleSource process and its source factory.

The source factory pairs a spatial parameterization (single grid point,
gaussian, uniform, ...) with a temporal one (constant, by-month, seasonal).
ParticleSource injects the resulting source field into a transported tracer.
Tests run against stock upstream climlab.
"""
import numpy as np
import pytest
from datetime import datetime
from climlab.domain.domain import zonal_mean_column
from climlab.domain.field import Field

from climlab_stardust_extension.dynamics.source_parameterizations import (
    create_source,
    register_space_type,
    validate_source_config,
    _SourceSpaceParameterization,
    SPACE_TYPE_REGISTRY,
)
from climlab_stardust_extension.dynamics.particle_source import ParticleSource

LAT_BOUNDS = np.linspace(-90, 90, 9)
LEV_BOUNDS = np.linspace(0, 1000, 7)
JAN = lambda: datetime(2024, 1, 15)   # noqa: E731
JULY = lambda: datetime(2024, 7, 15)  # noqa: E731


class TestSourceFactory:
    def test_single_grid_point_places_rate(self):
        """A single-grid-point source deposits its full rate in one cell."""
        cfg = {'name': 's', 'space_type': 'single_grid_point',
               'time_type': 'const', 'rate': 10.0,
               'point_source': [0.0, 500.0],
               'lat_bounds': LAT_BOUNDS, 'lev_bounds': LEV_BOUNDS}
        field = create_source(cfg).compute(JAN, np.zeros((8, 6)))
        assert field.sum() == pytest.approx(10.0)
        assert np.count_nonzero(field) == 1

    def test_gaussian_normalized_to_rate(self):
        """A gaussian source integrates to its specified rate."""
        cfg = {'name': 'g', 'space_type': 'gaussian', 'time_type': 'const',
               'rate': 4.0, 'gaussian_center': [0.0, 500.0],
               'gaussian_sigma': [2.0, 2.0],
               'lat_bounds': LAT_BOUNDS, 'lev_bounds': LEV_BOUNDS}
        field = create_source(cfg).compute(JAN, np.zeros((8, 6)))
        assert field.sum() == pytest.approx(4.0)

    def test_monthly_time_gates_by_month(self):
        """A by-month source is active only during the listed months."""
        cfg = {'name': 'm', 'space_type': 'single_grid_point',
               'time_type': 'by_month', 'rate': 1.0,
               'point_source': [0.0, 500.0], 'month_list': [7, 8],
               'lat_bounds': LAT_BOUNDS, 'lev_bounds': LEV_BOUNDS}
        src = create_source(cfg)
        assert src.compute(JULY, np.zeros((8, 6))).sum() == pytest.approx(1.0)
        assert src.compute(JAN, np.zeros((8, 6))).sum() == pytest.approx(0.0)

    def test_unknown_space_type_raises(self):
        with pytest.raises(ValueError):
            create_source({'name': 'bad', 'space_type': 'nonexistent',
                           'time_type': 'const', 'rate': 1.0})

    def test_validate_catches_missing_field(self):
        ok, msg = validate_source_config(
            {'name': 'x', 'space_type': 'gaussian', 'time_type': 'const'})
        assert not ok
        assert 'rate' in msg

    def test_register_custom_space_type(self):
        class _AllOnes(_SourceSpaceParameterization):
            def compute(self, grid):
                return np.ones_like(grid) * self.params['rate']
        register_space_type('all_ones', _AllOnes)
        assert SPACE_TYPE_REGISTRY['all_ones'] is _AllOnes


class TestParticleSource:
    @staticmethod
    def _make(sources_config, **kwargs):
        _, atm = zonal_mean_column(num_lat=8, num_lev=6)
        state = {'aerosol': Field(np.zeros((8, 6)), domain=atm)}
        return ParticleSource(
            name='src', state=state, timestep=86400.0,
            current_time=JULY, sources_config=sources_config, **kwargs)

    @staticmethod
    def _point_source(rate):
        return [{'name': 's', 'space_type': 'single_grid_point',
                 'time_type': 'const', 'rate': rate,
                 'point_source': [0.0, 500.0]}]

    def test_construct_from_config(self):
        """ParticleSource accepts a parsed config list directly (no JSON file)."""
        proc = self._make(self._point_source(1.0))
        assert len(proc.sources_list) == 1

    def test_requires_a_source_specification(self):
        _, atm = zonal_mean_column(num_lat=8, num_lev=6)
        state = {'aerosol': Field(np.zeros((8, 6)), domain=atm)}
        with pytest.raises(ValueError):
            ParticleSource(name='src', state=state, timestep=86400.0,
                           current_time=JULY)

    def test_injects_mass_at_configured_rate(self):
        """Injected mass over one step equals rate * timestep."""
        proc = self._make(self._point_source(2.0))
        tend = proc._compute()
        assert set(tend) == {'aerosol'}
        assert np.sum(tend['aerosol']) > 0.0
        total = float(np.ravel(np.asarray(proc.total_tracer_source))[0])
        assert total == pytest.approx(2.0 * 86400.0)

    def test_rate_string_expression_evaluated(self):
        """A string rate is parsed as a numeric arithmetic expression."""
        proc = self._make(self._point_source('1e9 / 2'))
        total = float(np.ravel(np.asarray(proc.total_tracer_source))[0])
        proc._compute()
        total = float(np.ravel(np.asarray(proc.total_tracer_source))[0])
        assert total == pytest.approx(5e8 * 86400.0)

    def test_rate_string_rejects_code_execution(self):
        """A non-arithmetic rate string is rejected, not executed."""
        with pytest.raises(ValueError):
            self._make(self._point_source("__import__('os').getcwd()"))
