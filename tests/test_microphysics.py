r"""Physical-invariant tests for the aerosol microphysics subpackage.

Covers the Brownian coagulation kernel, the Coagulation process, and the
sedimentation velocity. The tests assert physical invariants — mass
conservation, kernel symmetry, monotonicity — and spot-check the helper
functions against textbook values. They run against stock upstream climlab.
"""
import numpy as np
import pytest
from climlab.domain.domain import zonal_mean_column
from climlab.domain.field import Field

from climlab_stardust_extension.microphysics.coagulation import (
    air_viscosity_sutherland,
    cunningham_slip,
    particle_mass,
    K_brownian,
    Coagulation,
)
from climlab_stardust_extension.microphysics.sedimentation import (
    sedimentation_velocity,
)


def make_coagulation(nbins=4, num_lat=8, num_lev=6, temperature=230.0):
    """Build a small Coagulation process for testing.

    Uses geometric ('cores') bins of 1, 2, 4, ... monomers and a uniform
    non-zero aerosol field in every bin.
    """
    _, atm = zonal_mean_column(num_lat=num_lat, num_lev=num_lev)
    shape = (num_lat, num_lev)
    cores = {f'bin_{i}': 2 ** (i - 1) for i in range(1, nbins + 1)}
    state = {name: Field(np.full(shape, 1e-9), domain=atm) for name in cores}
    T = np.full(shape, temperature)
    return Coagulation(
        name='coagulation', state=state, timestep=86400.0,
        d0=0.3e-6, cores=cores, temperature=T,
        rho_p=2200.0, Df=1.6, kf=1.0,
    )


class TestSedimentationVelocity:
    """Sedimentation velocity of an aerosol aggregate in still stratospheric air."""

    D0 = 0.3e-6        # monomer diameter [m]
    T = 220.0          # K
    RHO_AIR = 0.079    # kg/m^3 at ~50 hPa, 220 K
    RHO_P = 2200.0     # kg/m^3, silica

    def test_finite_and_downward(self):
        """A settling particle yields a finite, downward (negative) result."""
        v = sedimentation_velocity(self.D0, 8, self.RHO_P, self.T, self.RHO_AIR)
        assert np.isfinite(v)
        assert v < 0.0

    def test_monotonic_in_core_number(self):
        """A larger aggregate (more monomers) settles faster."""
        small = sedimentation_velocity(self.D0, 1, self.RHO_P, self.T,
                                       self.RHO_AIR)
        large = sedimentation_velocity(self.D0, 100, self.RHO_P, self.T,
                                       self.RHO_AIR)
        assert abs(large) > abs(small)

    def test_monotonic_in_density(self):
        """A denser particle settles faster."""
        light = sedimentation_velocity(self.D0, 8, 1000.0, self.T,
                                       self.RHO_AIR)
        heavy = sedimentation_velocity(self.D0, 8, 4000.0, self.T,
                                       self.RHO_AIR)
        assert abs(heavy) > abs(light)

    def test_fractal_aggregate_branch(self):
        """The low-fractal-dimension (Df < 2) branch yields a finite result."""
        v = sedimentation_velocity(self.D0, 16, self.RHO_P, self.T,
                                   self.RHO_AIR, Df=1.6, kf=1.0)
        assert np.isfinite(v)
        assert v < 0.0


class TestBrownianKernelHelpers:
    """Spot-checks of the kernel helper functions against textbook values."""

    def test_air_viscosity_at_freezing(self):
        """Sutherland viscosity of dry air at 273.15 K is ~1.72e-5 Pa s."""
        assert air_viscosity_sutherland(273.15) == pytest.approx(1.72e-5,
                                                                 rel=0.02)

    def test_air_viscosity_increases_with_temperature(self):
        assert air_viscosity_sutherland(300.0) > air_viscosity_sutherland(220.0)

    def test_cunningham_slip_large_particle(self):
        """Slip correction approaches 1 for particles >> mean free path."""
        assert cunningham_slip(10e-6, 273.0, 1e5) == pytest.approx(1.0, abs=0.1)

    def test_cunningham_slip_small_particle(self):
        """Slip correction is large for particles << mean free path."""
        cc_small = cunningham_slip(10e-9, 273.0, 1e5)
        cc_large = cunningham_slip(10e-6, 273.0, 1e5)
        assert cc_small > 5.0
        assert cc_small > cc_large

    def test_particle_mass(self):
        """particle_mass returns the mass of a sphere: 4/3 pi r^3 rho."""
        r, rho = 1e-6, 2000.0
        assert particle_mass(r, rho) == pytest.approx(4.0 / 3.0 * np.pi
                                                      * r ** 3 * rho)


class TestKBrownian:
    """The Brownian coagulation kernel."""

    T, P = 230.0, 5000.0

    @staticmethod
    def _mass(d):
        return particle_mass(d / 2.0, 2200.0)

    def test_positive(self):
        d1, d2 = 0.2e-6, 0.5e-6
        K = K_brownian(d1, d2, self._mass(d1), self._mass(d2), self.T, self.P)
        assert K > 0.0

    def test_symmetric(self):
        """The kernel is invariant under exchange of the two particles."""
        d1, d2 = 0.15e-6, 0.6e-6
        m1, m2 = self._mass(d1), self._mass(d2)
        K_ij = K_brownian(d1, d2, m1, m2, self.T, self.P)
        K_ji = K_brownian(d2, d1, m2, m1, self.T, self.P)
        assert K_ij == pytest.approx(K_ji, rel=1e-12)


class TestCoagulationProcess:
    """The Coagulation climlab process."""

    def test_instantiation(self):
        proc = make_coagulation()
        assert proc.nbins == 4
        assert proc.Kcoag.shape == (4, 4, 8, 6)

    def test_kernel_matrix_symmetric(self):
        """The bin-bin kernel matrix is symmetric in the bin indices."""
        proc = make_coagulation()
        np.testing.assert_allclose(proc.Kcoag,
                                   np.swapaxes(proc.Kcoag, 0, 1), rtol=1e-12)

    def test_tendency_shapes(self):
        proc = make_coagulation(nbins=4, num_lat=8, num_lev=6)
        tend = proc._compute()
        assert set(tend) == set(proc.state)
        for value in tend.values():
            assert np.asarray(value).shape == (8, 6)

    def test_conserves_total_aerosol(self):
        """Coagulation redistributes mass between bins but conserves the sum.

        The per-bin tendencies must sum to zero in every grid cell.
        """
        proc = make_coagulation()
        tend = proc._compute()
        total = sum(np.asarray(v) for v in tend.values())
        per_bin = max(np.max(np.abs(np.asarray(v))) for v in tend.values())
        assert per_bin > 0.0  # coagulation actually did something
        assert np.max(np.abs(total)) < 1e-10 * per_bin

    def test_smallest_bin_only_loses(self):
        """The single-monomer bin can only coagulate upward, never gain."""
        proc = make_coagulation()
        tend = proc._compute()
        assert np.all(np.asarray(tend['bin_1']) <= 0.0)
