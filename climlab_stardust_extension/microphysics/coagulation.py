r"""Brownian coagulation of aerosol particles.

Provides the Brownian coagulation kernel (Fuchs-form harmonic mean of the
continuum and free-molecular limits, with Cunningham slip correction) and
the ``Coagulation`` climlab process, which redistributes coagulated mass
across a set of fractal-aggregate "cores" size bins.
"""
import numpy as np
from climlab import constants as const
from climlab.process.time_dependent_process import TimeDependentProcess


# Dynamic viscosity of air via Sutherland's formula (good ~200-400 K)
def air_viscosity_sutherland(T):
    """
    Dynamic viscosity mu(T) of dry air [Pa s] using Sutherland's formula.
    T: array-like [K]
    """
    T = np.asarray(T, dtype=float)
    return 1.458e-6 * T**1.5 / (T + 110.4)


def air_mean_free_path(T, P):
    """
    Calculate mean free path of air molecules at temperature T (K) and pressure P (Pa).
    Uses hard-sphere model for air molecules with effective diameter ~0.37 nm.
    """
    d_g = 3.7e-10  # effective diameter of air molecule [m]
    # number density of air (ideal gas law)
    n_air = P / (const.kBoltzmann * T)  # [m^-3]
    # mean free path = 1/(sqrt(2) * pi * d_g^2 * n_air)
    return 1.0 / (np.sqrt(2.0) * np.pi * (d_g**2) * n_air)


def particle_mean_free_path(c_p, D_p):
    """
    Mean free path of the particle.
    D_p particle effective diameter
    c_p particle thermal speed
    """
    lam_p = 8 * D_p / (np.pi * c_p)

    return lam_p


def cunningham_slip(d_p, T, P):
    """
    Cunningham slip correction Cc for a particle of diameter d_p [m],
    with air mean free path lam [m]. Uses Kn = 2*lam/d_p form.
    """
    lam_air = air_mean_free_path(T, P)
    Kn = 2.0 * lam_air / d_p
    return 1.0 + Kn * (1.257 + 0.4 * np.exp(-1.1/np.maximum(Kn, 1e-30)))


def particle_mass(r, rho):
    """Mass of a spherical particle [kg], radius r [m], density rho [kg/m^3]."""
    return (4.0/3.0) * np.pi * r**3 * rho


def particle_thermal_speed(T, m_p):
    """Mean thermal speed of a Brownian particle [m/s]: cbar = sqrt(8 kB T / (pi m))."""
    return np.sqrt(8.0 * const.kBoltzmann * T / (np.pi * m_p))


def Rg(Ro, N, kf=1.9, Df=2.0):
    # kf = 1.9  # best for small N - verified against the DDA configurations run up to N=5
    return Ro * (N / kf)**(1/Df)


def particle_diffusion_coeff(T, mu, d_p, P):
    """
    Particle diffusion coefficient D [m^2/s] with slip:
    D = k_B T * Cc / (3 pi mu d_p)
    """
    Cc = cunningham_slip(d_p, T, P)
    return const.kBoltzmann * T * Cc / (3.0 * np.pi * mu * d_p)


def particle_effective_free_path(d_p, lam_p):
    r"""
    delta_i length [m] as per Appendix A1.4:
    delta = [ (d+lam)^3 - (d^2 + lam^2)^(3/2) ] / (3 d lam) - d
    """
    # the factor sqrt(2) in Seinfeld is not present in Weisenstein and also probably
    # not in the plots that Seinfeld itself presents:
    # delta_p = ((d_p + lam_p)**3 - (d_p**2 + lam_p**2)**(3/2)) * np.sqrt(2) / (3 * d_p * lam_p) - d_p
    delta_p = ((d_p + lam_p)**3 - (d_p**2 + lam_p**2)**(3/2)) / (3 * d_p * lam_p) - d_p

    return delta_p


# ---------- Fuchs (harmonic mean) kernel with Cunningham slip ----------
def K_brownian(d_p_i, d_p_j, m_i, m_j, T, P):
    """
    Brownian coagulation kernel [m^3/s] using harmonic mean of continuum and
    free-molecular limits, with slip-corrected diffusion.
    Inputs may be scalars or arrays; broadcasting is supported across T and p.
    """
    T = np.asarray(T, dtype=float); P = np.asarray(P, dtype=float)
    mu = air_viscosity_sutherland(T)

    # Particle properties
    c_i = particle_thermal_speed(T, m_i)
    c_j = particle_thermal_speed(T, m_j)

    D_i = particle_diffusion_coeff(T, mu, d_p_i, P)
    D_j = particle_diffusion_coeff(T, mu, d_p_j, P)

    lam_p_i = particle_mean_free_path(c_i, D_i)
    lam_p_j = particle_mean_free_path(c_j, D_j)

    delta_i = particle_effective_free_path(d_p_i, lam_p_i)
    delta_j = particle_effective_free_path(d_p_j, lam_p_j)

    # Continuum limit (diffusion-limited)
    K_cont = 2.0 * np.pi * (D_i + D_j) * (d_p_i + d_p_j)  # [m^3/s]

    term1 = (d_p_i + d_p_j) / (d_p_i + d_p_j + 2 * np.sqrt(delta_i**2 + delta_j**2))
    term2 = 8 * (D_i + D_j) / ((d_p_i + d_p_j) * np.sqrt(c_i**2 + c_j**2))
    beta = (term1 + term2)**(-1)

    K = K_cont * beta

    return K


class Coagulation(TimeDependentProcess):
    """Brownian coagulation across fractal-aggregate 'cores' size bins.

    Each state variable is one size bin; coagulation events transfer mass
    between bins and conserve the total. The per-bin tendencies returned by
    ``_compute`` therefore sum to zero in every grid cell.
    """

    def __init__(self, d0=1e-6, cores=None, temperature=None, rho_p=1000.0,
                 Df=2.0, kf=1.9, diagnostic_name_suffix="", **kwargs):
        super(Coagulation, self).__init__(**kwargs)
        for dom in list(self.domains.values()):
            self._phibounds = np.deg2rad(dom.axes['lat'].bounds)
            self._latbounds = np.sin(self._phibounds) * const.a
            self._dlatbounds = np.diff(self._latbounds)
            self._levbounds = dom.axes['lev'].bounds * 1e2
            self._dlevbounds = np.diff(self._levbounds)
            self._latpoints = 0.5*(self._latbounds[1:] + self._latbounds[:-1])
            self._levpoints = 0.5*(self._levbounds[1:] + self._levbounds[:-1])
            self.nlat = len(self._latpoints)
            self.nlev = len(self._levpoints)
            break

        self.temperature = temperature
        self.rho_p = rho_p
        self.Df = Df
        self.kf = kf
        self.d0 = d0
        self.nbins = len(self.state.keys())
        self._nbin_to_name = {k: key for k, key in enumerate(self.state)}
        self.cores = cores
        self.d_gyration = {key: 2 * Rg(self.d0*0.5, value, self.kf, self.Df)
                           for key, value in self.cores.items()}
        self.cores_vec = np.array([val for n, val in enumerate(self.cores.values())])
        self.sizes_vec = self.d0 * self.cores_vec**(1/3)
        core_mass = self.d0**3.0 * np.pi / 6.0 * self.rho_p
        self.masses_vec = self.cores_vec * core_mass

        self.Kcoag = self.K_matrix(K_brownian)
        self.rho_air = self._levpoints[None, :] / const.Rd / self.temperature
        self.Kcoag *= self.rho_air
        self.bins_trajectories = self.bins_trajectories_matrix()

    def _compute(self):

        dx_dt = self.compute_dx_dt()

        tendencies = {}
        for i in range(self.nbins):
            tendencies[self._nbin_to_name[i]] = dx_dt[i, :, :]

        return tendencies

    def K_matrix(self, K_brownian):
        K = np.zeros((self.nbins, self.nbins, self.nlat, self.nlev), dtype=float)
        for i in range(self.nbins):
            for j in range(i, self.nbins):
                K_ij = K_brownian(self.d_gyration[self._nbin_to_name[i]],
                                  self.d_gyration[self._nbin_to_name[j]],
                                  self.masses_vec[i], self.masses_vec[j],
                                  self.temperature, self._levpoints[None, :])
                K[i, j, :, :] = K_ij
                K[j, i, :, :] = K_ij
        return K

    def bins_trajectories_matrix(self):
        coagulated_cores = self.cores_vec[:, None] + self.cores_vec[None, :]
        bin_indices = np.interp(coagulated_cores, self.cores_vec,
                                np.arange(len(self.cores_vec)))
        return bin_indices

    def compute_dx_dt(self):

        dx_dt = np.zeros((self.nbins, self.nlat, self.nlev))

        for i in range(self.nbins):
            for j in range(i, self.nbins):
                prefac = 0.5 if i == j else 1.0
                events = prefac * self.Kcoag[i, j, :, :] \
                    * self.state[self._nbin_to_name[i]] \
                    * self.state[self._nbin_to_name[j]]
                core_mass = self.d0**3.0 * np.pi / 6.0 * self.rho_p
                dmi = events / (self.cores_vec[j] * core_mass)
                dmj = events / (self.cores_vec[i] * core_mass)
                dx_dt[i, :, :] -= dmi
                dx_dt[j, :, :] -= dmj
                target_bin = int(self.bins_trajectories[i, j])
                f = self.bins_trajectories[i, j] - target_bin
                if target_bin == (self.nbins-1) or f < 1e-3:
                    dx_dt[target_bin, :, :] += (dmi + dmj)
                else:
                    dx_dt[target_bin, :, :] += (1-f) * (dmi + dmj)
                    dx_dt[target_bin+1, :, :] += f * (dmi + dmj)

        return dx_dt
