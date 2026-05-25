r"""Gravitational sedimentation velocity for aerosol particles."""
import numpy as np
from climlab import constants as const
from climlab_stardust_extension.utils import constants as ext_const


def sedimentation_velocity(d0, cores_number, rho_p, T, rho, Df=3.0, kf=1.0):
    r"""Sedimentation (gravitational settling) velocity of an aerosol aggregate.

    The particle is described as a fractal aggregate of ``cores_number``
    monomers, each of diameter ``d0``. Use ``Df=3.0`` (the default) for
    compact spherical particles.

    Parameters
    ----------
    d0 : float or ndarray
        Monomer diameter [m].
    cores_number : float or ndarray
        Number of monomers in the aggregate.
    rho_p : float or ndarray
        Monomer material mass density [kg/m^3].
    T : float or ndarray
        Air temperature [K].
    rho : float or ndarray
        Air mass density [kg/m^3].
    Df : float, optional
        Fractal dimension. The default of 3.0 describes compact spheres;
        lower values describe progressively more open aggregates
        (Weisenstein et al. 2015).
    kf : float, optional
        Fractal prefactor. With ``Df=3.0`` and ``kf=1.0`` the diameter of
        gyration reduces to the volume-equivalent sphere diameter.

    Returns
    -------
    omega_sed : float or ndarray
        Sedimentation velocity as a pressure tendency [Pa/s], negative
        (downward).

    Notes
    -----
    The Stokes-Cunningham slip correction follows Davies (Proc. Phys. Soc.,
    1945); air viscosity uses Sutherland's model. The diameter of gyration,
    particle mass and projected area are all derived from ``d0``,
    ``cores_number``, ``Df`` and ``kf`` for self-consistency.
    """
    sigma = 4e-19                                    # air-molecule cross-section [m^2]
    m = const.molecular_weight['dry air'] / ext_const.n_avogadro / 1000  # air molecule mass [kg]
    n = rho / m                                      # air number density [1/m^3]
    lamda = 1 / sigma / n / np.sqrt(2)                # mean free path [m]

    # diameter of gyration of the fractal aggregate
    Dg = d0 * (cores_number / kf) ** (1 / Df)
    rad = Dg / 2

    Kn = lamda / Dg                                  # Knudsen number
    # Stokes-Cunningham slip correction (Davies 1945)
    C = 1 + 2 * Kn * (1.257 + 0.4 * np.exp(-0.55 / 2 / Kn))
    # air viscosity from Sutherland's model [Pa s]
    mu = 1.716e-5 * (T / 273.15) ** 1.5 * ((273.15 + 110.4) / (T + 110.4))

    mass = cores_number * np.pi * d0 ** 3 / 6 * rho_p
    if Df >= 2.0:
        area2d = np.pi * rad ** 2
    else:  # open aggregate, see Weisenstein et al. (2015)
        area2d = cores_number * np.pi * (d0 / 2) ** 2

    V_sed = mass * const.g * rad * C / (6 * mu * area2d)

    return -V_sed * rho * const.g
