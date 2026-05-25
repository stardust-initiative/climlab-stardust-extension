"""Aerosol optical depth tables for RRTMG radiation.

Loads pre-computed Mie-theory optical-property tables from a remote
repository (GitHub), interpolates to the requested particle radius,
and computes broadband / spectrally-resolved optical depths, single
scattering albedos and asymmetry parameters suitable for RRTMG.

Requires a project configuration file (``config.json``) that specifies
the GitHub repository, token, and material paths.
"""

import numpy as np
import xarray as xr
from collections import namedtuple
from typing import Union

from climlab.process.time_dependent_process import TimeDependentProcess
from climlab import constants as const

from climlab_stardust_extension.radiation import load_config
from climlab_stardust_extension.radiation.insolation import daily_avg_of_x
from climlab_stardust_extension.utils.file_handling import load_repo_table
from climlab_stardust_extension.utils.constants import n_avogadro


aerosol_instance = namedtuple(
    'aerosol_instance', ['material_name', 'r_m', 'vmr']
)


# Optical materials served by the stardust_2d_inputs input-data engine, keyed by
# lower-cased material name -> engine registry key. Materials absent here (e.g.
# mars black, alpha-SiC) fall back to the legacy GitHub/pooch path until the
# private database holds them.
_OPTICAL_REGISTRY_KEY_MAP = {
    'silica': 'optical_silica',
    'sulfate': 'optical_sulfate',
    'calcite average ray': 'optical_calcite',
}


class AerosolsOptDepTables(TimeDependentProcess):
    """Process that computes aerosol optical depths from tabulated Mie properties.

    Parameters
    ----------
    domain : climlab Domain
        The atmospheric domain (must have a ``lev`` axis).
    aerosol_instance_list : list of aerosol_instance
        Each entry specifies a material name, particle radius, and VMR profile.
    coszen : ndarray
        Cosine of the solar zenith angle.
    aerosols_opt_tables_http : str, optional
        Base URL for optical-property tables.
    aerosols_tables_dict : dict, optional
        Mapping from material name to table metadata.
    aerosols_token : str, optional
        Authentication token for table downloads.
    proj_name : str, optional
        Project name for cache directory.
    qsca_fac_limit : float, optional
        Scattering-efficiency safety limit (default 5.0).
    do_gray : bool, optional
        If True, average spectral properties to gray (default False).
    days : ndarray, optional
        Day-of-year array for daily averaging (default 0..364).
    model_state : dict or state, optional
        If provided, ``get_radiation_with_aerosols_params`` is called at init.
    **kwargs
        Passed to ``TimeDependentProcess.__init__``.
    """

    def __init__(self, domain, aerosol_instance_list, coszen, **kwargs):
        # Load configuration defaults
        cfg = load_config(kwargs.pop('config_path', 'config.json'))
        if cfg is None:
            cfg = {
                'aerosols_opt_tables_http': '',
                'aerosols_tables_dict': {},
                'aerosols_token': '',
                'proj_name': '',
            }
        aerosols_opt_tables_http = kwargs.pop(
            'aerosols_opt_tables_http', cfg['aerosols_opt_tables_http']
        )
        aerosols_tables_dict = kwargs.pop(
            'aerosols_tables_dict', cfg['aerosols_tables_dict']
        )
        aerosols_token = kwargs.pop('aerosols_token', cfg['aerosols_token'])
        proj_name = kwargs.pop('proj_name', cfg['proj_name'])
        qsca_fac_limit = kwargs.pop('qsca_fac_limit', 5.0)
        do_gray = kwargs.pop('do_gray', False)

        super(AerosolsOptDepTables, self).__init__(**kwargs)
        self.name = 'AerosolsOptDepTables'
        self.mg_air = (
            1e-3 * const.molecular_weight['dry air'] / n_avogadro * const.g
        )
        self.aerosol_instance_list = aerosol_instance_list
        self.qsca_fac_limit = qsca_fac_limit
        self.coszen = np.squeeze(coszen)
        self.domain = domain

        self.keys_list = [
            (mat.material_name, mat.r_m) for mat in self.aerosol_instance_list
        ]
        self.n_mat = len(self.aerosol_instance_list)

        # Dictionaries for tabulated optical properties
        self.babs_lw_dict = {}
        self.bext_sw_dict = {}
        self.bsca_sw_dict = {}
        self.basc_sw_dict = {}
        self.mu_dict = {}
        self.alpha_dict = {}
        self.gamma_dict = {}
        self.bet_mu_bsca_sw_dict = {}
        self.bet_bar_bsca_sw_dict = {}

        mat_list = list(aerosols_tables_dict.keys())
        mat_list_lc = [k.lower() for k in mat_list]
        for k in self.keys_list:
            material_name_lc = k[0].lower()
            assert material_name_lc in mat_list_lc, (
                f"Unknown material used={material_name_lc}"
            )
            material_name = mat_list[mat_list_lc.index(material_name_lc)]
            r_m = k[1]
            # Public materials are served by the stardust_2d_inputs engine
            # (content-addressed store, transport-paper release pin,
            # hash-verified). Other materials use the legacy GitHub/pooch path
            # until the private database holds them. The engine config is found
            # via STARDUST_2D_INPUTS_CONFIG (see the engine's config.example.json).
            registry_key = _OPTICAL_REGISTRY_KEY_MAP.get(material_name_lc)
            if registry_key is not None:
                from stardust_2d_inputs.core.loader import load as _engine_load
                ds = _engine_load(registry_key)
            else:
                local_file, _ = load_repo_table(
                    aerosols_opt_tables_http,
                    aerosols_tables_dict[material_name]['mat_path'],
                    aerosols_token,
                    proj_name=proj_name,
                )
                ds = xr.open_dataset(local_file)
            mu = np.log(r_m)
            i_mu, fac_mu = self.get_interp_param(mu, ds.mu_samples.values)
            babs_lw_full = ds.babs_lw.values
            bext_sw_full = ds.bext_sw.values
            bsca_sw_full = ds.bsca_sw.values
            basc_sw_full = ds.basc_sw.values
            if do_gray:
                babs_lw_full = self.to_gray(babs_lw_full)
                bext_sw_full = self.to_gray(bext_sw_full)
                bsca_sw_full = self.to_gray(bsca_sw_full)
                basc_sw_full = self.to_gray(basc_sw_full)

            if 'coszen' in ds.keys():
                self.mu_dict[k] = ds.coszen.values
                self.has_mu = True
                if do_gray:
                    bsca_sw_full = self.to_gray(bsca_sw_full)
                    basc_sw_full = self.to_gray(basc_sw_full)
                    bext_sw_full = self.to_gray(bext_sw_full)
            else:
                self.has_mu = False

            if fac_mu < 1.0:
                self.babs_lw_dict[k] = np.squeeze(
                    babs_lw_full[i_mu, :] ** fac_mu
                    * babs_lw_full[i_mu + 1, :] ** (1.0 - fac_mu)
                )
                self.bext_sw_dict[k] = np.squeeze(
                    bext_sw_full[i_mu, ...] ** fac_mu
                    * bext_sw_full[i_mu + 1, ...] ** (1.0 - fac_mu)
                )
                self.bsca_sw_dict[k] = np.squeeze(
                    bsca_sw_full[i_mu, ...] ** fac_mu
                    * bsca_sw_full[i_mu + 1, ...] ** (1.0 - fac_mu)
                )
                self.basc_sw_dict[k] = np.squeeze(
                    basc_sw_full[i_mu, ...] ** fac_mu
                    * basc_sw_full[i_mu + 1, ...] ** (1.0 - fac_mu)
                )
                self.bet_mu_bsca_sw_dict[k] = np.squeeze(
                    ds.bet_mu_bsca_sw.values[i_mu, ...] ** fac_mu
                    * ds.bet_mu_bsca_sw.values[i_mu + 1, ...] ** (1.0 - fac_mu)
                )
                self.bet_bar_bsca_sw_dict[k] = np.squeeze(
                    ds.bet_bar_bsca_sw.values[i_mu, :] ** fac_mu
                    * ds.bet_bar_bsca_sw.values[i_mu + 1, :] ** (1.0 - fac_mu)
                )
            else:
                self.babs_lw_dict[k] = np.squeeze(babs_lw_full[i_mu, :])
                self.bext_sw_dict[k] = np.squeeze(bext_sw_full[i_mu, ...])
                self.bsca_sw_dict[k] = np.squeeze(bsca_sw_full[i_mu, ...])
                self.basc_sw_dict[k] = np.squeeze(basc_sw_full[i_mu, ...])
                self.bet_mu_bsca_sw_dict[k] = np.squeeze(
                    ds.bet_mu_bsca_sw.values[i_mu, ...]
                )
                self.bet_bar_bsca_sw_dict[k] = np.squeeze(
                    ds.bet_bar_bsca_sw.values[i_mu, :]
                )

        vmr_dict = {
            (mat.material_name, mat.r_m): mat.vmr
            for mat in self.aerosol_instance_list
        }
        self.vmr_dict = vmr_dict
        self.days = kwargs.get('days', np.arange(365))
        if 'model_state' in kwargs:
            self.model_state = kwargs['model_state']
            self.rad_with_aero_param_dict = get_radiation_with_aerosols_params(
                self.model_state, self, coszen, days=self.days,
            )

    @property
    def vmr_dict(self):
        return self._vmr_dict

    @vmr_dict.setter
    def vmr_dict(self, value):
        self._vmr_dict = value
        self.param['vmr_dict'] = value
        self._compute_optical_depths()

    def _compute_optical_depths(self):
        eps = 1e-10
        dp = 1e2 * np.diff(self.domain.axes['lev'].bounds)
        count = 0
        domain_shape = self.domain.shape
        ls = len(domain_shape)
        has_lat = ls == 2
        if has_lat:
            nlat, nlev = domain_shape
        else:
            assert ls == 1, 'Tatm has an unsupported structure'
            nlev = domain_shape[0]

        self.tauaer_sw_dict = {}
        self.ssaaer_sw_dict = {}
        self.asmaer_sw_dict = {}
        for k in self.keys_list:
            vmr = self._vmr_dict[k]
            babs_lw = self.babs_lw_dict[k]
            bext_sw = self.bext_sw_dict[k]
            bsca_sw = self.bsca_sw_dict[k]
            basc_sw = self.basc_sw_dict[k]
            nlw, nsw = babs_lw.shape[0], bext_sw.shape[0]
            if has_lat:
                babs_lw = babs_lw[:, np.newaxis, np.newaxis]
                bext_sw = bext_sw[:, np.newaxis, np.newaxis]
                bsca_sw = bsca_sw[:, np.newaxis, np.newaxis]
                basc_sw = basc_sw[:, np.newaxis, np.newaxis]
            else:
                babs_lw = babs_lw[:, np.newaxis]
                bext_sw = bext_sw[:, np.newaxis]
                bsca_sw = bsca_sw[:, np.newaxis]
                basc_sw = basc_sw[:, np.newaxis]

            if count == 0:
                nlw0, nsw0 = babs_lw.shape[0], bext_sw.shape[0]
                if has_lat:
                    tauaer_lw = np.zeros((nlw, nlat, nlev))
                    tauaer_sw = np.zeros((nsw, nlat, nlev))
                    ssaaer_sw_temp = np.zeros((nsw, nlat, nlev))
                    asmaer_sw_temp = np.zeros((nsw, nlat, nlev))
                else:
                    tauaer_lw = np.zeros((nlw, nlev))
                    tauaer_sw = np.zeros((nsw, nlev))
                    ssaaer_sw_temp = np.zeros((nsw, nlev))
                    asmaer_sw_temp = np.zeros((nsw, nlev))

            assert nlw == nlw0, "not all added aerosols have the same nlw"
            assert nsw == nsw0, "not all added aerosols have the same nsw"

            if has_lat:
                dn_da_particles = vmr * (dp / self.mg_air)[np.newaxis, np.newaxis, :]
            else:
                dn_da_particles = vmr * (dp / self.mg_air)[np.newaxis, :]

            dtau_lw = dn_da_particles * babs_lw
            dtau_sw = dn_da_particles * bext_sw
            dsca_sw = dn_da_particles * bsca_sw
            dasm_sw = dn_da_particles * basc_sw

            tauaer_lw += dtau_lw
            tauaer_sw += dtau_sw
            ssaaer_sw_temp += dsca_sw
            asmaer_sw_temp += dasm_sw
            self.tauaer_sw_dict[k] = dtau_sw
            self.ssaaer_sw_dict[k] = dsca_sw / (dtau_sw + eps)
            self.asmaer_sw_dict[k] = dasm_sw / (dsca_sw + eps)

            count += 1

        ssaaer_sw = ssaaer_sw_temp / (tauaer_sw + eps)
        asmaer_sw = asmaer_sw_temp / (ssaaer_sw_temp + eps)
        self.tauaer_lw = tauaer_lw
        self.tauaer_sw = tauaer_sw
        self.ssaaer_sw = ssaaer_sw
        self.asmaer_sw = asmaer_sw
        if hasattr(self, 'model_state'):
            rad_with_aero_param_dict = get_radiation_with_aerosols_params(
                self.model_state, self, self.coszen, days=self.days,
            )
            for key, val in rad_with_aero_param_dict.items():
                if isinstance(val, np.ndarray):
                    self.rad_with_aero_param_dict[key][:] = val[:]

    @staticmethod
    def to_gray(spectral_mat):
        s = spectral_mat.shape
        ax = len(s) - 1
        n = s[-1]
        return np.repeat(
            np.mean(spectral_mat, axis=ax)[..., np.newaxis], n, axis=ax
        )

    @staticmethod
    def get_interp_param(x0, x_vect):
        ix = np.argmin(np.abs(x_vect - x0))
        nx = len(x_vect)
        fac_was_set = False
        if ix == 0 and x_vect[ix] >= x0:
            fac = 1.0
            fac_was_set = True
        elif ix == nx - 1:
            if x_vect[ix] <= x0:
                fac = 0.0
                fac_was_set = True
            ix = ix - 1
        if not fac_was_set:
            fac = (x_vect[ix + 1] - x0) / (x_vect[ix + 1] - x_vect[ix])
        fac = min([1.0, max([0.0, fac])])
        return ix, fac

    @staticmethod
    def opt_path_instance_name(material_name, r_m):
        return f'OpticPath_{material_name}_{int(r_m * 1e9)}'

    def update_vmr(self, aerosol_instance_list):
        """Update VMR profiles from a new list of aerosol instances."""
        vmr_dict = {
            (mat.material_name, mat.r_m): mat.vmr
            for mat in aerosol_instance_list
        }
        assert set(vmr_dict.keys()) == set(self.keys_list), (
            'update aerosols table does not match structure of original table'
        )
        self.vmr_dict = vmr_dict

    def _compute(self):
        tendencies = {}
        self._compute_optical_depths()
        return tendencies


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def construct_uni_layer_vmr(rho_particle, M, h_layer, w_layer, r_m, state):
    """Construct a uniform-layer VMR profile at height *h_layer*."""
    has_lat = 'lat' in state.Tatm.domain.axes.keys()
    nlat = len(state.Tatm.domain.lat.points) if has_lat else 1
    lev = state.Tatm.domain.lev.points
    nlev = len(lev)
    Se = 4 * np.pi * const.a ** 2
    m = 4 * np.pi / 3 * r_m ** 3 * rho_particle
    dn_dv_particles = M / m / Se / w_layer

    m_air = 1e-3 * const.molecular_weight['dry air'] / n_avogadro
    lev_bounds = state.Tatm.domain.axes['lev'].bounds
    dp = 1e2 * np.diff(lev_bounds)[np.newaxis, :]
    p = 1e2 * lev[np.newaxis, :]

    dn_da_air = dp / (m_air * const.g)
    dz = dn_da_air * const.kBoltzmann * state.Tatm / p
    z = np.fliplr(np.concatenate(
        (np.zeros((nlat, 1)), np.cumsum(np.fliplr(dz), axis=1)), axis=1
    ))

    zmin, zmax = h_layer - w_layer / 2, h_layer + w_layer / 2
    dz_cell_mat = np.zeros((nlat, nlev))
    for ilat in range(nlat):
        for ilev in range(nlev):
            z2 = z[ilat, ilev]
            z1 = z[ilat, ilev + 1]
            if z2 <= zmax and z1 >= zmin:
                dz_cell = z2 - z1
            elif z2 <= zmax and z2 >= zmin:
                dz_cell = z2 - zmin
            elif z1 >= zmin and z1 <= zmax:
                dz_cell = zmax - z1
            else:
                dz_cell = 0.0
            dz_cell_mat[ilat, ilev] = dz_cell
    vmr = dn_dv_particles * dz_cell_mat / dn_da_air
    if not has_lat:
        vmr = vmr[0, :]
    return vmr


def construct_uni_layer_vmr_p_based(
    rho_particle,
    density_profile_kg_m2: Union[float, np.ndarray],
    p_min, p_max, r_m, state,
):
    """Construct a uniform-layer VMR profile between *p_min* and *p_max* (hPa)."""
    has_lat = 'lat' in state.Tatm.domain.axes.keys()
    nlat = len(state.Tatm.domain.lat.points) if has_lat else 1
    lev = state.Tatm.domain.lev.points
    m = 4.0 * np.pi / 3.0 * r_m ** 3 * rho_particle

    if type(density_profile_kg_m2) == np.dtype('float'):
        density_profile_kg_m2 = density_profile_kg_m2 * np.ones(nlat)

    dn_da_dp_particles = density_profile_kg_m2 / m / (p_max - p_min)

    m_air = 1e-3 * const.molecular_weight['dry air'] / n_avogadro
    lev_bounds = state.Tatm.domain.axes['lev'].bounds
    dp = np.diff(lev_bounds)

    dp_cell_vect = np.zeros_like(lev)
    ind2 = np.where(lev_bounds <= p_max)[0]
    ind1 = np.where(lev_bounds >= p_min)[0]
    i1 = ind1[0]
    i2 = ind2[-1]
    ind = np.arange(i1, i2)
    if len(ind) > 0:
        dp_cell_vect[ind] = lev_bounds[ind + 1] - lev_bounds[ind]
    if i2 == i1 - 1:
        dp_cell_vect[i2] = p_max - p_min
    else:
        dp_cell_vect[i2] = p_max - lev_bounds[i2]
        dp_cell_vect[i1 - 1] = lev_bounds[i1] - p_min

    dn_da_air = 1e2 * dp / (m_air * const.g)
    vmr = (dn_da_dp_particles[:, np.newaxis] * dp_cell_vect) / dn_da_air[np.newaxis, :]
    return vmr


def get_radiation_with_aerosols_params(
    state, aerosols_table_obj, coszen, nh=201, **kwargs
):
    """Compute RRTMG parameters for aerosol-layer treatment.

    Returns a dict with ``tauaer_lw``, ``add_aero_layer``, ``r_mu``,
    ``t_mu``, ``r_bar``, ``t_bar``.
    """
    do_avg = kwargs.get('do_avg', True)
    initialized = False
    coszen_small = kwargs.get('coszen_small', 1e-3)
    days = kwargs.get('days', np.arange(365))
    try:
        Tatm_domain = state['Tatm'].domain
    except Exception:
        Tatm_domain = state.Tatm.domain
    lat = Tatm_domain.lat.points
    for key in aerosols_table_obj.bet_mu_bsca_sw_dict.keys():
        bsca = aerosols_table_obj.bsca_sw_dict[key]
        n_sw = len(bsca)
        if not initialized:
            s = (n_sw, len(coszen), Tatm_domain.lev.num_points)
            one_minus_log_r_mu = np.zeros(s)
            t_mu = np.ones(s)
            one_minus_log_r_bar = np.zeros(s)
            t_bar = np.ones(s)
            initialized = True

        mu_vect_table = aerosols_table_obj.mu_dict[key]
        beta_mu_table = (
            aerosols_table_obj.bet_mu_bsca_sw_dict[key]
            / bsca[:, np.newaxis]
        )

        if do_avg:
            beta_mu_lat, h0 = daily_avg_of_x(
                lat, days, mu_vect_table, beta_mu_table, nh=nh,
            )
            abs_fac = np.mean(np.array(h0), axis=1) / np.pi
            coszen_daily_avg, _ = daily_avg_of_x(
                lat, days, mu_vect_table,
                mu_vect_table[np.newaxis, :], nh=nh,
            )
            factor_bar = coszen_daily_avg.T / (coszen_small + coszen)
            factor_bar = factor_bar[np.newaxis, ...]
        else:
            beta_mu_lat = np.zeros((n_sw, len(lat)))
            for ng1 in range(n_sw):
                beta_mu_lat[ng1, :] = np.interp(
                    coszen[:, 0], mu_vect_table, beta_mu_table[ng1, :]
                )
            abs_fac = np.ones_like(lat)
            factor_bar = np.ones((1, len(lat), 1))

        beta_bar = aerosols_table_obj.bet_bar_bsca_sw_dict[key] / bsca
        omega_tau = (
            aerosols_table_obj.ssaaer_sw_dict[key]
            * aerosols_table_obj.tauaer_sw_dict[key]
        )
        one_minus_omega_tau = (
            (1.0 - aerosols_table_obj.ssaaer_sw_dict[key])
            * aerosols_table_obj.tauaer_sw_dict[key]
        )

        coszen_inv = 1.0 / (coszen.ravel() + coszen_small)
        one_minus_log_r_mu1 = (
            beta_mu_lat[..., np.newaxis] * omega_tau
            * coszen_inv[np.newaxis, :, np.newaxis]
        )
        one_minus_log_a_mu1 = (
            one_minus_omega_tau
            * (abs_fac * coszen_inv)[np.newaxis, :, np.newaxis]
        )
        one_minus_log_r_mu1 = np.where(
            one_minus_log_r_mu1 <= 1.0, one_minus_log_r_mu1, 1.0
        )
        one_minus_log_a_mu1 = np.where(
            one_minus_log_a_mu1 <= 1.0, one_minus_log_a_mu1, 1.0
        )
        one_minus_log_r_bar1 = (
            factor_bar * 2.0 * beta_bar[:, np.newaxis, np.newaxis]
            * omega_tau
        )
        one_minus_log_a_bar1 = factor_bar * 2.0 * one_minus_omega_tau
        t_mu1 = np.exp(-one_minus_log_r_mu1 - one_minus_log_a_mu1)
        t_bar1 = np.exp(-one_minus_log_r_bar1 - one_minus_log_a_bar1)
        one_minus_log_r_mu += one_minus_log_r_mu1
        t_mu *= t_mu1
        one_minus_log_r_bar += one_minus_log_r_bar1
        t_bar *= t_bar1

    r_mu = 1.0 - np.exp(-one_minus_log_r_mu)
    r_bar = 1.0 - np.exp(-one_minus_log_r_bar)

    return {
        'tauaer_lw': aerosols_table_obj.tauaer_lw,
        'add_aero_layer': 1,
        'r_mu': r_mu,
        't_mu': t_mu,
        'r_bar': r_bar,
        't_bar': t_bar,
    }
