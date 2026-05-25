"""Extended thermodynamic functions building on climlab.utils.thermo.

Provides ERA5-compatible saturation vapor pressure formulas, temperature
derivatives of qsat, and helper utilities.
"""

import numpy as np
from climlab.utils.thermo import tempCtoK

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False


def _input_util_func(mat):
    """Detect input type and return (type_str, numpy-or-xarray lib, array)."""
    if HAS_XARRAY and isinstance(mat, xr.DataArray):
        return "xarray", xr, mat
    elif np.isscalar(mat):
        return "scalar", np, np.array([mat])
    else:
        return "ndarray", np, mat


def _tetens(T, a1, a3, a4, T0):
    """Teten's formula for saturation vapor pressure (Pa)."""
    return a1 * np.exp(a3 * (T - T0) / (T - a4))


def _d_tetens_dT(T, a1, a3, a4, T0):
    """Temperature derivative of Teten's formula (Pa/K)."""
    return a1 * np.exp(a3 * (T - T0) / (T - a4)) * (
        a3 * (T0 - a4) / (T - a4)**2
    )


eps = 0.622  # Ratio of molecular weights of water vapor to dry air


def clausius_clapeyron_extended(T, do_era5=False):
    """Compute saturation vapor pressure as function of temperature.

    Supports the standard Bolton (1980) formula used in climlab, plus an
    ERA5-compatible Teten's formula with ice/water split (IFS documentation,
    Part IV, chapter 7.2, equations 7.5-6).

    Parameters
    ----------
    T : float, ndarray, or xarray.DataArray
        Temperature in Kelvin
    do_era5 : bool
        If True, use the ERA5 Teten's formula with ice/water transition.
        If False, use climlab's standard formula.

    Returns
    -------
    es : same type as T
        Saturation vapor pressure in hPa
    """
    input_type, lib, T1 = _input_util_func(T)
    pa_to_hpa = 1e-2
    if do_era5:
        T_ice, T0 = 250.16, 273.16
        a1 = 611.21
        a3_water, a3_ice = 17.502, 22.587
        a4_water, a4_ice = 32.19, -0.7
    else:
        T_ice, T0 = 0.0, tempCtoK
        a1 = 611.2
        a3_water, a3_ice = 17.67, 17.67
        a4_water, a4_ice = 29.65, 29.65
    alpha = lib.where(
        T1 <= T_ice, 0.0,
        lib.where(T1 >= T0, 1.0, ((T1 - T_ice) / (T0 - T_ice))**2)
    )
    es_water = pa_to_hpa * _tetens(T1, a1, a3_water, a4_water, T0)
    es_ice = pa_to_hpa * _tetens(T1, a1, a3_ice, a4_ice, T0)
    es = alpha * es_water + (1.0 - alpha) * es_ice

    if input_type == "scalar":
        return float(es[0])
    elif input_type == "xarray":
        return xr.DataArray(es, coords=T.coords, dims=T.dims,
                            attrs={"units": "hPa"})
    return es


def clausius_clapeyron_T_deriv(T, do_era5=False):
    """Compute temperature derivative of saturation vapor pressure.

    Parameters
    ----------
    T : float, ndarray, or xarray.DataArray
        Temperature in Kelvin
    do_era5 : bool
        If True, use the ERA5 Teten's formula.

    Returns
    -------
    des_dT : same type as T
        d(es)/dT in hPa/K
    """
    input_type, lib, T1 = _input_util_func(T)
    pa_to_hpa = 1e-2
    if do_era5:
        T_ice, T0 = 250.16, 273.16
        a1 = 611.21
        a3_water, a3_ice = 17.502, 22.587
        a4_water, a4_ice = 32.19, -0.7
    else:
        T_ice, T0 = 0.0, tempCtoK
        a1 = 611.21
        a3_water, a3_ice = 17.67, 17.67
        a4_water, a4_ice = 29.65, 29.65
    alpha = lib.where(
        T1 <= T_ice, 0.0,
        lib.where(T1 >= T0, 1.0, ((T1 - T_ice) / (T0 - T_ice))**2)
    )
    d_alpha_dT = lib.where(
        T1 <= T_ice, 0.0,
        lib.where(T1 >= T0, 0.0, 2 * (T1 - T_ice) / (T0 - T_ice)**2)
    )
    es_water = pa_to_hpa * _tetens(T1, a1, a3_water, a4_water, T0)
    des_water_dT = pa_to_hpa * _d_tetens_dT(T1, a1, a3_water, a4_water, T0)
    es_ice = pa_to_hpa * _tetens(T1, a1, a3_ice, a4_ice, T0)
    des_ice_dT = pa_to_hpa * _d_tetens_dT(T1, a1, a3_ice, a4_ice, T0)
    des_dT = (d_alpha_dT * es_water - d_alpha_dT * es_ice
              + alpha * des_water_dT + (1.0 - alpha) * des_ice_dT)

    if input_type == "scalar":
        return float(des_dT[0])
    elif input_type == "xarray":
        return xr.DataArray(des_dT, coords=T.coords, dims=T.dims,
                            attrs={"units": "hPa"})
    return des_dT


def qsat_extended(T, p, do_era5=False, small=0.0, do_simplified=False):
    """Compute saturation specific humidity.

    Parameters
    ----------
    T : float, ndarray, or xarray.DataArray
        Temperature in Kelvin
    p : float or ndarray
        Pressure in hPa
    do_era5 : bool
        Use ERA5-compatible formula
    small : float
        Regularization parameter for numerical stability
    do_simplified : bool
        Use simplified denominator formula

    Returns
    -------
    qs : same type as T
        Saturation specific humidity (dimensionless)
    """
    input_type, _, T1 = _input_util_func(T)
    es = clausius_clapeyron_extended(T, do_era5=do_era5)
    if do_simplified:
        qs = eps * es * p / (p**2 + small**2)
    else:
        denom = (p - (1 - eps) * es + 0.0 * T1).clip(0.0)
        qs = eps * es * denom / (denom**2 + small**2)

    if input_type == "scalar":
        return float(qs) if np.isscalar(qs) else float(np.asarray(qs).flat[0])
    elif input_type == "xarray":
        return xr.DataArray(qs, coords=T.coords, dims=T.dims,
                            attrs={"units": "kg/kg"})
    return qs


def dqsat_dT(T, p, do_era5=False, small=0.0, do_simplified=False):
    """Compute temperature derivative of saturation specific humidity.

    Parameters
    ----------
    T : float, ndarray, or xarray.DataArray
        Temperature in Kelvin
    p : float or ndarray
        Pressure in hPa
    do_era5 : bool
        Use ERA5-compatible formula
    small : float
        Regularization parameter
    do_simplified : bool
        Use simplified denominator formula

    Returns
    -------
    dqs_dT : same type as T
        d(qsat)/dT in 1/K
    """
    input_type, _, T1 = _input_util_func(T)
    es = clausius_clapeyron_extended(T, do_era5=do_era5)
    des_dT = clausius_clapeyron_T_deriv(T, do_era5=do_era5)
    if do_simplified:
        dqs_dT = eps * des_dT * p / (p**2 + small**2)
    else:
        denom = p - (1 - eps) * es + 0.0 * T1
        mask = denom < 0.0
        d_denom_dT = -(1 - eps) * des_dT + 0.0 * T1
        denom = np.where(mask, 0.0, denom)
        d_denom_dT = np.where(mask, 0.0, d_denom_dT)
        dqs_dT = (eps * des_dT * denom / (denom**2 + small**2)
                  + eps * es * d_denom_dT / (denom**2 + small**2)
                  - eps * es * denom * (2 * denom * d_denom_dT)
                    / (denom**2 + small**2)**2)

    if input_type == "scalar":
        return float(dqs_dT) if np.isscalar(dqs_dT) else float(np.asarray(dqs_dT).flat[0])
    elif input_type == "xarray":
        return xr.DataArray(dqs_dT, coords=T.coords, dims=T.dims,
                            attrs={"units": "1/K"})
    return dqs_dT
