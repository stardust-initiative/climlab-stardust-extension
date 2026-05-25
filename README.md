# climlab-stardust-extension

Extension package that adds functionality on top of
[climlab](https://climlab.readthedocs.io/) for stratospheric-aerosol
column-climate experiments. Designed as a proper extension package —
importing climlab and *inheriting* from its classes rather than forking
the source tree.

---

## What this package provides

Extended versions of climlab's radiation, convection, dynamics, and surface-flux
modules, plus aerosol optical-depth tables and shared utilities used by the
`stardust-climate` experiment driver (`climate_runs_ext`). Designed to sit
alongside a stock `climlab` install plus the native-code packages
`climlab-rrtmg_stardust` and `climlab-sbm-convection_stardust`.

### Naming convention

An extended class follows the pattern `ClassName_extended` (e.g.
`RRTMG_extended` inherits from climlab's `RRTMG`).  Entirely new classes
keep their plain names.

### Package layout

```
climlab_stardust_extension/
├── radiation/
│   ├── rrtm/                                 # RRTMG_extended, RRTMG_SW_extended,
│   │                                         #   RRTMG_LW_extended — aerosol-aware
│   │                                         #   radiation with MCICA cloud-overlap
│   │                                         #   repeat knobs
│   ├── optical_depth_tables_aerosols.py      # aerosol_instance,
│   │                                         #   AerosolsOptDepTables,
│   │                                         #   construct_uni_layer_vmr_p_based,
│   │                                         #   get_radiation_with_aerosols_params
│   └── insolation.py
├── convection/
│   ├── simplified_betts_miller.py            # SimplifiedBettsMiller_extended
│   │                                         #   (input sanitization + pmin cutoff)
│   └── large_scale_convection.py             # back-compatibility shim
├── dynamics/
│   ├── large_scale_condensation.py           # LargeScaleCondensation_extended
│   ├── meridional_moist_diffusion.py         # MoistMeridionalAdvectionDiffusion
│   ├── two_dimensional_advection_diffusion.py # 2-D MSE/q transport process
│   └── two_d_adv_diff_numerics.py             # NIRVANA advection numerics
├── surface/
│   ├── turbulent.py                          # LatentHeatFlux_extended,
│   │                                         #   SensibleHeatFlux_extended
│   └── oceanic_heat_uptake.py                # OceanicHeatUptake
└── utils/
    ├── thermo.py                             # qsat_extended,
    │                                         #   clausius_clapeyron_extended
    │                                         #   (Bolton + ERA5 Teten options)
    ├── file_handling.py                      # load_repo_table
    │                                         #   (pooch-cached downloads with
    │                                         #    commit-hash-keyed addressing
    │                                         #    and flat-cache fallback),
    │                                         #   _get_latest_commit_hash
    └── constants.py
```

---

## Relationship to other repos

```
                      ┌───────────────────────┐
                      │ climlab (upstream)    │
                      │ public, pip-installed │
                      └───────────┬───────────┘
                                  │ imports/inherits
                                  ▼
┌─────────────────────────┐   ┌─────────────────────────────┐
│ climlab-rrtmg_stardust  │◀─▶│ climlab-stardust-extension  │  ← THIS REPO
│ (Fortran RRTMG)         │   │ (extension classes,         │
└─────────────────────────┘   │  aerosol tables, utilities) │
                              └──────────────┬──────────────┘
┌─────────────────────────┐                  │
│ climlab-sbm-convection_ │◀─────────────────┤ used by
│ stardust (Fortran SBM)  │                  │
└─────────────────────────┘                  ▼
                              ┌─────────────────────────────┐
                              │ stardust-climate            │
                              │ (climate_runs_ext)          │
                              │ experiment driver + CLI     │
                              └─────────────────────────────┘
```

This package depends on:

- `climlab` (conda-forge / pip) — upstream radiation-convection framework.
- [`climlab-rrtmg` (Stardust fork)](https://github.com/stardust-initiative/climlab-rrtmg_stardust) — compiled Fortran RRTMG bindings with OpenMP column parallelization, ensemble cloud sampling, and spectral SW flux exposure.
- [`climlab-sbm-convection` (Stardust fork)](https://github.com/stardust-initiative/climlab-sbm-convection_stardust) — compiled Fortran Simplified Betts–Miller convection with the `betts_miller_pstar` (explicit surface pressure) variant.
- `numpy`, `xarray`, `scipy`, `pooch`, `requests`.

This package is consumed by [`stardust-climate`](https://github.com/stardust-initiative/stardust-climate) — the zonal-mean column climate runner and experiment CLI used for the stratospheric-aerosol SARF/ERF grid sweeps.

---

## Installation

This package is normally pulled transitively as a URL-pinned dependency of
[`stardust-climate`](https://github.com/stardust-initiative/stardust-climate);
the canonical paper-reproduction install lives in the umbrella repository
[`solid-sai-2d-paper`](https://github.com/stardust-initiative/solid-sai-2d-paper).

To **develop on this package directly** (or use it without the runner), install it locally:

```bash
conda create -n climlab_stardust_ext_env python=3.11 -y
conda activate climlab_stardust_ext_env
conda install -c conda-forge climlab compilers meson meson-python -y
pip install -e .
```

The two Stardust forks (`climlab-rrtmg`, `climlab-sbm-convection`) are pinned as URL dependencies in `pyproject.toml`, so `pip install` compiles them from source — that's why the conda environment above includes the C/Fortran toolchain.

Alternatively, the `environment.yml` shipped here pins both forks via `pip:` URLs and can be used directly:

```bash
conda env create -f environment.yml
conda activate climlab_stardust_ext_env
```

### Verify

```bash
python -c "import climlab_stardust_extension; print('ok')"
python -m pytest tests/ -q
```

All 67 tests should pass (as of `v0.1.2`).

---

## Usage

This package is a library; it doesn't ship a command-line entry point.
The expected pattern is to build a climlab model with extended subprocesses
from this package. Minimal example using `RRTMG_extended` and
`SimplifiedBettsMiller_extended`:

```python
import climlab
import numpy as np
from climlab_stardust_extension.radiation.rrtm import RRTMG_extended
from climlab_stardust_extension.convection.simplified_betts_miller import (
    SimplifiedBettsMiller_extended,
)

# Single-column state with humidity
state = climlab.column_state(num_lev=50)
state['q'] = 1e-3 * np.ones_like(state['Tatm'])

# Extended radiation process with MCICA cloud-overlap repeat knobs
rad = RRTMG_extended(
    name='Radiation', state=state, timestep=86400.0,
    n_rrtmg_repeat=5, do_seed_permutation=True, do_col_by_col=True,
)

# Extended convection with input sanitization and upper-level cutoff
conv = SimplifiedBettsMiller_extended(
    state=state, timestep=3600.0,
    pmin=10.0,          # zero tendencies above 10 hPa (stratosphere)
    do_envsat=False,
    do_shallower=True,
    do_changeqref=True,
)

# Couple and integrate as usual
model = climlab.couple([rad, conv], name='mini-RCE')
model.integrate_days(30.0)
```

### Aerosol-aware radiation

For stratospheric-aerosol experiments, build an `AerosolsOptDepTables`
and pass it into `RRTMG_extended` via `rad_with_aero_param_dict`:

```python
from climlab_stardust_extension.radiation.optical_depth_tables_aerosols import (
    aerosol_instance,
    AerosolsOptDepTables,
    construct_uni_layer_vmr_p_based,
    get_radiation_with_aerosols_params,
)

# Uniform layer of silica particles (250 nm radius) between 30-80 hPa
rho = 2196.0  # silica density [kg/m^3]
r_m = 250e-9  # particle radius [m]
burden_kg_m2 = np.ones(len(lat))  # column mass per latitude
vmr = construct_uni_layer_vmr_p_based(
    rho, burden_kg_m2, 30.0 * 100, 80.0 * 100, r_m, rad.state,
)

aer_table = AerosolsOptDepTables(
    aerosol_instance_list=[aerosol_instance('silica', r_m, vmr)],
    domain=rad.Tatm.domain, coszen=rad.coszen,
    **config['aerosols_input_dict'],
)
rad_with_aero_param_dict = get_radiation_with_aerosols_params(
    rad.state, aer_table, rad.coszen,
)

# ... pass rad_with_aero_param_dict into an RRTMG_extended subclass
```

For a full-featured worked example (rev6 reference model, SARF / ERF grid
sweeps, ERA5 initial state, file-based transport layers), see the
[`stardust-climate` experiment driver](https://github.com/stardust-initiative/stardust-climate).

---

## Development notes

### Data access

Aerosol optical-property tables and ERA5-derived inputs are loaded via the
[`stardust-2d-inputs`](https://github.com/stardust-initiative/stardust-2d-inputs)
engine — the lean runtime loader + provenance registry that backs the
paper's input set. Files are content-addressed and fetched on demand from
the public Zenodo deposit
([10.5281/zenodo.20271742](https://doi.org/10.5281/zenodo.20271742)),
sha-256-verified, and locally cached. No credentials required for the
public release manifest. See the engine's README for backend configuration
options.

### Tests

```bash
python -m pytest tests/ -q
```

Tests use mocks for the external data dependencies and synthetic inputs
for the Fortran bindings, so no network access or real ERA5 data is
required. A full end-to-end integration test lives in
[`stardust-climate`](https://github.com/stardust-initiative/stardust-climate).

### Known constants that look wrong but aren't

- `convection/simplified_betts_miller.py` sets `es0 = 1.0`. **This is not a
  physical saturation vapor pressure**; it is an internal normalization
  constant of the `lcltabl` lookup table inside the
  `climlab_sbm_convection` Fortran scheme. Do NOT change it to 611 Pa — the
  lookup table is calibrated assuming `es0 = 1` and any other value shifts
  the LCL lookup argument by `ln(es0)` and returns unphysical LCL
  temperatures, silently silencing convective triggering across the entire
  column. See the comment at the top of `simplified_betts_miller.py` and
  the `test_convection_triggers_on_unstable_profile` regression test.

## License

[MIT](LICENSE).

## Contact

Stardust Labs ltd — [stardust-initiative](https://github.com/stardust-initiative) on GitHub.
