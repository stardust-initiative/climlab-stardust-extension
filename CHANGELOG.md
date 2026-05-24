# Changelog

All notable changes to this project are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-05-24

Initial public release.

### Added

- Extended climlab subprocesses for stratospheric-aerosol column-climate
  experiments:
  - **Radiation** — `RRTMG_extended`, `RRTMG_SW_extended`,
    `RRTMG_LW_extended` (aerosol-aware with MCICA cloud-overlap repeat
    knobs, ensemble cloud sampling, spectral SW flux exposure);
    `AerosolsOptDepTables` for RRTMG-banded optical-property tables.
  - **Convection** — `SimplifiedBettsMiller_extended` (input sanitisation
    + `pmin` cutoff).
  - **Dynamics** — `LargeScaleCondensation_extended`,
    `MoistMeridionalAdvectionDiffusion`, the 2-D MSE/q transport process
    with NIRVANA advection numerics.
  - **Surface** — `LatentHeatFlux_extended`, `SensibleHeatFlux_extended`,
    `OceanicHeatUptake`.
- Pinned dependencies on the Stardust forks
  [`climlab-rrtmg` v0.5.0](https://github.com/stardust-initiative/climlab-rrtmg_stardust)
  and
  [`climlab-sbm-convection` v0.3.0](https://github.com/stardust-initiative/climlab-sbm-convection_stardust)
  via `git+https` URLs in both `pyproject.toml` and `environment.yml`.
- Data access routed through the
  [`stardust-2d-inputs`](https://github.com/stardust-initiative/stardust-2d-inputs)
  engine — sha-256-verified, locally cached, no credentials required for
  the public `transport-paper-v1` release.
