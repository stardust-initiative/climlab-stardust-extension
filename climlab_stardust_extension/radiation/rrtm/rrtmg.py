"""Extended combined RRTMG radiation model.

Extends climlab's ``RRTMG`` with:

* ``n_timestep_sw`` / ``n_timestep_lw`` — integer multipliers that set
  independent timesteps for the SW and LW subprocesses.
* Uses ``RRTMG_LW_extended`` and ``RRTMG_SW_extended`` as subprocesses,
  enabling ensemble Monte-Carlo averaging and aerosol-layer parameters.
* Passes ``liqflgsw`` to the SW subprocess (missing in original climlab).
"""

import numpy as np
from climlab import constants as const
from climlab.radiation.rrtm.rrtmg import RRTMG

from .rrtmg_lw import RRTMG_LW_extended
from .rrtmg_sw import RRTMG_SW_extended, nbndsw


class RRTMG_extended(RRTMG):
    """Extended RRTMG with per-component timesteps and ensemble support.

    After the parent ``RRTMG.__init__`` creates the standard LW and SW
    subprocesses, this class replaces them with extended versions that
    call the ``climlab-rrtmg-stardust`` ensemble Fortran drivers.

    Parameters
    ----------
    n_timestep_sw : int
        Multiplier for the SW subprocess timestep (default 1).
    n_timestep_lw : int
        Multiplier for the LW subprocess timestep (default 1).
    do_col_by_col : bool
        Column-by-column mode (default False).
    do_seed_permutation : bool
        Advance McICA seed each step (default False).
    n_rrtmg_repeat : int
        Number of ensemble members (default 1).
    return_spectral_asr : bool
        Return spectrally-resolved ASR (default False).
    kmodts : int
        Control flag for aerosol layer (default 2).
    add_aero_layer : int
        Enable aerosol layer treatment (default 0).
    r_mu, t_mu, r_bar, t_bar : float or ndarray
        Aerosol layer optical parameters.
    **kwargs
        Passed to ``RRTMG.__init__``.
    """

    def __init__(self,
                 n_timestep_sw=1,
                 n_timestep_lw=1,
                 do_col_by_col=False,
                 do_seed_permutation=False,
                 n_rrtmg_repeat=1,
                 return_spectral_asr=False,
                 kmodts=2,
                 add_aero_layer=0,
                 r_mu=0.0, t_mu=0.0, r_bar=0.0, t_bar=0.0,
                 **kwargs):
        assert int(n_timestep_sw) == n_timestep_sw, \
            'n_timestep_sw must be an integer'
        assert int(n_timestep_lw) == n_timestep_lw, \
            'n_timestep_lw must be an integer'

        timestep = kwargs.get('timestep', const.seconds_per_day)

        # Call parent init (creates original LW/SW subprocesses)
        super(RRTMG_extended, self).__init__(**kwargs)

        # Store extended attributes
        self.do_col_by_col = do_col_by_col
        self.do_seed_permutation = do_seed_permutation
        self.n_rrtmg_repeat = n_rrtmg_repeat

        # Extract common parameters already set by parent
        icld = self.icld
        irng = self.irng
        idrv = self.idrv
        iaer = self.iaer if hasattr(self, 'iaer') else kwargs.get('iaer', 0)

        # Collect kwargs for subprocesses, removing items already consumed
        # or passed explicitly to the LW/SW constructors below.
        sub_kwargs = dict(kwargs)
        remove_list = [
            'absorber_vmr', 'cldfrac', 'clwp', 'ciwp', 'r_liq', 'r_ice',
            'emissivity', 'aldif', 'aldir', 'asdif', 'asdir', 'S0', 'coszen',
            'irradiance_factor', 'insolation', 'timestep',
            # LW/SW shared flags
            'icld', 'irng', 'idrv', 'iaer', 'permuteseed',
            # LW-specific flags
            'inflglw', 'iceflglw', 'liqflglw', 'tauc_lw', 'tauaer_lw',
            # SW-specific flags
            'inflgsw', 'iceflgsw', 'liqflgsw', 'tauc_sw',
            'ssac_sw', 'asmc_sw', 'fsfc_sw',
            'tauaer_sw', 'ssaaer_sw', 'asmaer_sw', 'ecaer_sw',
            'dyofyr', 'isolvar', 'indsolvar', 'bndsolvar', 'solcycfrac',
            # Generic tauc/tauaer names that RRTMG.__init__ might set
            'tauc', 'tauaer', 'ssac', 'asmc', 'fsfc',
            'ssaaer', 'asmaer', 'ecaer',
        ]
        for item in remove_list:
            sub_kwargs.pop(item, None)

        # Replace LW subprocess with extended version
        LW = RRTMG_LW_extended(
            absorber_vmr=self.absorber_vmr,
            cldfrac=self.cldfrac,
            clwp=self.clwp,
            ciwp=self.ciwp,
            r_liq=self.r_liq,
            r_ice=self.r_ice,
            icld=icld,
            irng=irng,
            idrv=idrv,
            iaer=iaer,
            permuteseed=self.permuteseed_lw,
            emissivity=self.emissivity,
            inflglw=self.inflglw,
            iceflglw=self.iceflglw,
            liqflglw=self.liqflglw,
            tauc=self.tauc_lw,
            tauaer=self.tauaer_lw,
            do_col_by_col=do_col_by_col,
            do_seed_permutation=do_seed_permutation,
            n_rrtmg_repeat=n_rrtmg_repeat,
            timestep=n_timestep_lw * timestep,
            **sub_kwargs,
        )

        # Replace SW subprocess with extended version
        SW = RRTMG_SW_extended(
            absorber_vmr=self.absorber_vmr,
            cldfrac=self.cldfrac,
            clwp=self.clwp,
            ciwp=self.ciwp,
            r_liq=self.r_liq,
            r_ice=self.r_ice,
            icld=icld,
            irng=irng,
            permuteseed=self.permuteseed_sw,
            aldif=self.aldif,
            aldir=self.aldir,
            asdif=self.asdif,
            asdir=self.asdir,
            S0=self.S0,
            coszen=self.coszen,
            irradiance_factor=self.irradiance_factor,
            dyofyr=self.dyofyr,
            inflgsw=self.inflgsw,
            iceflgsw=self.iceflgsw,
            liqflgsw=self.liqflgsw,
            tauc=self.tauc_sw,
            ssac=self.ssac_sw,
            asmc=self.asmc_sw,
            fsfc=self.fsfc_sw,
            iaer=iaer,
            tauaer=self.tauaer_sw,
            ssaaer=self.ssaaer_sw,
            asmaer=self.asmaer_sw,
            ecaer=self.ecaer_sw,
            isolvar=self.isolvar,
            indsolvar=self.indsolvar,
            bndsolvar=self.bndsolvar,
            solcycfrac=self.solcycfrac,
            do_col_by_col=do_col_by_col,
            do_seed_permutation=do_seed_permutation,
            n_rrtmg_repeat=n_rrtmg_repeat,
            return_spectral_asr=return_spectral_asr,
            kmodts=kmodts,
            add_aero_layer=add_aero_layer,
            r_mu=r_mu,
            t_mu=t_mu,
            r_bar=r_bar,
            t_bar=t_bar,
            timestep=n_timestep_sw * timestep,
            **sub_kwargs,
        )

        self.add_subprocess('SW', SW)
        self.add_subprocess('LW', LW)
