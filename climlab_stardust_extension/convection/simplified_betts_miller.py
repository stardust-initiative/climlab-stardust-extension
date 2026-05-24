"""Extended Simplified Betts-Miller convection scheme.

Extends climlab's SimplifiedBettsMiller with surface pressure support,
an upper-level tendency cutoff (pmin), and input sanitization to
prevent segfaults in the Fortran ``betts_miller`` extension when
temperatures or humidities reach extreme values during spin-up.
"""

import warnings
import numpy as np
from climlab.convection.simplified_betts_miller import SimplifiedBettsMiller

try:
    from climlab_sbm_convection import betts_miller
except ImportError:
    betts_miller = None

from climlab import constants as const

HLv = const.Lhvap
Cp_air = const.cp
Grav = const.g
rdgas = const.Rd
rvgas = const.Rv
kappa = const.Rd / const.cp
# NB: es0 is NOT the physical saturation vapor pressure (≈611 Pa at 273 K).
# It is an internal normalization constant of the `lcltabl` lookup table
# in climlab_sbm_convection's betts_miller.f90. The table is calibrated
# for es0 = 1.0 — passing any other value shifts the log-lookup argument
# by ln(es0) and returns wrong LCL temperatures, which in turn silences
# convective triggering across the entire column. This matches what
# climlab (both upstream and the Stardust fork) has always passed.
# See fix/sbm-es0-normalization for the diagnostic trail.
es0 = 1.0


class SimplifiedBettsMiller_extended(SimplifiedBettsMiller):
    """Extended Simplified Betts-Miller convection scheme.

    Adds:
    - ``sp``: surface pressure parameter for the Fortran convection code.
    - ``pmin``: pressure level (hPa) above which tendencies are zeroed out,
      preventing convective adjustment in the upper stratosphere.
    - Input sanitization: temperatures are clamped to [100, 500] K and
      humidities to [0, 0.1] kg/kg before calling the Fortran extension,
      preventing overflow-triggered segfaults.

    Parameters
    ----------
    sp : float
        Surface pressure in hPa (default: 1000)
    pmin : float
        Minimum pressure level in hPa; tendencies above this are zeroed
        (default: 10)
    **kwargs :
        All other arguments passed to SimplifiedBettsMiller
    """
    def __init__(self, sp=1000.0, pmin=10.0, **kwargs):
        super(SimplifiedBettsMiller_extended, self).__init__(**kwargs)
        self.pmin = pmin
        if hasattr(self, '_IX') and hasattr(self, '_JX'):
            self.sp = 1e2 * sp * np.ones((self._IX, self._JX))
        else:
            self.sp = 1e2 * sp

    def _compute(self):
        T = self._climlab_to_sbm(self.state['Tatm'])
        RHBM = self._climlab_to_sbm(self.rhbm)
        dom = self.state['Tatm'].domain
        P = self._climlab_to_sbm(dom.lev.points) * 100.0
        PH = self._climlab_to_sbm(dom.lev.bounds) * 100.0
        Q = self._climlab_to_sbm(self.state['q'])
        dt = self.timestep

        # Sanitize inputs to prevent Fortran overflow / segfault
        T = np.clip(T, 100.0, 500.0)
        Q = np.clip(Q, 0.0, 0.1)

        with warnings.catch_warnings():
            warnings.filterwarnings('error', category=RuntimeWarning)
            try:
                (rain, tdel, qdel, q_ref, bmflag, klzbs, cape, cin, t_ref,
                 invtau_bm_t, invtau_bm_q, capeflag) = \
                    betts_miller(dt, T, Q, RHBM, P, PH,
                                 HLv, Cp_air, Grav, rdgas, rvgas, kappa, es0,
                                 self.tau_bm, self.do_simp, self.do_shallower,
                                 self.do_changeqref, self.do_envsat,
                                 self.do_taucape,
                                 self.capetaubm, self.tau_min,
                                 self._IX, self._JX, self._KX)
            except RuntimeWarning:
                # Overflow in Fortran cast — return zero tendencies
                tendencies = {
                    'Tatm': 0.0 * self.state['Tatm'],
                    'q': 0.0 * self.state['q'],
                }
                if 'Ts' in self.state:
                    tendencies['Ts'] = 0.0 * self.state['Ts']
                return tendencies

        dTdt = tdel / dt
        dQdt = qdel / dt
        tendencies = {
            'Tatm': self._sbm_to_climlab(dTdt) * np.ones_like(self.state['Tatm']),
            'q': self._sbm_to_climlab(dQdt) * np.ones_like(self.state['q']),
        }
        if 'Ts' in self.state:
            tendencies['Ts'] = 0.0 * self.state['Ts']
        if self.multidim:
            self.precipitation[:, 0] = self._sbm_to_climlab(rain) / dt
            self.cape[:, 0] = self._sbm_to_climlab(cape)
            self.cin[:, 0] = self._sbm_to_climlab(cin)
        else:
            self.precipitation[:] = self._sbm_to_climlab(rain) / dt
            self.cape[:] = self._sbm_to_climlab(cape)
            self.cin[:] = self._sbm_to_climlab(cin)

        # Zero tendencies above pmin
        ind = np.where(dom.lev.points <= self.pmin)[0]
        if len(self.Tatm.shape) > 1:
            tendencies['Tatm'][:, ind] = 0.0
            tendencies['q'][:, ind] = 0.0
        else:
            tendencies['Tatm'][ind] = 0.0
            tendencies['q'][ind] = 0.0
        return tendencies
