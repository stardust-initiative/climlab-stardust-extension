r"""ParticleSource: aerosol injection process.

ParticleSource builds a list of source parameterizations (see
``source_parameterizations``) and injects the resulting mass source into a
single transported tracer.

The source list may be supplied directly as ``sources_config`` (a list of
source dictionaries) or loaded from a JSON file via ``sources_filename``.
The JSON file has the form ``{"sources": [ {...}, {...} ]}``; each source
dictionary requires:

- ``name``
- ``space_type``: ``single_grid_point``, ``gaussian``, ``uniform``,
  ``external_func``
- ``time_type``: ``const``, ``by_month``, ``seasonal``
- ``rate``: a number, or a string arithmetic expression of numbers
  (e.g. ``"1e9 / (365 * 86400)"``)

Space-/time-type-specific fields (``point_source``, ``month_list``, ...)
are documented in ``source_parameterizations``.
"""
import ast
import json
import operator

import numpy as np
from climlab import constants as const
from climlab.process.time_dependent_process import TimeDependentProcess

from climlab_stardust_extension.dynamics.source_parameterizations import (
    create_sources_list,
)

_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval_number(expr):
    """Evaluate a numeric arithmetic expression without executing code.

    Accepts numeric literals combined with ``+ - * / **`` and unary signs;
    rejects names, calls, attribute access and anything else.
    """
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
            return _UNARYOPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"unsupported element in numeric expression: {expr!r}")
    return _eval(ast.parse(expr, mode='eval'))


class ParticleSource(TimeDependentProcess):
    """Inject an aerosol mass source into a single transported tracer."""

    def __init__(self, bin_fraction=1.0, current_time=None,
                 diagnostic_name_suffix="", sources_filename=None,
                 sources_config=None, temperature=None, **kwargs):
        super(ParticleSource, self).__init__(**kwargs)
        for dom in list(self.domains.values()):
            self._phibounds = np.deg2rad(dom.axes['lat'].bounds)
            self._latbounds = np.sin(self._phibounds) * const.a
            self._dlatbounds = np.diff(self._latbounds)
            self._levbounds = dom.axes['lev'].bounds * 1e2
            self._dlevbounds = np.diff(self._levbounds)
            self._latpoints = 0.5*(self._latbounds[1:] + self._latbounds[:-1])
            self._levpoints = 0.5*(self._levbounds[1:] + self._levbounds[:-1])
            self.current_time = current_time
            self.M_air = (2*np.pi*const.a**2.0
                          * np.diff(np.sin(self._phibounds))[:, None]
                          * self._dlevbounds[None, :] / const.g)
            self.total_tracer_source = 0.0
            self.bin_fraction = bin_fraction
            self.volume = None
            if temperature is not None:
                rho_air = self._levpoints[None, :] / const.Rd / temperature
                self.volume = self.M_air / rho_air
        self.diagname_total_tracer_source = \
            'total_tracer_source' + diagnostic_name_suffix
        self.add_diagnostic(self.diagname_total_tracer_source, np.array([0.0]))

        if sources_config is None:
            if sources_filename is None:
                raise ValueError(
                    "ParticleSource requires either 'sources_config' (a list "
                    "of source dicts) or 'sources_filename' (a JSON file path)."
                )
            with open(sources_filename, 'r') as f:
                sources_config = json.load(f)['sources']

        for cfg in sources_config:
            cfg["lat_bounds"] = np.rad2deg(self._phibounds)
            cfg["lev_bounds"] = self._levbounds / 1e2
            if "rate" in cfg and isinstance(cfg["rate"], str):
                try:
                    cfg["rate"] = _safe_eval_number(cfg["rate"])
                except Exception as e:
                    raise ValueError(
                        f"Invalid rate expression '{cfg['rate']}' in "
                        f"{cfg.get('name', 'unnamed source')}"
                    ) from e
            cfg["rate"] = cfg["rate"] * self.bin_fraction

        self.sources_list = create_sources_list(sources_config)

    def _compute(self):
        assert len(self.state.keys()) == 1, \
            f'ParticleSource supports a single state variable only: {self.state.keys()}'

        grid = self.M_air if self.volume is None else self.volume
        all_sources_mass = sum(source.compute(self.current_time, grid)
                               for source in self.sources_list)
        M_step = np.sum(all_sources_mass) * self.timestep
        x_plus = all_sources_mass / self.M_air

        tendencies = {}
        for name, value in self.state.items():
            tendencies[name] = value * 0 + x_plus

        self.total_tracer_source += M_step
        self.__setattr__(self.diagname_total_tracer_source,
                         np.array([self.total_tracer_source * 1.0]))

        return tendencies
