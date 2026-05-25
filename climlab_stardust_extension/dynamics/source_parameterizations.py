r"""Source-parameterization factory for aerosol injection.

Pairs a spatial parameterization (single grid point, gaussian, uniform,
external function) with a temporal one (constant, by-month, seasonal) to
build configurable aerosol source terms from plain dictionaries.
"""
from typing import Any, Dict, List, Type

import numpy as np

# ============================================================================
# Base Classes
# ============================================================================

class _SourceSpaceParameterization:
    def __init__(self, name, params, **kwargs):
        self.name = name
        self.params = params

    def compute(self, grid):
        return 0 * grid


class _SourceTimeParameterization:
    def __init__(self, name, params, **kwargs):
        self.name = name
        self.params = params

    def compute(self, t):
        time_factor = 1.0
        return time_factor


class _SourceSpaceTimeParameterization:
    def __init__(self, source_space_class: _SourceSpaceParameterization,
                 source_time_class: _SourceTimeParameterization,
                 name, params, **kwargs):
        self.space_proc = source_space_class(name, params, **kwargs)
        self.time_proc = source_time_class(name, params, **kwargs)
        self.name = name

    def compute(self, t, grid):
        return self.space_proc.compute(grid) * self.time_proc.compute(t)


# ============================================================================
# Space Parameterization Classes
# ============================================================================

class _SingleGridPoint(_SourceSpaceParameterization):
    def __init__(self, name, params, **kwargs):
        super(_SingleGridPoint, self).__init__(name, params, **kwargs)

    def compute(self, grid):
        source = np.zeros_like(grid)
        lat_index = np.searchsorted(self.params['lat_bounds'],
                                     self.params['point_source'][0], side='right') - 1
        lev_index = np.searchsorted(self.params['lev_bounds'],
                                     self.params['point_source'][1], side='right') - 1
        source[lat_index, lev_index] = self.params['rate']
        return source


class _ExternalFunc(_SourceSpaceParameterization):
    def __init__(self, name, params, **kwargs):
        super(_ExternalFunc, self).__init__(name, params, **kwargs)

    def compute(self, grid):
        source = self.params['func'](grid)
        return source

class _GaussianSpace(_SourceSpaceParameterization):
    """2D Gaussian spatial distribution - vectorized version"""
    def __init__(self, name, params, **kwargs):
        super(_GaussianSpace, self).__init__(name, params, **kwargs)

    def compute(self, grid):
        # Get parameters
        center_lat, center_lev = self.params['gaussian_center']  # [lat, lev]
        sigma_lat, sigma_lev = self.params['gaussian_sigma']     # [sigma_lat, sigma_lev]

        # Find center indices
        center_lat_idx = np.searchsorted(self.params['lat_bounds'], center_lat, side='right') - 1
        center_lev_idx = np.searchsorted(self.params['lev_bounds'], center_lev, side='right') - 1

        # Create index grids (meshgrid)
        n_lat, n_lev = grid.shape
        i_grid, j_grid = np.meshgrid(np.arange(n_lat), np.arange(n_lev), indexing='ij')

        # Distance from center in each dimension (vectorized)
        dlat = i_grid - center_lat_idx
        dlev = j_grid - center_lev_idx

        # 2D Gaussian (vectorized): exp(-0.5 * ((dlat/sigma_lat)^2 + (dlev/sigma_lev)^2))
        exponent = -0.5 * ((dlat / sigma_lat)**2 + (dlev / sigma_lev)**2)
        source = np.exp(exponent)

        # Normalize to 1
        total = source.sum()
        if total > 0:
            source = source / total

        # Apply rate
        source = source * self.params['rate']

        return source

class _UniformSpace(_SourceSpaceParameterization):
    """Uniform distribution over a region"""
    def __init__(self, name, params, **kwargs):
        super(_UniformSpace, self).__init__(name, params, **kwargs)

    def compute(self, grid):
        source = np.zeros_like(grid)

        # Get bounds
        lat_min, lat_max = self.params['lat_range']
        lev_min, lev_max = self.params['lev_range']

        # Find indices
        lat_idx_min = np.searchsorted(self.params['lat_bounds'], lat_min, side='right') - 1
        lat_idx_max = np.searchsorted(self.params['lat_bounds'], lat_max, side='right')
        lev_idx_min = np.searchsorted(self.params['lev_bounds'], lev_min, side='right') - 1
        lev_idx_max = np.searchsorted(self.params['lev_bounds'], lev_max, side='right')

        # Apply uniform rate
        n_cells = (lat_idx_max - lat_idx_min) * (lev_idx_max - lev_idx_min)
        if n_cells > 0:
            # source[lat_idx_min:lat_idx_max, lev_idx_min:lev_idx_max] = self.params['rate'] / n_cells
            # take into account the different volumes of cells:
            # M_air = 2*np.pi/3*const.a**2.0*np.diff(np.sin(np.deg2rad(self.params['lat_bounds'])))[:, None] * np.diff(self.params['lev_bounds']*100.0)[None,:] / const.g
            # m_region = M_air[lat_idx_min:lat_idx_max, lev_idx_min:lev_idx_max]
            m_region = np.copy(grid[lat_idx_min:lat_idx_max, lev_idx_min:lev_idx_max])
            m_region /= np.sum(m_region)
            source[lat_idx_min:lat_idx_max, lev_idx_min:lev_idx_max] = self.params['rate'] * m_region

        return source


# ============================================================================
# Time Parameterization Classes
# ============================================================================

class _ConstTime(_SourceTimeParameterization):
    def __init__(self, name, params, **kwargs):
        super(_ConstTime, self).__init__(name, params, **kwargs)

    def compute(self, t):
        time_factor = 1.0
        return time_factor


class _MonthlyTime(_SourceTimeParameterization):
    def __init__(self, name, params, **kwargs):
        super(_MonthlyTime, self).__init__(name, params, **kwargs)

    def compute(self, t):
        # Extract month from time (assuming t has a month attribute or similar)
        # Adjust this based on your actual time representation
        # if hasattr(t, 'month'):
        #     current_month = t.month
        # elif isinstance(t, (int, float)):
        #     # If t is a numeric value, you might need different logic
        #     current_month = int((t % 12) + 1)  # Simple example
        # else:
        #     current_month = 1  # Default
        current_month = t().month

        # Check if current month is in the active month list
        if current_month in self.params['month_list']:
            time_factor = 1.0
        else:
            time_factor = 0.0

        return time_factor


class _SeasonalTime(_SourceTimeParameterization):
    """Seasonal time parameterization"""
    def __init__(self, name, params, **kwargs):
        super(_SeasonalTime, self).__init__(name, params, **kwargs)

        # Define season to month mapping
        self.season_months = {
            'winter': [12, 1, 2],
            'spring': [3, 4, 5],
            'summer': [6, 7, 8],
            'fall': [9, 10, 11],
            'autumn': [9, 10, 11]
        }

    def compute(self, t):
        # Extract month
        # if hasattr(t, 'month'):
        #     current_month = t.month
        # elif isinstance(t, (int, float)):
        #     current_month = int((t % 12) + 1)
        # else:
        #     current_month = 1
        current_month = t().month

        # Check if current month is in the specified season
        season = self.params.get('season', 'winter')
        active_months = self.season_months.get(season.lower(), [])

        if current_month in active_months:
            time_factor = 1.0
        else:
            time_factor = 0.0

        return time_factor


# ============================================================================
# Registry and Factory Functions
# ============================================================================

# Registry mappings for space and time types
SPACE_TYPE_REGISTRY: Dict[str, Type[_SourceSpaceParameterization]] = {
    'single_grid_point': _SingleGridPoint,
    'external_func': _ExternalFunc,
    'gaussian': _GaussianSpace,
    'uniform': _UniformSpace,
}

TIME_TYPE_REGISTRY: Dict[str, Type[_SourceTimeParameterization]] = {
    'const': _ConstTime,
    'constant': _ConstTime,
    'by_month': _MonthlyTime,
    'monthly': _MonthlyTime,
    'seasonal': _SeasonalTime,
}


def register_space_type(name: str, space_class: Type[_SourceSpaceParameterization]):
    """
    Register a custom spatial parameterization class

    Args:
        name: String identifier for the space type
        space_class: Class inheriting from _SourceSpaceParameterization
    """
    SPACE_TYPE_REGISTRY[name] = space_class


def register_time_type(name: str, time_class: Type[_SourceTimeParameterization]):
    """
    Register a custom temporal parameterization class

    Args:
        name: String identifier for the time type
        time_class: Class inheriting from _SourceTimeParameterization
    """
    TIME_TYPE_REGISTRY[name] = time_class


def create_source(source_config: Dict[str, Any], **kwargs) -> _SourceSpaceTimeParameterization:
    """
    Create a source object from a configuration dictionary

    Args:
        source_config: Dictionary containing:
            - name: Source name
            - space_type: Type of spatial parameterization
            - time_type: Type of temporal parameterization
            - rate: Emission rate
            - Additional parameters specific to space/time types
        **kwargs: Additional keyword arguments passed to the source constructors

    Returns:
        _SourceSpaceTimeParameterization object

    Example:
        >>> config = {
        ...     'name': 'source1',
        ...     'space_type': 'single_grid_point',
        ...     'time_type': 'const',
        ...     'rate': 10.0,
        ...     'point_source': [45.0, 500.0],
        ...     'lat_bounds': np.array([...]),
        ...     'lev_bounds': np.array([...])
        ... }
        >>> source = create_source(config)
    """
    # Extract required fields
    name = source_config.get('name', f"source_{id(source_config)}")
    space_type = source_config.get('space_type')
    time_type = source_config.get('time_type')

    if not space_type or not time_type:
        raise ValueError(f"Source '{name}': Both 'space_type' and 'time_type' must be specified")

    # Look up the appropriate classes
    space_class = SPACE_TYPE_REGISTRY.get(space_type)
    time_class = TIME_TYPE_REGISTRY.get(time_type)

    if space_class is None:
        raise ValueError(
            f"Source '{name}': Unknown space_type '{space_type}'. "
            f"Available types: {list(SPACE_TYPE_REGISTRY.keys())}"
        )

    if time_class is None:
        raise ValueError(
            f"Source '{name}': Unknown time_type '{time_type}'. "
            f"Available types: {list(TIME_TYPE_REGISTRY.keys())}"
        )

    # Create and return the source object
    return _SourceSpaceTimeParameterization(
        source_space_class=space_class,
        source_time_class=time_class,
        name=name,
        params=source_config,
        **kwargs
    )


def create_sources_list(sources_config: List[Dict[str, Any]], **kwargs) -> List[_SourceSpaceTimeParameterization]:
    """
    Create a list of source objects from a list of configuration dictionaries

    Args:
        sources_config: List of source configuration dictionaries
        **kwargs: Additional keyword arguments passed to all source constructors

    Returns:
        List of _SourceSpaceTimeParameterization objects that can be used as:
        all_sources_mass = sum([source.compute(current_time, M_air) for source in sources_list])

    Example:
        >>> configs = [
        ...     {
        ...         'name': 'source1',
        ...         'space_type': 'single_grid_point',
        ...         'time_type': 'const',
        ...         'rate': 10.0,
        ...         'point_source': [45.0, 500.0],
        ...         'lat_bounds': lat_bounds,
        ...         'lev_bounds': lev_bounds
        ...     },
        ...     {
        ...         'name': 'source2',
        ...         'space_type': 'single_grid_point',
        ...         'time_type': 'by_month',
        ...         'rate': 5.0,
        ...         'point_source': [30.0, 700.0],
        ...         'month_list': [6, 7, 8],
        ...         'lat_bounds': lat_bounds,
        ...         'lev_bounds': lev_bounds
        ...     }
        ... ]
        >>> sources_list = create_sources_list(configs)
        >>> # Use in your code:
        >>> all_sources_mass = sum([source.compute(current_time, M_air) for source in sources_list])
    """
    sources_list = []

    for i, source_config in enumerate(sources_config):
        try:
            source = create_source(source_config, **kwargs)
            sources_list.append(source)
        except Exception as e:
            # Include context about which source failed
            source_name = source_config.get('name', f'source {i}')
            raise ValueError(f"Failed to create {source_name}: {str(e)}") from e

    return sources_list


# ============================================================================
# Validation Functions
# ============================================================================

def validate_source_config(source_config: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate a source configuration dictionary

    Args:
        source_config: Dictionary to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    name = source_config.get('name', 'unnamed')

    # Check required base fields
    required_fields = ['space_type', 'time_type', 'rate']
    for field in required_fields:
        if field not in source_config:
            return False, f"Source '{name}': Missing required field '{field}'"

    space_type = source_config['space_type']
    time_type = source_config['time_type']

    # Validate space_type
    if space_type not in SPACE_TYPE_REGISTRY:
        return False, f"Source '{name}': Invalid space_type '{space_type}'"

    # Validate time_type
    if time_type not in TIME_TYPE_REGISTRY:
        return False, f"Source '{name}': Invalid time_type '{time_type}'"

    # Space-type specific validation
    if space_type == 'single_grid_point':
        if 'point_source' not in source_config:
            return False, f"Source '{name}': space_type 'single_grid_point' requires 'point_source' field"
        if 'lat_bounds' not in source_config or 'lev_bounds' not in source_config:
            return False, f"Source '{name}': space_type 'single_grid_point' requires 'lat_bounds' and 'lev_bounds' fields"

    if space_type == 'external_func':
        if 'func' not in source_config:
            return False, f"Source '{name}': space_type 'external_func' requires 'func' field"

    if space_type == 'gaussian':
        required = ['center_lat', 'center_lev', 'sigma', 'lat_bounds', 'lev_bounds']
        for field in required:
            if field not in source_config:
                return False, f"Source '{name}': space_type 'gaussian' requires '{field}' field"

    if space_type == 'uniform':
        required = ['lat_range', 'lev_range', 'lat_bounds', 'lev_bounds']
        for field in required:
            if field not in source_config:
                return False, f"Source '{name}': space_type 'uniform' requires '{field}' field"

    # Time-type specific validation
    if time_type in ['by_month', 'monthly']:
        if 'month_list' not in source_config:
            return False, f"Source '{name}': time_type '{time_type}' requires 'month_list' field"

    if time_type == 'seasonal':
        if 'season' not in source_config:
            return False, f"Source '{name}': time_type 'seasonal' requires 'season' field"

    if time_type == 'hourly':
        if 'hour_profile' not in source_config:
            return False, f"Source '{name}': time_type 'hourly' requires 'hour_profile' field"
        elif len(source_config['hour_profile']) != 24:
            return False, f"Source '{name}': 'hour_profile' must have exactly 24 values"

    return True, ""


def validate_sources_config(sources_config: List[Dict[str, Any]]) -> tuple[bool, List[str]]:
    """
    Validate a list of source configurations

    Args:
        sources_config: List of dictionaries to validate

    Returns:
        Tuple of (all_valid, list_of_error_messages)
    """
    errors = []
    for i, source_config in enumerate(sources_config):
        is_valid, error_msg = validate_source_config(source_config)
        if not is_valid:
            errors.append(error_msg)

    return len(errors) == 0, errors
