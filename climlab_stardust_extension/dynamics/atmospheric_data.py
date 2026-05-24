r"""Time- and space-interpolation of atmospheric driver fields.

AtmosphericData regrids time-varying atmospheric fields (winds, eddy
diffusivities, tropopause pressure) onto the simulation grid and
interpolates them in time -- either as a repeating monthly climatology
or by real date.
"""
import calendar
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from climlab.process.time_dependent_process import TimeDependentProcess

class AtmosphericData(TimeDependentProcess):

    def __init__(self,
                 param_configs=None,  # Dict of dicts: {'u': {'data': xarray, 'method': 'linear', 'grid_type': 'centers*centers'}, ...}
                 t_0=None,  # datetime object for initial time
                 time_type=1,  # 0: constant, 1: yearly cycle, 2: real date
                 **kwargs):
        super(AtmosphericData, self).__init__(**kwargs)
        
        # Initialize t_0 as datetime
        if t_0 is None:
            self._current_time = datetime(2000, 1, 1)  # Default start
        elif isinstance(t_0, datetime):
            self._current_time = t_0
        else:
            raise ValueError("t_0 must be a datetime object")
        
        self._time_type = time_type
        
        # Extract grid information from state (inherited from parent class)
        domain = list(self.domains.values())[0]
        self._lat_bounds = domain.axes['lat'].bounds
        self._lev_bounds = domain.axes['lev'].bounds
        
        # Calculate centers as midpoints between bounds
        self._lat_centers = (self._lat_bounds[:-1] + self._lat_bounds[1:]) / 2
        self._lev_centers = (self._lev_bounds[:-1] + self._lev_bounds[1:]) / 2
        
        # Process and interpolate parameter configurations
        self._param_configs = {}
        if param_configs is not None:
            for param_name, config in param_configs.items():
                self.add_param(param_name, 
                             config.get('data'),
                             config.get('method', 'linear'),
                             config.get('grid_type', 'centers*centers'))
        
        self._current_data = self._interp_data(self._current_time)

    def _get_target_grid(self, grid_type):
        """
        Get target latitude and level grids based on grid_type.
        
        Parameters:
        -----------
        grid_type : str
            For 2D data: 'centers*centers', 'bounds*bounds', 'centers*bounds', 'bounds*centers'
            For 1D data: 'centers' or 'bounds' (latitude only)
            Format for 2D: 'lat_type*lev_type'
            Format for 1D: 'lat_type'
            
        Returns:
        --------
        tuple : (target_lats, target_levs) for 2D, or (target_lats,) for 1D
        """
        # Check if this is 1D or 2D grid specification
        if '*' not in grid_type:
            # 1D case - latitude only
            if grid_type == 'centers':
                return (self._lat_centers,)
            elif grid_type == 'bounds':
                return (self._lat_bounds,)
            else:
                raise ValueError(f"Invalid 1D grid type: {grid_type}. Must be 'centers' or 'bounds'")
        
        # 2D case
        grid_parts = grid_type.split('*')
        if len(grid_parts) != 2:
            raise ValueError(f"Invalid grid_type format: {grid_type}. Expected 'lat_type*lev_type' or 'lat_type'")
        
        lat_type, lev_type = grid_parts
        
        # Select latitude grid
        if lat_type == 'centers':
            target_lats = self._lat_centers
        elif lat_type == 'bounds':
            target_lats = self._lat_bounds
        else:
            raise ValueError(f"Invalid latitude grid type: {lat_type}. Must be 'centers' or 'bounds'")
        
        # Select level grid
        if lev_type == 'centers':
            target_levs = self._lev_centers
        elif lev_type == 'bounds':
            target_levs = self._lev_bounds
        else:
            raise ValueError(f"Invalid level grid type: {lev_type}. Must be 'centers' or 'bounds'")
        
        return target_lats, target_levs

    def _interpolate_to_simulation_grid(self, data, grid_type):
        """
        Interpolate data from its original lat(-level) grid to the simulation grid.
        Handles both 1D (latitude only) and 2D (latitude and level) data.
        Ensures 2D output has dimensions ordered as (lat, level).
        
        Parameters:
        -----------
        data : xarray.DataArray
            Input data with dimensions including lat and optionally level
        grid_type : str
            For 2D: 'centers*centers', 'bounds*bounds', etc.
            For 1D: 'centers' or 'bounds'
            
        Returns:
        --------
        xarray.DataArray : Data interpolated to the target grid
        """
        # Get target grids
        target_grids = self._get_target_grid(grid_type)
        is_1d = len(target_grids) == 1
        
        # Find the latitude and level dimension names in the input data
        lat_dim = None
        lev_dim = None
        
        possible_lat_names = ['lat', 'latitude', 'Lat', 'Latitude', 'LAT']
        possible_lev_names = ['lev', 'level', 'Level', 'LEV', 'pressure', 'z', 'altitude']
        
        for dim in data.dims:
            if dim in possible_lat_names:
                lat_dim = dim
            if dim in possible_lev_names:
                lev_dim = dim
        
        if lat_dim is None:
            raise ValueError(f"Could not find lat dimension in data. Available dims: {data.dims}")
        
        # Build interpolation dictionary
        if is_1d:
            # 1D case - latitude only
            target_lats = target_grids[0]
            if lev_dim is not None:
                raise ValueError(f"1D grid_type '{grid_type}' specified but data has level dimension '{lev_dim}'")
            
            interp_dict = {lat_dim: target_lats}
        else:
            # 2D case - latitude and level
            target_lats, target_levs = target_grids
            if lev_dim is None:
                raise ValueError(f"2D grid_type '{grid_type}' specified but data has no level dimension")
            
            interp_dict = {
                lat_dim: target_lats,
                lev_dim: target_levs
            }
        
        # Perform interpolation with extrapolation for out-of-bounds points
        interpolated_data = data.interp(interp_dict, method='linear')

        # interpolated_data = data.interp(interp_dict, method='linear')
        # interpolated_data.data=np.where(np.isnan(interpolated_data.data),0.0,interpolated_data.data)
        # # Option A: Use extrapolate for slightly out-of-bounds points
        # interpolated_data = data.interp(interp_dict, method='linear', 
        #                                 kwargs={'fill_value': 'extrapolate'})
        
        # Option B: Use nearest neighbor for out-of-bounds points
        # interpolated_data = data.interp(interp_dict, method='linear')
        if interpolated_data.isnull().any():
            nearest_data = data.interp(interp_dict, method='nearest', 
                                  kwargs={'fill_value': 'extrapolate'})
            interpolated_data = interpolated_data.fillna(nearest_data)        
        
        # Rename dimensions to standard names
        rename_dict = {lat_dim: 'lat'}
        if lev_dim is not None:
            rename_dict[lev_dim] = 'level'
        interpolated_data = interpolated_data.rename(rename_dict)
        
        # Ensure dimensions are ordered correctly
        if not is_1d and 'lat' in interpolated_data.dims and 'level' in interpolated_data.dims:
            # 2D case: ensure (lat, level) order
            all_dims = list(interpolated_data.dims)
            all_dims.remove('lat')
            all_dims.remove('level')
            new_dim_order = all_dims + ['lat', 'level']
            interpolated_data = interpolated_data.transpose(*new_dim_order)
        
        return interpolated_data

    def _compute(self):
        """
        Update atmospheric data to match the current time.
        Updates data in-place when possible to maintain references.
        
        Returns:
        --------
        tendencies : dict
            Empty dictionary (atmospheric data doesn't compute tendencies,
            it just updates to match current time).
        """
        tendencies = {}
        
        # Advance the time by timestep
        self._current_time = self._current_time + timedelta(seconds=self.timestep)
        
        # Get new interpolated data
        new_data = self._interp_data(self._current_time)
        
        # Update existing data in-place to maintain references
        for param_name, new_array in new_data.items():
            if param_name in self._current_data:
                # Update the values in-place
                self._current_data[param_name].values[:] = new_array.values
            else:
                # New parameter - add it
                self._current_data[param_name] = new_array
        
        return tendencies
    
    def _datetime_to_month_fraction(self, dt):
        """
        Convert datetime to fractional month [1.0 - 13.0).
        Month 1 = January center, Month 12 = December center
        Month 13 wraps back to Month 1 for interpolation purposes.
        
        Examples:
        - Jan 1 00:00 -> ~1.0 (start of month, but data point is at center)
        - Jan 15 12:00 (mid-month) -> ~1.5
        - Dec 31 23:59 -> ~12.97
        """
        month = dt.month
        day = dt.day
        hour = dt.hour
        minute = dt.minute
        second = dt.second
        
        days_in_month = calendar.monthrange(dt.year, dt.month)[1]
        
        # Convert time of day to fraction of a day
        day_fraction = (hour + minute/60 + second/3600) / 24
        
        # Day progress: day 1 at 00:00 -> 0.0, day 15 at 12:00 in a 30-day month -> 0.5
        day_progress = (day - 1 + day_fraction) / days_in_month
        
        return month + day_progress

    def _ensure_lat_level_order(self, data):
        """
        Ensure that 2D data has dimensions ordered as (lat, level).
        1D data (latitude only) is left unchanged.
        
        Parameters:
        -----------
        data : xarray.DataArray
            Input data
            
        Returns:
        --------
        xarray.DataArray : Data with dims ordered correctly
        """
        # Only reorder if both lat and level dimensions exist (2D case)
        if 'lat' in data.dims and 'level' in data.dims:
            # Get current dimension order
            current_dims = list(data.dims)
            
            # Check if lat comes before level
            lat_idx = current_dims.index('lat')
            level_idx = current_dims.index('level')
            
            if lat_idx > level_idx:
                # Need to reorder - swap lat and level
                # Find all dims and put lat before level
                other_dims = [d for d in current_dims if d not in ['lat', 'level']]
                new_order = other_dims + ['lat', 'level']
                data = data.transpose(*new_order)
            elif lat_idx < level_idx:
                # Already in correct order, but ensure no other dims between them
                other_dims = [d for d in current_dims if d not in ['lat', 'level']]
                new_order = other_dims + ['lat', 'level']
                data = data.transpose(*new_order)
        # If only 'lat' dimension exists (1D case), no reordering needed
        
        return data

    def _interp_data(self, time):
        """
        Interpolate all parameters at the given time.
        Each parameter uses its own xarray and interpolation method.
        Returns a dict with interpolated xarray.DataArrays (2D spatial fields with dims (lat, level)).
        """
        interpolated = {}
        
        for param_name, config in self._param_configs.items():
            data = config.get('data')
            method = config.get('method', 'linear')
            
            if data is None:
                continue
            
            # Get the time dimension for this specific xarray
            time_dim = self._get_time_dimension(data)
            
            if time_dim is None or self._time_type == 0:
                # No time dimension or constant in time - return mean over time or as-is
                if time_dim and time_dim in data.dims:
                    result = data.mean(dim=time_dim)
                else:
                    result = data
            elif self._time_type == 1:
                # Yearly cycle - interpolate based on month
                month_frac = self._datetime_to_month_fraction(time)
                result = self._interp_monthly_cycle(
                    data, time_dim, month_frac, method
                )
            elif self._time_type == 2:
                # Real date - direct interpolation
                result = self._interp_real_date(
                    data, time_dim, time, method
                )
            
            # Ensure dimensions are ordered as (lat, level)
            interpolated[param_name] = self._ensure_lat_level_order(result)
        
        return interpolated
    
    def _interp_monthly_cycle(self, data, time_dim, month_frac, method):
        """
        Interpolate monthly climatology data with cyclic boundary conditions.
        Assumes data has 12 points representing months 1-12, where each point
        represents the CENTER of the month (e.g., Jan 15, Feb 14, etc.).
        Returns interpolated 2D array (or whatever spatial dimensions exist).
        """
        # Ensure we have exactly 12 data points
        if data.sizes[time_dim] != 12:
            raise ValueError(f"For yearly cycle, expected 12 data points, got {data.sizes[time_dim]}")
        
        # Calculate the center of each month as fractional month values
        # For a 31-day month like January: center is day 16 -> (16-1)/31 = 0.484
        # For a 28-day month like February: center is day 15 -> (15-1)/28 = 0.5
        # We'll use a standard year (non-leap) for the climatology
        days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        month_centers = []
        for i, days in enumerate(days_in_months):
            center_day = (days + 1) / 2  # Center of month (e.g., 15.5 for 30-day month)
            month_centers.append(i + 1 + (center_day - 1) / days)
        
        month_centers = np.array(month_centers)
        
        # Create extended array for cyclic interpolation
        # Add December center at the beginning (shifted by -12) and January center at the end (shifted by +12)
        extended_data = xr.concat([data.isel({time_dim: -1}),  # December
                                   data, 
                                   data.isel({time_dim: 0})],  # January
                                  dim=time_dim)
        
        extended_month_centers = np.concatenate([
            month_centers[-1:] - 12,  # December center from previous year
            month_centers,             # All 12 month centers
            month_centers[:1] + 12     # January center from next year
        ])
        
        extended_data = extended_data.assign_coords({time_dim: extended_month_centers})
        
        # Interpolate - this preserves all spatial dimensions
        interpolated_val = extended_data.interp(
            {time_dim: month_frac},
            method=method
        )
        
        return interpolated_val
    
    def _interp_real_date(self, data, time_dim, time, method):
        """
        Interpolate data using real datetime values.
        Assumes the xarray has datetime64 coordinates.
        Returns interpolated 2D array (or whatever spatial dimensions exist).
        """
        interp_kwargs = {'fill_value': 'extrapolate'}
        interpolated_val = data.interp(
            {time_dim: time},
            method=method,
            kwargs=interp_kwargs
        )
        return interpolated_val
    
    def _get_time_dimension(self, data_array):
        """
        Find the time dimension in the xarray.
        Looks for common time dimension names.
        """
        if isinstance(data_array, xr.Dataset):
            dims = data_array.dims
        else:  # DataArray
            dims = data_array.dims
            
        possible_time_dims = ['time', 't', 'Time', 'TIME', 'month']
        for dim in possible_time_dims:
            if dim in dims:
                return dim
        return None
    
    def add_param(self, param_name, data, method='linear', grid_type='centers*centers'):
        """
        Add a new parameter or update an existing one.
        Data is automatically interpolated to the simulation grid with dims (lat, level) for 2D
        or (lat,) for 1D data.
        
        Parameters:
        -----------
        param_name : str
            Name of the parameter (e.g., 'u', 'v', 'kyy', 'tropopause_height')
        data : xarray.DataArray or xarray.Dataset
            Time-series data for the parameter.
            For 2D data:
                - time_type=1 (yearly cycle): shape (12, lat, level) with month coordinate [1-12]
                - time_type=2 (real date): shape (time, lat, level) with datetime64 coordinates
            For 1D data:
                - time_type=1 (yearly cycle): shape (12, lat) with month coordinate [1-12]
                - time_type=2 (real date): shape (time, lat) with datetime64 coordinates
        method : str
            Interpolation method for time ('linear', 'nearest', 'cubic', etc.)
        grid_type : str
            Target grid type for spatial interpolation.
            For 2D: 'centers*centers', 'bounds*bounds', 'centers*bounds', 'bounds*centers'
            For 1D: 'centers' or 'bounds'
        """
        # Interpolate data to simulation grid (this also ensures correct dimension order)
        interpolated_data = self._interpolate_to_simulation_grid(data, grid_type)
        
        self._param_configs[param_name] = {
            'data': interpolated_data,
            'method': method,
            'grid_type': grid_type
        }
        
        # Update current data with the new parameter
        self._current_data = self._interp_data(self._current_time)
    
    def remove_param(self, param_name):
        """Remove a parameter from the configuration."""
        if param_name in self._param_configs:
            del self._param_configs[param_name]
            if param_name in self._current_data:
                del self._current_data[param_name]
    
    def get_param(self, param_name):
        """
        Get the current interpolated 2D field for a specific parameter.
        Returns xarray.DataArray with spatial dimensions ordered as (lat, level).
        """
        return self._current_data.get(param_name, None)
    
    def get_param_at_location(self, param_name, lat=None, level=None, method='nearest'):
        """
        Get the interpolated value at a specific latitude and level.
        
        Parameters:
        -----------
        param_name : str
            Name of the parameter
        lat : float
            Latitude value
        level : float
            Vertical level value
        method : str
            Interpolation method for spatial interpolation
            
        Returns:
        --------
        float : Interpolated value at the specified location
        """
        data = self.get_param(param_name)
        if data is None:
            return None
        
        interp_dict = {}
        if lat is not None and 'lat' in data.dims:
            interp_dict['lat'] = lat
        if level is not None and 'level' in data.dims:
            interp_dict['level'] = level
        
        if interp_dict:
            interpolated = data.interp(interp_dict, method=method)
            return float(interpolated.values)
        else:
            return float(data.values)
    
    def get_all_params(self):
        """
        Get all current interpolated 2D fields as a dictionary.
        Returns dict of xarray.DataArrays with dims (lat, level).
        """
        return self._current_data.copy()
    
    def available_params(self):
        """Return list of available parameter names."""
        return list(self._param_configs.keys())
    
    def get_current_time(self):
        """Return the current time as a datetime object."""
        return self._current_time
    
    def update_current_data(self):
        """Update _current_data to match the current time."""
        self._current_data = self._interp_data(self._current_time)
    
    @property
    def current_time(self):
        """Property to access current time."""
        return self._current_time
    
    @current_time.setter
    def current_time(self, value):
        """Property setter for current time."""
        if isinstance(value, datetime):
            self._current_time = value
        else:
            raise ValueError("current_time must be a datetime object")
