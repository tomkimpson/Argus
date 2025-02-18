"""Module for loading pulsar data."""

import numpy as np
import pandas as pd
from functools import reduce
from enterprise.pulsar import Pulsar as EnterprisePulsar


class LoadWidebandPulsarData:
    """A class to load and process pulsar data at a single frequency channel.

    Attributes
    ----------
    toas : np.ndarray
        Times of arrival of the pulsar signals.
    toaerrs : np.ndarray
        Errors associated with the times of arrival.
    residuals : np.ndarray
        Residuals of the pulsar timing model.
    fitpars : dict
        Fitted parameters of the pulsar timing model.
    toa_diffs : np.ndarray
        Differences between consecutive times of arrival.
    toa_diff_errors : np.ndarray
        Errors associated with the differences between consecutive times of arrival.
    M_matrix : np.ndarray
        Design matrix for the pulsar timing model.
    name : str
        Name of the pulsar.
    RA : float or str
        Right Ascension of the pulsar.
    DEC : float or str
        Declination of the pulsar.

    Methods
    -------
    __init__(ds_psr)
        Initializes the LoadWidebandPulsarData object with pulsar data.
    read_par_tim(par_file, tim_file, **kwargs)
        Class method to load pulsar data from parameter and timing files.
    read_multiple_par_tim(par_files, tim_files, max_files=None)
        Class method to load multiple par/tim file pairs and return aggregated
        DataFrames and an angular separation matrix.

    """

    def __init__(self, ds_psr):
        """Initialize the LoadWidebandPulsarData object with pulsar data.

        Parameters
        ----------
        ds_psr : object
            An object containing pulsar data (e.g., an instance of enterprise.pulsar.Pulsar)
            with attributes: toas, toaerrs, residuals, fitpars, Mmat, name, _raj, and _decj.

        """
        self.toas = ds_psr.toas
        self.toaerrs = ds_psr.toaerrs
        self.residuals = ds_psr.residuals
        self.fitpars = ds_psr.fitpars
        self.M_matrix = ds_psr.Mmat
        self.name = ds_psr.name
        self.RA = ds_psr._raj
        self.DEC = ds_psr._decj

        # Compute differences between consecutive TOAs and propagate errors.
        self.toa_diffs = np.diff(self.toas)
        self.toa_diff_errors = np.sqrt(self.toaerrs[1:] ** 2 + self.toaerrs[:-1] ** 2)


    @staticmethod
    def pairwise_angular_separation(ra_rad, dec_rad):
        """Compute the pairwise angular separations for a set of celestial coordinates in radians.
        
        This function takes arrays of right ascension (RA) and declination (Dec), both in radians,
        and returns an NxN matrix of angular separations, where N is the length of the input arrays.
        Each entry (i, j) in the output is the angular separation between the coordinate pair
        (ra_rad[i], dec_rad[i]) and (ra_rad[j], dec_rad[j]).
        
        Parameters
        ----------
        ra_rad : numpy.ndarray
            1D array of right ascensions in radians, of length N.
        dec_rad : numpy.ndarray
            1D array of declinations in radians, of length N.
            
        Returns
        -------
        sep_rad : numpy.ndarray
            NxN matrix (2D array) of pairwise angular separations in radians.
        
        Notes
        -----
        The spherical distance formula used is:
        
            cos(theta) = sin(dec1) * sin(dec2)
                        + cos(dec1) * cos(dec2) * cos(ra1 - ra2)
                        
        where (ra1, dec1) and (ra2, dec2) are coordinate pairs in radians.

        """
        # Reshape for broadcasting
        ra1 = ra_rad[:, None]
        ra2 = ra_rad[None, :]
        dec1 = dec_rad[:, None]
        dec2 = dec_rad[None, :]

        # Spherical distance formula:
        #   cos(theta) = sin(dec1)*sin(dec2) + cos(dec1)*cos(dec2)*cos(ra1 - ra2)
        cos_sep = (np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))
        
        # Clip values to avoid floating-point errors outside [-1, 1] when taking arccos
        cos_sep = np.clip(cos_sep, -1.0, 1.0)
        
        # Compute separation in radians
        sep_rad = np.arccos(cos_sep)
        
        return sep_rad


    @staticmethod
    def post_process_residuals(residuals_data: pd.DataFrame) -> np.ndarray:
        """Post-process residuals data to extract the non-NaN residuals and their indices.

        Parameters
        ----------
        residuals_data : pd.DataFrame
            DataFrame containing residuals data for multiple pulsars.

        Returns
        -------
        np.ndarray
            A 2D array containing the non-NaN residuals and their corresponding pulsar indices.

        """
        #1. Select columns that start with 'residuals_'
        residual_columns = [col for col in residuals_data.columns if col.startswith('residuals_')]
        
        
        #2. Create a mask to identify non-NaN values in the selected columns. Mask is a DataFrame of booleans.
        mask = ~residuals_data[residual_columns].isna()

        #3. For each row, find the *position* of the True (non-NaN) column
        ##  np.argmax returns the index of the first True in each row.
        ## idx is a NumPy array of shape (Nrows,)
        idx = np.argmax(mask.values, axis=1)  

        #4. Extract the numeric part of the column name. 
        ##  e.g. "residuals_3" -> 3
        subscript_list = [int(col.split('_')[-1]) for col in residual_columns]

        #5. Map each row’s True position to its "residuals_i" subscript
        subscripts = np.array(subscript_list)[idx]

        #6. Index to get the non-NaN values
        row_indices = np.arange(len(residuals_data))  # 0,1,2,... up to len(df)-1
        residuals_values = residuals_data[residual_columns].values[row_indices, idx]

        # 7. Finally, stack them into a 2D array:
        #   - Column 0: the non-NaN residual value
        #   - Column 1: the subscript i
        result = np.column_stack([residuals_data['toas'].values,residuals_values, subscripts])

        return result


    @classmethod
    def read_par_tim(cls, par_file: str, tim_file: str, **kwargs) -> "LoadWidebandPulsarData":
        """Load the pulsar data from the specified parameter and timing files.

        Parameters
        ----------
        par_file : str
            Path to the parameter file.
        tim_file : str
            Path to the timing file.
        **kwargs : dict
            Additional keyword arguments to pass to enterprise.pulsar.Pulsar.

        Returns
        -------
        LoadWidebandPulsarData
            An instance of LoadWidebandPulsarData initialized with the loaded data.

        """
        try:
            pulsar_object = EnterprisePulsar(par_file, tim_file, **kwargs)
            return cls(pulsar_object)
        except Exception as e:
            print(f"Error loading pulsar data from {par_file} and {tim_file}: {e}")
            raise

    @classmethod
    def read_multiple_par_tim(cls, par_files: list[str], 
                              tim_files: list[str], 
                              max_files: int | None = None, 
                              **kwargs) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """Load multiple par/tim file pairs.
         
        Merge their TOAs/residuals into a DataFrame,and collect metadata (pulsar name, RA, DEC, etc.) in a second DataFrame.
        Also, compute the angular separation matrix between all loaded pulsars.

        Parameters
        ----------
        par_files : list of str
            List of parameter file paths.
        tim_files : list of str
            List of timing file paths.
        max_files : int, optional
            If provided, only the first `max_files` pairs will be processed.
        **kwargs : dict
            Additional keyword arguments to pass to enterprise.pulsar.Pulsar.

        Returns
        -------
        merged_df : pd.DataFrame
            A DataFrame with a "toas" column and additional columns for each pulsar's
            residuals (e.g., 'residuals_0', 'residuals_1', ...).
        meta_df : pd.DataFrame
            A DataFrame containing per-pulsar metadata such as name, RA, DEC, and
            the dimension of the design matrix.
        angle_matrix : np.ndarray
            A 2D array (N × N) containing pairwise angular separations (in radians)
            between the loaded pulsars.

        Notes
        -----
        For standard RA/DEC in radians:
            - RA is treated as the azimuth (φ).
            - DEC is converted to co-latitude: θ = π/2 − DEC.

        """
        # Combine the par and tim files into pairs; optionally limit to max_files.
        file_pairs = list(zip(par_files, tim_files))
        if max_files is not None:
            file_pairs = file_pairs[:max_files]

        dfs = []      # List to hold individual pulsar TOA/residual DataFrames.
        dfs_meta = [] # List to hold individual pulsar metadata DataFrames.

        for i, (par_file, tim_file) in enumerate(file_pairs):
            psr = cls.read_par_tim(par_file, tim_file,**kwargs)

            # DataFrame for TOAs and residuals for this pulsar.
            df = pd.DataFrame({
                "toas": psr.toas,
                f"residuals_{i}": psr.residuals
            })

            # DataFrame for metadata for this pulsar.
            df_meta = pd.DataFrame({
                "name": [psr.name],
                "dim_M": [psr.M_matrix.shape[-1]],
                "RA": [psr.RA],
                "DEC": [psr.DEC]
            })

            dfs.append(df)
            dfs_meta.append(df_meta)

        # Merge all individual pulsar DataFrames on 'toas' using an outer merge.
        merged_df = reduce(lambda left, right: pd.merge(left, right, on="toas", how="outer"), dfs)
        meta_df = pd.concat(dfs_meta, ignore_index=True)

        return merged_df, meta_df
    

