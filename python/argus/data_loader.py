"""Module for loading pulsar data."""

import numpy as np
import pandas as pd
from functools import reduce
from enterprise.pulsar import Pulsar as EnterprisePulsar


def get_par_value(filename, parameter):
    """Get the value of a parameter from a parameter file.
    
    TK note: It feels like there should be a better way to do this, just using the enterprise.pulsar.Pulsar object.
    However, I have not been able to figure out how to do this yet.
    We require F0 as part of the measurement model. 
    I take it as a known parameter.
    """
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0].startswith('#'):  # skip empty or commented lines
                continue
            if parts[0] == parameter:
                return float(parts[1])  # assumes the value is the second item
    return None  # if the parameter isn't found



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
        self.toas      = ds_psr.toas      #units of seconds, https://github.com/nanograv/enterprise/blob/master/enterprise/pulsar.py#L201
        self.toaerrs   = ds_psr.toaerrs   #units of seconds, https://github.com/nanograv/enterprise/blob/master/enterprise/pulsar.py#L216
        self.residuals = ds_psr.residuals #units of seconds, https://github.com/nanograv/enterprise/blob/master/enterprise/pulsar.py#L211
        self.fitpars   = ds_psr.fitpars
        self.M_matrix  = ds_psr.Mmat
        self.name      = ds_psr.name
        self.RA        = ds_psr._raj
        self.DEC       = ds_psr._decj

      



        #print(np.max(self.M_matrix,axis=0))
        #print(np.min(self.M_matrix,axis=0))
        # Scale the M matrix columns to have unit norm
        col_scales = np.sqrt(np.sum(self.M_matrix**2, axis=0))
        self.M_scaled = self.M_matrix / col_scales


        # Compute the covariance matrix of the residuals
        #print("Computing the covariance matrix of the residuals")
        Ninv = np.diag(1.0 / self.toaerrs**2)
        MtNinvM = self.M_scaled.T @ Ninv @ self.M_scaled
        self.P_eps = np.linalg.inv(MtNinvM)
 
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
        cos_sep = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(
            ra1 - ra2
        )

        # Clip values to avoid floating-point errors outside [-1, 1] when taking arccos
        cos_sep = np.clip(cos_sep, -1.0, 1.0)

        # Compute separation in radians
        sep_rad = np.arccos(cos_sep)

        return sep_rad

    @staticmethod
    def post_process_residuals(list_of_dfs):
        """Post-process the residuals from a list of DataFrames.

        This function takes a list of DataFrames, each expected to have the same shape
        and contain 'toas', 'residuals', and 'error' columns. It calculates the
        average 'toas' across all DataFrames and collects the 'residuals' and 'error'
        values into matrices.

        Args:
            list_of_dfs: A list of pandas DataFrames. Each DataFrame must have
                         the same shape and contain the columns 'toas', 'residuals',
                         and 'error'.

        Returns:
            A tuple containing three NumPy arrays:
            - average_toas_array: 1D array of average TOAs across all input
                                  DataFrames for each row index (shape: nrows).
            - residuals_array: 2D array where each column corresponds to the
                               'residuals' from one input DataFrame
                               (shape: nrows x num_dfs).
            - errors_array: 2D array where each column corresponds to the
                            'error' from one input DataFrame
                            (shape: nrows x num_dfs).

        Raises:
            ValueError: If the input list `list_of_dfs` is empty.
            ValueError: If not all DataFrames in the list have the same shape.
            ValueError: If any DataFrame is missing one of the required columns
                        ('toas', 'residuals', 'error').
        """
        # --- Input Validation ---
        if not list_of_dfs:
            raise ValueError("Input list of DataFrames cannot be empty.")

        # Check shapes consistency
        first_shape = list_of_dfs[0].shape
        if not all(df.shape == first_shape for df in list_of_dfs):
             raise ValueError("All DataFrames in the list must have the same shape.")

        # Check required columns
        required_cols = ["toas", "residuals", "error"]
        for i, df in enumerate(list_of_dfs):
             missing_cols = [col for col in required_cols if col not in df.columns]
             if missing_cols:
                 raise ValueError(f"DataFrame at index {i} is missing required columns: {missing_cols}")
        # --- End Input Validation ---

        # 1. Average TOAs array
        toas_series_list = [df['toas'] for df in list_of_dfs]
        toas_df = pd.concat(toas_series_list, axis=1)
        average_toas_series = toas_df.mean(axis=1)
        average_toas_array = average_toas_series.to_numpy()

        # 2. Combined Residuals array (nrows x Num dfs)
        residuals_series_list = [df['residuals'] for df in list_of_dfs]
        residuals_df = pd.concat(residuals_series_list, axis=1)
        residuals_array = residuals_df.to_numpy()

        # 3. Combined Errors array (nrows x Num dfs)
        errors_series_list = [df['error'] for df in list_of_dfs]
        errors_df = pd.concat(errors_series_list, axis=1)
        errors_array = errors_df.to_numpy()

        return [average_toas_array, residuals_array, errors_array]


    @classmethod
    def read_par_tim(
        cls, par_file: str, tim_file: str, **kwargs
    ) -> "LoadWidebandPulsarData":
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
    def read_multiple_par_tim(
        cls,
        par_files: list[str],
        tim_files: list[str],
        max_files: int | None = None,
        **kwargs,
    ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
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

        dfs              = []  # List to hold individual pulsar TOA/residual DataFrames.
        dfs_meta         = []  # List to hold individual pulsar metadata DataFrames.
        np_arrays_design = []  # List to hold individual pulsar design matrix DataFrames.
        np_arrays_P_eps  = []  # List to hold individual pulsar design matrix DataFrames.
        
        for i, (par_file, tim_file) in enumerate(file_pairs):
            psr = cls.read_par_tim(par_file, tim_file, **kwargs)

            f0 = get_par_value(par_file, 'F0')
            print(f"PSR: {psr.name}, F0: {f0}, # TOAs: {len(psr.toas)}")

            # DataFrame for TOAs and residuals for this pulsar.
            df = pd.DataFrame(
                {
                    "toas": psr.toas,
                    f"residuals": psr.residuals,
                    f"error": psr.toaerrs
                }
            )

            # DataFrame for metadata for this pulsar.
            df_meta = pd.DataFrame(
                {
                    "name": [psr.name],
                    "dim_M": [psr.M_matrix.shape[-1]],
                    "RA": [psr.RA],
                    "DEC": [psr.DEC],
                    "F0": [f0]
                }
            )


            dfs.append(df)
            dfs_meta.append(df_meta)    
            np_arrays_design.append(psr.M_scaled)
            np_arrays_P_eps.append(psr.P_eps)

        meta_df = pd.concat(dfs_meta, ignore_index=True)
    
        return dfs, meta_df, np_arrays_design, np_arrays_P_eps
