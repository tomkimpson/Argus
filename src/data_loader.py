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

        for d in dir(ds_psr):
            print(d)
        
        print("here is the noise dict")
        print(ds_psr.noisedict)

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

        # Convert RA and DEC to numpy arrays of type float.
        ra = meta_df["RA"].to_numpy(dtype=float)
        dec = meta_df["DEC"].to_numpy(dtype=float)

        # Local helper function to compute unit vectors from spherical coordinates.
        # Here, RA is treated as the azimuth (φ) and DEC is converted to co-latitude (θ = π/2 − DEC).
        def _unit_vector(θ, φ):
            qx = np.sin(θ) * np.cos(φ)
            qy = np.sin(θ) * np.sin(φ)
            qz = np.cos(θ)
            return np.column_stack([qx, qy, qz])

        # Convert DEC (declination) to co-latitude: θ = π/2 − DEC.
        q = _unit_vector(np.pi / 2.0 - dec, ra)
        Npsr = len(meta_df)

        # Compute the pairwise angular separation matrix.
        angle_matrix = np.zeros((Npsr, Npsr), dtype=float)
        # Compute only for i < j and mirror the values, since the matrix is symmetric.
        for i in range(Npsr):
            for j in range(i + 1, Npsr):
                dot_product = np.dot(q[i, :], q[j, :])
                angle = np.arccos(dot_product)
                angle_matrix[i, j] = angle
                angle_matrix[j, i] = angle

        return merged_df, meta_df, angle_matrix
