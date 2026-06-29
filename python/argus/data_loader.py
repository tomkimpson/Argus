"""Module for loading pulsar data."""

import glob
import json
import os
import types

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.feather

from argus import gravitational_waves


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
        self.toas = (
            ds_psr.toas
        )  # units of seconds, https://github.com/nanograv/enterprise/blob/master/enterprise/pulsar.py#L201
        self.toaerrs = (
            ds_psr.toaerrs
        )  # units of seconds, https://github.com/nanograv/enterprise/blob/master/enterprise/pulsar.py#L216
        self.residuals = (
            ds_psr.residuals
        )  # units of seconds, https://github.com/nanograv/enterprise/blob/master/enterprise/pulsar.py#L211
        self.fitpars = ds_psr.fitpars
        self.M_matrix = ds_psr.Mmat
        self.name = ds_psr.name
        self.RA = ds_psr._raj
        self.DEC = ds_psr._decj

        # Pulsar distance in kpc: (distance, uncertainty)
        # enterprise returns (1.0, 0.2) as default if not found in par file
        pdist = ds_psr._pdist
        self.distance_kpc = pdist[0] if pdist is not None else 1.0
        self.distance_err_kpc = pdist[1] if pdist is not None else 0.2

        # Scale the M matrix columns to have unit norm
        col_scales = np.sqrt(np.sum(self.M_matrix**2, axis=0))
        self.M_scaled = self.M_matrix / col_scales

        # Compute the covariance matrix of the residuals
        Ninv = np.diag(1.0 / self.toaerrs**2)
        MtNinvM = self.M_scaled.T @ Ninv @ self.M_scaled
        self.P_eps = np.linalg.inv(MtNinvM)

        # Compute differences between consecutive TOAs and propagate errors.
        self.toa_diffs = np.diff(self.toas)
        self.toa_diff_errors = np.sqrt(self.toaerrs[1:] ** 2 + self.toaerrs[:-1] ** 2)

    @staticmethod
    def process_pulsar_residuals_by_epoch(list_of_dfs):
        """Post-process the residuals from a list of pulsar DataFrames that share the same (or very similar)time sampling.

        This function takes a list of DataFrames, each expected to have the same shape
        and contain 'toas', 'residuals', and 'error' columns. It assumes a perfect 1:1
        correspondence between rows across all DataFrames, meaning each row index represents
        the same observation epoch across all DataFrames. It calculates the average 'toas'
        across all DataFrames and collects the 'residuals' and 'error' values into matrices.

        IMPORTANT: This function is NOT suitable for processing collections of pulsars with
        uneven sampling or different observation epochs. It requires that all DataFrames have
        identical row indices representing the same epochs.

        Args:
            list_of_dfs: A list of pandas DataFrames. Each DataFrame must have
                        the same shape and contain the columns 'toas', 'residuals',
                        and 'error'.

        Returns
        -------
            A dictionary containing three NumPy arrays:
            - 'toas': 1D array of average TOAs across all input DataFrames for each row index (shape: nrows).
            - 'residuals': 2D array where each column corresponds to the 'residuals' from one input DataFrame (shape: nrows x num_dfs).
            - 'errors': 2D array where each column corresponds to the 'error' from one input DataFrame (shape: nrows x num_dfs).

        Raises
        ------
            ValueError: If the input list `list_of_dfs` is empty.
            ValueError: If not all DataFrames in the list have the same shape.
            ValueError: If any DataFrame is missing one of the required columns
                        ('toas', 'residuals', 'error').
        """
        # --- Input Validation ---
        if not list_of_dfs:
            raise ValueError("Input list of DataFrames cannot be empty.")

        # Check shapes consistency and required columns in one pass
        required_cols = ["toas", "residuals", "error"]
        first_shape = list_of_dfs[0].shape

        for i, df in enumerate(list_of_dfs):
            if df.shape != first_shape:
                raise ValueError(
                    f"DataFrame at index {i} has shape {df.shape}, expected {first_shape}"
                )

            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"DataFrame at index {i} is missing required columns: {missing_cols}"
                )

        # Process all columns at once instead of separate operations
        result_arrays = []

        for col in required_cols:
            # Extract the column from each DataFrame
            series_list = [df[col] for df in list_of_dfs]
            combined_df = pd.concat(series_list, axis=1)

            # For 'toas', compute the mean; for others, just convert to numpy
            if col == "toas":
                result_arrays.append(combined_df.mean(axis=1).to_numpy())
            else:
                result_arrays.append(combined_df.to_numpy())

        # Return as a dictionary instead of tuple
        return {
            "toas": result_arrays[0],  # average TOAs array
            "residuals": result_arrays[1],  # residuals array
            "errors": result_arrays[2],  # errors array
        }

    @staticmethod
    def process_pulsar_residuals_per_pulsar(list_of_dfs):
        """Process residuals keeping per-pulsar observation times (no epoch alignment).

        Each pulsar retains its own observation times, residuals, and errors.
        This is used for CW mode where pulsars are processed independently.

        Args:
            list_of_dfs: A list of pandas DataFrames, each containing
                        'toas', 'residuals', and 'error' columns.

        Returns
        -------
            A dictionary containing:
            - 'toas': list of 1D numpy arrays, one per pulsar.
            - 'residuals': list of 1D numpy arrays, one per pulsar.
            - 'errors': list of 1D numpy arrays, one per pulsar.
            - 'n_obs': 1D numpy array of observation counts per pulsar.

        Raises
        ------
            ValueError: If the input list is empty.
            ValueError: If any DataFrame is missing required columns.
        """
        if not list_of_dfs:
            raise ValueError("Input list of DataFrames cannot be empty.")

        required_cols = ["toas", "residuals", "error"]
        toas_list = []
        residuals_list = []
        errors_list = []
        n_obs_list = []

        for i, df in enumerate(list_of_dfs):
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"DataFrame at index {i} is missing required columns: {missing_cols}"
                )

            toas_list.append(df["toas"].to_numpy())
            residuals_list.append(df["residuals"].to_numpy())
            errors_list.append(df["error"].to_numpy())
            n_obs_list.append(len(df))

        return {
            "toas": toas_list,
            "residuals": residuals_list,
            "errors": errors_list,
            "n_obs": np.array(n_obs_list),
        }

    @staticmethod
    def get_processed_residuals(directory, excluded_psrs=[], mode="gwb"):
        """Get the processed residuals from the data.

        Parameters
        ----------
        directory : str
            Path to data directory containing .par and .tim files.
        excluded_psrs : list of str
            Pulsar names to exclude from the analysis.
        mode : str
            Signal model mode: 'gwb' for epoch-aligned processing (default),
            'cw' for per-pulsar processing.

        Returns
        -------
        dict
            A dictionary containing:
            - 'processed_residuals': epoch-aligned dict (gwb) or per-pulsar dict (cw)
            - 'metadata': DataFrame containing pulsar metadata
            - 'design_matrices': list of design matrices for each pulsar
            - 'parameter_covariances': list of parameter covariance matrices
            - 'hd_correlation': Hellings-Downs correlation matrix (gwb) or None (cw)
        """
        # Validate directory path
        if not directory:
            raise ValueError("Data directory path cannot be empty")

        if not os.path.exists(directory):
            raise FileNotFoundError(f"Data directory does not exist: {directory}")

        if not os.path.isdir(directory):
            raise NotADirectoryError(f"Path is not a directory: {directory}")

        # Prefer the Argus feather cache (no enterprise needed); fall back to par/tim.
        feather_files = sorted(glob.glob(os.path.join(directory, "*.feather")))
        # Exclude e.g. J1640+2224, whose exponent is too small for the OU process.
        feather_files = [
            f for f in feather_files if not any(psr in f for psr in excluded_psrs)
        ]

        if feather_files:
            print(
                f"Getting the data. Loading {len(feather_files)} pulsars "
                f"from feather cache in {directory}"
            )
            (
                pulsar_residuals,
                pulsar_metadata,
                pulsar_design_matrices,
                P_eps_matrices,
            ) = LoadWidebandPulsarData.read_multiple_feather(feather_files)
        else:
            # Get all .par and .tim files in the directory
            par_files = sorted(glob.glob(os.path.join(directory, "*.par")))
            tim_files = sorted(glob.glob(os.path.join(directory, "*.tim")))

            if len(par_files) == 0 and len(tim_files) == 0:
                raise FileNotFoundError(
                    f"No .feather, .par or .tim files found in directory: {directory}"
                )

            if len(par_files) != len(tim_files):
                raise ValueError(
                    f"Mismatch between .par ({len(par_files)}) and .tim ({len(tim_files)}) file counts in {directory}"
                )

            par_files = [
                f for f in par_files if not any(psr in f for psr in excluded_psrs)
            ]
            tim_files = [
                f for f in tim_files if not any(psr in f for psr in excluded_psrs)
            ]

            # Get the data
            print(
                f"Getting the data. Loading {len(par_files)} pulsars from {directory}"
            )
            (
                pulsar_residuals,
                pulsar_metadata,
                pulsar_design_matrices,
                P_eps_matrices,
            ) = LoadWidebandPulsarData.read_multiple_par_tim(par_files, tim_files)

        if mode == "gwb":
            # GWB mode: epoch-aligned processing with Hellings-Downs correlation
            ra = pulsar_metadata["RA"].to_numpy(dtype=float)
            dec = pulsar_metadata["DEC"].to_numpy(dtype=float)
            angular_separation_matrix = gravitational_waves.pairwise_angular_separation(
                ra, dec
            )
            hd_correlation_matrix = gravitational_waves.hellings_downs(
                angular_separation_matrix
            )

            processed_pulsar_residuals = (
                LoadWidebandPulsarData.process_pulsar_residuals_by_epoch(
                    pulsar_residuals
                )
            )

            return {
                "processed_residuals": processed_pulsar_residuals,
                "metadata": pulsar_metadata,
                "design_matrices": pulsar_design_matrices,
                "parameter_covariances": P_eps_matrices,
                "hd_correlation": hd_correlation_matrix,
            }
        elif mode == "cw":
            # CW mode: per-pulsar processing, no HD correlation needed
            processed_pulsar_residuals = (
                LoadWidebandPulsarData.process_pulsar_residuals_per_pulsar(
                    pulsar_residuals
                )
            )

            return {
                "processed_residuals": processed_pulsar_residuals,
                "metadata": pulsar_metadata,
                "design_matrices": pulsar_design_matrices,
                "parameter_covariances": P_eps_matrices,
                "hd_correlation": None,
            }
        else:
            raise ValueError(
                f"Unknown signal model mode: '{mode}'. Must be 'gwb' or 'cw'."
            )

    @staticmethod
    def get_par_value(filename: str, parameter: str) -> float | None:
        """Get the value of a parameter from a parameter file.

        Args:
            filename: Path to the parameter file
            parameter: Name of the parameter to retrieve

        Returns
        -------
            The parameter value as a float, or None if not found

        Raises
        ------
            FileNotFoundError: If the parameter file doesn't exist
            ValueError: If the parameter value cannot be converted to float
        """
        try:
            with open(filename, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts or parts[0].startswith("#"):
                        continue
                    if parts[0] == parameter:
                        try:
                            return float(parts[1])
                        except (IndexError, ValueError) as e:
                            raise ValueError(
                                f"Invalid parameter value for {parameter}: {parts[1]}"
                            ) from e
            return None
        except FileNotFoundError:
            raise FileNotFoundError(f"Parameter file not found: {filename}")

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
            # Lazy import: enterprise is a data-prep-only dependency. The package
            # imports (and the feather runtime path) work without it installed.
            from enterprise.pulsar import Pulsar as EnterprisePulsar

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
    ) -> tuple[list[pd.DataFrame], pd.DataFrame, list[np.ndarray], list[np.ndarray]]:
        """Load multiple par/tim file pairs.

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
        pulsar_data_frames : list of pd.DataFrame
            List of DataFrames, each containing TOAs, residuals, and errors for a single pulsar.
            Each DataFrame has columns: 'toas', 'residuals', 'error'.
        metadata_combined : pd.DataFrame
            A DataFrame containing per-pulsar metadata such as name, RA, DEC,
            the dimension of the design matrix, and F0 (pulsar frequency).
        design_matrices : list of np.ndarray
            List of scaled design matrices (M_scaled) for each pulsar.
        parameter_covariances : list of np.ndarray
            List of parameter covariance matrices (P_eps) for each pulsar.

        Notes
        -----
        For standard RA/DEC in radians:
            - RA is treated as the azimuth (φ).
            - DEC is converted to co-latitude: θ = π/2 − DEC.
        """
        # Input validation
        if len(par_files) != len(tim_files):
            raise ValueError(
                f"Number of par files ({len(par_files)}) must match number of tim files ({len(tim_files)})"
            )

        # Combine the par and tim files into pairs; optionally limit to max_files.
        file_pairs = list(zip(par_files, tim_files))
        if max_files is not None:
            file_pairs = file_pairs[:max_files]

        pulsar_data_frames = (
            []
        )  # List to hold individual pulsar TOA/residual DataFrames
        pulsar_metadata_frames = (
            []
        )  # List to hold individual pulsar metadata DataFrames
        design_matrices = []  # List to hold individual pulsar design matrices
        parameter_covariances = (
            []
        )  # List to hold individual pulsar parameter covariance matrices

        for i, (par_file, tim_file) in enumerate(file_pairs):
            try:
                psr = cls.read_par_tim(par_file, tim_file, **kwargs)

                f0 = cls.get_par_value(par_file, "F0")
                print(f"PSR: {psr.name}, F0: {f0}, # TOAs: {len(psr.toas)}")

                # DataFrame for TOAs and residuals for this pulsar
                pulsar_df = pd.DataFrame(
                    {"toas": psr.toas, "residuals": psr.residuals, "error": psr.toaerrs}
                )

                # DataFrame for metadata for this pulsar
                metadata_df = pd.DataFrame(
                    {
                        "name": [psr.name],
                        "dim_M": [psr.M_matrix.shape[-1]],
                        "RA": [psr.RA],
                        "DEC": [psr.DEC],
                        "F0": [f0],
                        "distance_kpc": [psr.distance_kpc],
                        "par_file": [par_file],
                        "tim_file": [tim_file],
                    }
                )

                pulsar_data_frames.append(pulsar_df)
                pulsar_metadata_frames.append(metadata_df)
                design_matrices.append(psr.M_scaled)
                parameter_covariances.append(psr.P_eps)

            except Exception as e:
                print(
                    f"Error processing pulsar pair {i+1}/{len(file_pairs)} ({par_file}, {tim_file}): {str(e)}"
                )
                # Optionally: raise or continue based on preference

        if not pulsar_data_frames:
            raise ValueError("No pulsar data was successfully loaded")

        metadata_combined = pd.concat(pulsar_metadata_frames, ignore_index=True)

        return (
            pulsar_data_frames,
            metadata_combined,
            design_matrices,
            parameter_covariances,
        )

    # ------------------------------------------------------------------
    # Feather cache (Argus-native; no enterprise/discovery dependency)
    # ------------------------------------------------------------------

    def save_feather(self, path: str, F0: float | None = None) -> None:
        """Serialize the raw pulsar inputs to an Argus feather cache file.

        Stores exactly the quantities consumed by :meth:`__init__` and
        :meth:`read_multiple_feather`, so a later :meth:`read_feather` reconstructs
        an identical object without needing enterprise/tempo2. The on-disk layout
        (per-TOA columns + a JSON ``schema.metadata`` blob) mirrors discovery's
        feather convention but is written with pyarrow directly.

        Parameters
        ----------
        path : str
            Output ``.feather`` path.
        F0 : float, optional
            Pulsar spin frequency (Hz). If None, falls back to ``self.F0`` when
            present. Stored in metadata for downstream use.
        """
        M = np.asarray(self.M_matrix)
        n_cols = M.shape[1]

        pydict = {
            "toas": np.asarray(self.toas),
            "toaerrs": np.asarray(self.toaerrs),
            "residuals": np.asarray(self.residuals),
        }
        for i in range(n_cols):
            pydict[f"Mmat_{i}"] = M[:, i]

        if F0 is None:
            F0 = getattr(self, "F0", None)

        fitpars = self.fitpars
        if isinstance(fitpars, np.ndarray):
            fitpars = fitpars.tolist()

        meta = {
            "name": self.name,
            "RA": float(self.RA),
            "DEC": float(self.DEC),
            "distance_kpc": float(self.distance_kpc),
            "distance_err_kpc": float(self.distance_err_kpc),
            "F0": None if F0 is None else float(F0),
            "fitpars": list(fitpars) if fitpars is not None else None,
            "n_mmat_cols": int(n_cols),
        }

        table = pyarrow.Table.from_pydict(pydict, metadata={"json": json.dumps(meta)})
        pyarrow.feather.write_feather(table, path)

    @classmethod
    def read_feather(cls, path: str) -> "LoadWidebandPulsarData":
        """Load a pulsar from an Argus feather cache file.

        Reconstructs the raw inputs and feeds them through :meth:`__init__`
        unchanged, so all derived quantities (``M_scaled``, ``P_eps`` ...) are
        identical to a direct enterprise load.

        Parameters
        ----------
        path : str
            Path to a ``.feather`` file written by :meth:`save_feather`.

        Returns
        -------
        LoadWidebandPulsarData
            An instance initialized from the cached data (with ``F0`` attached).
        """
        table = pyarrow.feather.read_table(path)
        meta = json.loads(table.schema.metadata[b"json"])

        n_cols = int(meta["n_mmat_cols"])
        if n_cols > 0:
            Mmat = np.column_stack(
                [table.column(f"Mmat_{i}").to_numpy() for i in range(n_cols)]
            )
        else:
            Mmat = np.empty((table.num_rows, 0))

        # Map the cache back onto the enterprise attribute names that __init__ reads.
        ds_psr = types.SimpleNamespace(
            toas=table.column("toas").to_numpy(),
            toaerrs=table.column("toaerrs").to_numpy(),
            residuals=table.column("residuals").to_numpy(),
            fitpars=meta.get("fitpars"),
            Mmat=Mmat,
            name=meta["name"],
            _raj=meta["RA"],
            _decj=meta["DEC"],
            _pdist=(meta["distance_kpc"], meta["distance_err_kpc"]),
        )

        obj = cls(ds_psr)
        obj.F0 = meta.get("F0")
        return obj

    @classmethod
    def read_multiple_feather(
        cls,
        feather_files: list[str],
        max_files: int | None = None,
    ) -> tuple[list[pd.DataFrame], pd.DataFrame, list[np.ndarray], list[np.ndarray]]:
        """Load multiple pulsars from Argus feather cache files.

        Mirrors :meth:`read_multiple_par_tim` but reads from feather caches (no
        enterprise) and sources ``F0`` from feather metadata instead of the par
        file.

        Parameters
        ----------
        feather_files : list of str
            List of ``.feather`` cache file paths.
        max_files : int, optional
            If provided, only the first ``max_files`` files are processed.

        Returns
        -------
        Same 4-tuple as :meth:`read_multiple_par_tim`:
        (pulsar_data_frames, metadata_combined, design_matrices, parameter_covariances).
        """
        if max_files is not None:
            feather_files = feather_files[:max_files]

        pulsar_data_frames = []
        pulsar_metadata_frames = []
        design_matrices = []
        parameter_covariances = []

        for i, feather_file in enumerate(feather_files):
            try:
                psr = cls.read_feather(feather_file)
                f0 = getattr(psr, "F0", None)
                print(f"PSR: {psr.name}, F0: {f0}, # TOAs: {len(psr.toas)}")

                pulsar_df = pd.DataFrame(
                    {"toas": psr.toas, "residuals": psr.residuals, "error": psr.toaerrs}
                )

                metadata_df = pd.DataFrame(
                    {
                        "name": [psr.name],
                        "dim_M": [psr.M_matrix.shape[-1]],
                        "RA": [psr.RA],
                        "DEC": [psr.DEC],
                        "F0": [f0],
                        "distance_kpc": [psr.distance_kpc],
                        "par_file": [feather_file],
                        "tim_file": [None],
                    }
                )

                pulsar_data_frames.append(pulsar_df)
                pulsar_metadata_frames.append(metadata_df)
                design_matrices.append(psr.M_scaled)
                parameter_covariances.append(psr.P_eps)

            except Exception as e:
                print(
                    f"Error processing feather {i+1}/{len(feather_files)} ({feather_file}): {str(e)}"
                )

        if not pulsar_data_frames:
            raise ValueError("No pulsar data was successfully loaded")

        metadata_combined = pd.concat(pulsar_metadata_frames, ignore_index=True)

        return (
            pulsar_data_frames,
            metadata_combined,
            design_matrices,
            parameter_covariances,
        )
