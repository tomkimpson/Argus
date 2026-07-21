"""Unit tests for data_loader module."""

import os
import types

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from argus import data_loader

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FIXTURE_FEATHER = os.path.join(DATA_DIR, "test_pulsar.feather")


def _synthetic_pulsar(seed=1234, n=12):
    """Build a deterministic LoadWidebandPulsarData from synthetic inputs."""
    rng = np.random.default_rng(seed)
    ds_psr = types.SimpleNamespace(
        name="J9999+9999",
        toas=np.linspace(53000.0, 54000.0, n) * 86400.0,
        toaerrs=np.full(n, 1e-7),
        residuals=rng.standard_normal(n) * 1e-6,
        fitpars=["Offset", "F0", "F1", "RAJ", "DECJ"],
        Mmat=rng.standard_normal((n, 5)),
        _raj=0.75,
        _decj=-0.25,
        _pdist=(1.5, 0.3),
    )
    return data_loader.LoadWidebandPulsarData(ds_psr)


class TestLoadWidebandPulsarData:
    """Tests for LoadWidebandPulsarData class."""

    def test_initialization(self, mock_enterprise_pulsar):
        """Test basic initialization."""
        psr_data = data_loader.LoadWidebandPulsarData(mock_enterprise_pulsar)

        assert psr_data.name == "PSR_J0030+0451"
        assert len(psr_data.toas) == 10
        assert len(psr_data.toaerrs) == 10
        assert len(psr_data.residuals) == 10

        # Pulsar distance is read from enterprise's _pdist = (distance, uncertainty)
        assert psr_data.distance_kpc == 1.0
        assert psr_data.distance_err_kpc == 0.2

    def test_m_matrix_scaling(self, mock_enterprise_pulsar):
        """Test that M matrix is scaled to unit norm."""
        psr_data = data_loader.LoadWidebandPulsarData(mock_enterprise_pulsar)

        # Check that M_scaled has unit norm columns
        col_norms = np.sqrt(np.sum(psr_data.M_scaled**2, axis=0))
        assert np.allclose(col_norms, 1.0)

    def test_toa_differences(self, mock_enterprise_pulsar):
        """Test TOA differences calculation."""
        psr_data = data_loader.LoadWidebandPulsarData(mock_enterprise_pulsar)

        # Should have one less diff than TOAs
        assert len(psr_data.toa_diffs) == len(psr_data.toas) - 1

        # Check values
        expected_diffs = np.diff(psr_data.toas)
        assert np.allclose(psr_data.toa_diffs, expected_diffs)

    def test_toa_diff_errors(self, mock_enterprise_pulsar):
        """Test TOA difference error propagation."""
        psr_data = data_loader.LoadWidebandPulsarData(mock_enterprise_pulsar)

        # Error should be sqrt of sum of squares
        expected_errors = np.sqrt(
            psr_data.toaerrs[1:] ** 2 + psr_data.toaerrs[:-1] ** 2
        )
        assert np.allclose(psr_data.toa_diff_errors, expected_errors)

    def test_covariance_matrix_calculation(self, mock_enterprise_pulsar):
        """Test parameter covariance matrix (P_eps) calculation."""
        psr_data = data_loader.LoadWidebandPulsarData(mock_enterprise_pulsar)

        # P_eps should be square matrix with size = number of timing parameters
        n_params = psr_data.M_matrix.shape[1]
        assert psr_data.P_eps.shape == (n_params, n_params)

        # Should be positive definite
        eigenvalues = np.linalg.eigvalsh(psr_data.P_eps)
        assert np.all(eigenvalues > 0)


class TestProcessPulsarResidualsByEpoch:
    """Tests for process_pulsar_residuals_by_epoch static method."""

    def test_basic_processing(self):
        """Test basic residual processing."""
        # Create sample dataframes
        df1 = pd.DataFrame(
            {
                "toas": [100.0, 200.0, 300.0],
                "residuals": [1e-6, 2e-6, 3e-6],
                "error": [1e-7, 1e-7, 1e-7],
            }
        )
        df2 = pd.DataFrame(
            {
                "toas": [101.0, 201.0, 301.0],
                "residuals": [1.5e-6, 2.5e-6, 3.5e-6],
                "error": [1.5e-7, 1.5e-7, 1.5e-7],
            }
        )

        result = data_loader.LoadWidebandPulsarData.process_pulsar_residuals_by_epoch(
            [df1, df2]
        )

        assert "toas" in result
        assert "residuals" in result
        assert "errors" in result

        # TOAs should be averaged
        expected_toas = np.array([100.5, 200.5, 300.5])
        assert np.allclose(result["toas"], expected_toas)

        # Residuals and errors should be matrices
        assert result["residuals"].shape == (3, 2)
        assert result["errors"].shape == (3, 2)

    def test_empty_list_error(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            data_loader.LoadWidebandPulsarData.process_pulsar_residuals_by_epoch([])

    def test_shape_mismatch_error(self):
        """Test that mismatched shapes raise ValueError."""
        df1 = pd.DataFrame(
            {"toas": [100.0, 200.0], "residuals": [1e-6, 2e-6], "error": [1e-7, 1e-7]}
        )
        df2 = pd.DataFrame(
            {
                "toas": [101.0, 201.0, 301.0],
                "residuals": [1.5e-6, 2.5e-6, 3.5e-6],
                "error": [1.5e-7, 1.5e-7, 1.5e-7],
            }
        )

        with pytest.raises(ValueError, match="shape"):
            data_loader.LoadWidebandPulsarData.process_pulsar_residuals_by_epoch(
                [df1, df2]
            )

    def test_missing_column_error(self):
        """Test that missing columns raise ValueError."""
        df1 = pd.DataFrame(
            {
                "toas": [100.0, 200.0],
                "residuals": [1e-6, 2e-6],
                # Missing 'error' column
            }
        )

        with pytest.raises(ValueError, match="missing required columns"):
            data_loader.LoadWidebandPulsarData.process_pulsar_residuals_by_epoch([df1])


class TestGetProcessedResiduals:
    """Tests for get_processed_residuals static method."""

    @patch("os.path.isdir")
    @patch("os.path.exists")
    @patch("argus.data_loader.LoadWidebandPulsarData.read_multiple_par_tim")
    @patch("argus.gravitational_waves.pairwise_angular_separation")
    @patch("argus.gravitational_waves.hellings_downs")
    @patch("glob.glob")
    def test_basic_functionality(
        self, mock_glob, mock_hd, mock_sep, mock_read, mock_exists, mock_isdir
    ):
        """Test basic get_processed_residuals functionality."""
        # Setup mocks
        mock_exists.return_value = True
        mock_isdir.return_value = True
        mock_glob.side_effect = [
            [],  # no feather files -> fall back to par/tim
            ["/data/psr1.par", "/data/psr2.par"],
            ["/data/psr1.tim", "/data/psr2.tim"],
        ]

        df1 = pd.DataFrame(
            {"toas": [100.0, 200.0], "residuals": [1e-6, 2e-6], "error": [1e-7, 1e-7]}
        )
        df2 = pd.DataFrame(
            {
                "toas": [100.0, 200.0],
                "residuals": [1.5e-6, 2.5e-6],
                "error": [1.5e-7, 1.5e-7],
            }
        )

        metadata = pd.DataFrame(
            {"name": ["PSR1", "PSR2"], "RA": [0.5, 1.0], "DEC": [0.3, 0.6]}
        )

        mock_read.return_value = (
            [df1, df2],
            metadata,
            [np.eye(5), np.eye(5)],
            [np.eye(5), np.eye(5)],
        )
        mock_sep.return_value = np.array([[0, 1], [1, 0]])
        mock_hd.return_value = np.array([[1.0, 0.5], [0.5, 1.0]])

        result = data_loader.LoadWidebandPulsarData.get_processed_residuals("/data")

        assert "processed_residuals" in result
        assert "metadata" in result
        assert "design_matrices" in result
        assert "parameter_covariances" in result
        assert "hd_correlation" in result

    def test_directory_validation_empty(self):
        """Test that empty directory raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            data_loader.LoadWidebandPulsarData.get_processed_residuals("")

    def test_directory_validation_nonexistent(self):
        """Test that nonexistent directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            data_loader.LoadWidebandPulsarData.get_processed_residuals(
                "/nonexistent/path"
            )

    @patch("os.path.isdir")
    @patch("os.path.exists")
    @patch("glob.glob")
    def test_no_files_error(self, mock_glob, mock_exists, mock_isdir):
        """Test error when no par/tim files found."""
        mock_exists.return_value = True
        mock_isdir.return_value = True
        mock_glob.side_effect = [[], [], []]

        with pytest.raises(
            FileNotFoundError, match="No .feather, .par or .tim files found"
        ):
            data_loader.LoadWidebandPulsarData.get_processed_residuals("/data")

    @patch("os.path.isdir")
    @patch("os.path.exists")
    @patch("glob.glob")
    def test_file_count_mismatch_error(self, mock_glob, mock_exists, mock_isdir):
        """Test error when par and tim file counts don't match."""
        mock_exists.return_value = True
        mock_isdir.return_value = True
        mock_glob.side_effect = [
            [],  # no feather files -> fall back to par/tim
            ["/data/psr1.par"],
            ["/data/psr1.tim", "/data/psr2.tim"],
        ]

        with pytest.raises(ValueError, match="Mismatch"):
            data_loader.LoadWidebandPulsarData.get_processed_residuals("/data")

    @patch("os.path.isdir")
    @patch("os.path.exists")
    @patch("argus.data_loader.LoadWidebandPulsarData.read_multiple_par_tim")
    @patch("glob.glob")
    def test_pulsar_exclusion(self, mock_glob, mock_read, mock_exists, mock_isdir):
        """Test that excluded pulsars are filtered out."""
        mock_exists.return_value = True
        mock_isdir.return_value = True
        mock_glob.side_effect = [
            [],  # no feather files -> fall back to par/tim
            ["/data/psr1.par", "/data/J1640+2224.par"],
            ["/data/psr1.tim", "/data/J1640+2224.tim"],
        ]

        # read_multiple_par_tim should only be called with non-excluded files
        mock_read.return_value = ([], pd.DataFrame(), [], [])

        with pytest.raises(Exception):  # Will fail due to empty return, but that's ok
            data_loader.LoadWidebandPulsarData.get_processed_residuals(
                "/data", excluded_psrs=["J1640+2224"]
            )

        # Check that J1640+2224 was filtered out
        called_par_files = mock_read.call_args[0][0]
        assert not any("J1640+2224" in f for f in called_par_files)


class TestGetParValue:
    """Tests for get_par_value static method."""

    def test_get_existing_parameter(self, tmp_path):
        """Test getting an existing parameter."""
        par_file = tmp_path / "test.par"
        par_file.write_text("""PSR J0030+0451
F0 200.12345
F1 -1.2e-15
RAJ 00:30:27.4
""")

        value = data_loader.LoadWidebandPulsarData.get_par_value(str(par_file), "F0")

        assert value == 200.12345

    def test_get_nonexistent_parameter(self, tmp_path):
        """Test getting a parameter that doesn't exist."""
        par_file = tmp_path / "test.par"
        par_file.write_text("""PSR J0030+0451
F0 200.12345
""")

        value = data_loader.LoadWidebandPulsarData.get_par_value(str(par_file), "PBDOT")

        assert value is None

    def test_file_not_found(self):
        """Test FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            data_loader.LoadWidebandPulsarData.get_par_value(
                "/nonexistent/file.par", "F0"
            )

    def test_skip_comments(self, tmp_path):
        """Test that comments are skipped."""
        par_file = tmp_path / "test.par"
        par_file.write_text("""# This is a comment
PSR J0030+0451
# F0 100.0  (commented out)
F0 200.0
""")

        value = data_loader.LoadWidebandPulsarData.get_par_value(str(par_file), "F0")

        assert value == 200.0

    def test_invalid_value_error(self, tmp_path):
        """Test ValueError for invalid parameter value."""
        par_file = tmp_path / "test.par"
        par_file.write_text("""PSR J0030+0451
F0 not_a_number
""")

        with pytest.raises(ValueError, match="Invalid parameter value"):
            data_loader.LoadWidebandPulsarData.get_par_value(str(par_file), "F0")


class TestReadMultipleParTim:
    """Tests for read_multiple_par_tim classmethod."""

    @patch("argus.data_loader.LoadWidebandPulsarData.read_par_tim")
    @patch("argus.data_loader.LoadWidebandPulsarData.get_par_value")
    def test_basic_reading(self, mock_get_par, mock_read):
        """Test basic reading of multiple par/tim pairs."""
        # Setup mock pulsar
        mock_psr = Mock()
        mock_psr.name = "PSR1"
        mock_psr.toas = np.array([100.0, 200.0])
        mock_psr.residuals = np.array([1e-6, 2e-6])
        mock_psr.toaerrs = np.array([1e-7, 1e-7])
        mock_psr.M_matrix = np.random.randn(2, 5)

        mock_read.return_value = mock_psr
        mock_get_par.return_value = 200.0

        par_files = ["/data/psr1.par"]
        tim_files = ["/data/psr1.tim"]

        result = data_loader.LoadWidebandPulsarData.read_multiple_par_tim(
            par_files, tim_files
        )

        pulsar_dfs, metadata, design_mats, cov_mats = result

        assert len(pulsar_dfs) == 1
        assert len(metadata) == 1
        assert len(design_mats) == 1
        assert len(cov_mats) == 1

    def test_file_count_mismatch(self):
        """Test error when par and tim file counts don't match."""
        with pytest.raises(ValueError, match="must match"):
            data_loader.LoadWidebandPulsarData.read_multiple_par_tim(
                ["/data/psr1.par"], ["/data/psr1.tim", "/data/psr2.tim"]
            )

    @patch("argus.data_loader.LoadWidebandPulsarData.read_par_tim")
    @patch("argus.data_loader.LoadWidebandPulsarData.get_par_value")
    def test_max_files_limit(self, mock_get_par, mock_read):
        """Test that max_files parameter limits processing."""
        mock_psr = Mock()
        mock_psr.name = "PSR1"
        mock_psr.toas = np.array([100.0])
        mock_psr.residuals = np.array([1e-6])
        mock_psr.toaerrs = np.array([1e-7])
        mock_psr.M_matrix = np.random.randn(1, 5)
        mock_psr.M_scaled = np.random.randn(1, 5)
        mock_psr.P_eps = np.eye(5)
        mock_psr.RA = 0.5
        mock_psr.DEC = 0.3

        mock_read.return_value = mock_psr
        mock_get_par.return_value = 200.0

        par_files = [f"/data/psr{i}.par" for i in range(5)]
        tim_files = [f"/data/psr{i}.tim" for i in range(5)]

        result = data_loader.LoadWidebandPulsarData.read_multiple_par_tim(
            par_files, tim_files, max_files=2
        )

        pulsar_dfs, metadata, _, _ = result

        # Should only process 2 files
        assert len(pulsar_dfs) == 2


class TestFeatherCache:
    """Tests for the Argus-native feather cache (no enterprise dependency)."""

    def test_save_read_roundtrip_bit_identical(self, tmp_path):
        """save_feather -> read_feather reproduces all arrays bit-for-bit."""
        psr = _synthetic_pulsar()
        path = str(tmp_path / "roundtrip.feather")
        psr.save_feather(path, F0=311.49)

        psr2 = data_loader.LoadWidebandPulsarData.read_feather(path)

        for attr in [
            "toas",
            "toaerrs",
            "residuals",
            "M_matrix",
            "M_scaled",
            "P_eps",
            "toa_diffs",
            "toa_diff_errors",
        ]:
            a = np.asarray(getattr(psr, attr))
            b = np.asarray(getattr(psr2, attr))
            assert np.array_equal(a, b), f"{attr} differs after round-trip"

        assert psr2.name == psr.name
        assert psr2.RA == psr.RA
        assert psr2.DEC == psr.DEC
        assert psr2.distance_kpc == psr.distance_kpc
        assert psr2.distance_err_kpc == psr.distance_err_kpc
        assert psr2.F0 == 311.49
        assert list(psr2.fitpars) == list(psr.fitpars)

    def test_mask_roundtrip_and_absent_by_default(self, tmp_path):
        """A per-epoch mask survives save->read; feathers without one read as None."""
        psr = _synthetic_pulsar(n=12)

        # No mask supplied -> no mask column, read back as None.
        path_nomask = str(tmp_path / "nomask.feather")
        psr.save_feather(path_nomask, F0=311.49)
        assert data_loader.LoadWidebandPulsarData.read_feather(path_nomask).mask is None

        # Mask supplied -> round-trips bit-for-bit.
        mask = np.ones(12)
        mask[[2, 5, 9]] = 0.0
        path_mask = str(tmp_path / "mask.feather")
        psr.save_feather(path_mask, F0=311.49, mask=mask)
        back = data_loader.LoadWidebandPulsarData.read_feather(path_mask)
        assert back.mask is not None
        assert np.array_equal(np.asarray(back.mask), mask)

    def test_epoch_alignment_collects_mask(self, tmp_path):
        """process_pulsar_residuals_by_epoch surfaces the mask as an (nepoch, Npsr) array."""
        psr = _synthetic_pulsar(n=12)
        m0 = np.ones(12)
        m0[[1, 4]] = 0.0
        m1 = np.ones(12)
        m1[[7]] = 0.0
        p0 = str(tmp_path / "A.feather")
        p1 = str(tmp_path / "B.feather")
        psr.save_feather(p0, F0=311.49, mask=m0)
        psr.save_feather(p1, F0=311.49, mask=m1)

        dfs, *_ = data_loader.LoadWidebandPulsarData.read_multiple_feather([p0, p1])
        result = data_loader.LoadWidebandPulsarData.process_pulsar_residuals_by_epoch(
            dfs
        )
        assert "mask" in result
        assert result["mask"].shape == (12, 2)
        assert np.array_equal(result["mask"][:, 0], m0)
        assert np.array_equal(result["mask"][:, 1], m1)

    def test_epoch_alignment_omits_mask_when_absent(self, tmp_path):
        """No mask key when the feathers carry none (backward-compatible default)."""
        psr = _synthetic_pulsar(n=12)
        p0 = str(tmp_path / "A.feather")
        psr.save_feather(p0, F0=311.49)
        dfs, *_ = data_loader.LoadWidebandPulsarData.read_multiple_feather([p0])
        result = data_loader.LoadWidebandPulsarData.process_pulsar_residuals_by_epoch(
            dfs
        )
        assert "mask" not in result

    def test_read_committed_fixture(self):
        """The committed fixture feather loads into a valid object."""
        psr = data_loader.LoadWidebandPulsarData.read_feather(FIXTURE_FEATHER)
        assert psr.name == "J9999+9999"
        assert psr.F0 == 311.49
        assert len(psr.toas) == 12
        assert psr.M_matrix.shape == (12, 5)
        # Derived quantities are recomputed by __init__.
        assert psr.P_eps.shape == (5, 5)
        col_norms = np.sqrt(np.sum(psr.M_scaled**2, axis=0))
        assert np.allclose(col_norms, 1.0)

    def test_read_multiple_feather(self):
        """read_multiple_feather returns the same 4-tuple shape as par/tim."""
        result = data_loader.LoadWidebandPulsarData.read_multiple_feather(
            [FIXTURE_FEATHER]
        )
        pulsar_dfs, metadata, design_matrices, covariances = result

        assert len(pulsar_dfs) == 1
        assert list(pulsar_dfs[0].columns) == ["toas", "residuals", "error"]
        assert metadata.loc[0, "name"] == "J9999+9999"
        assert metadata.loc[0, "F0"] == 311.49
        assert metadata.loc[0, "dim_M"] == 5
        assert len(design_matrices) == 1
        assert len(covariances) == 1

    def test_get_processed_residuals_prefers_feather(self, tmp_path):
        """get_processed_residuals uses feathers when present (no par/tim needed)."""
        psr = _synthetic_pulsar()
        psr.save_feather(str(tmp_path / "J9999+9999.feather"), F0=311.49)

        result = data_loader.LoadWidebandPulsarData.get_processed_residuals(
            str(tmp_path), mode="cw"
        )
        assert "processed_residuals" in result
        assert result["metadata"].loc[0, "name"] == "J9999+9999"
        assert len(result["design_matrices"]) == 1

    def test_empty_directory_raises(self, tmp_path):
        """A directory with no feather/par/tim files raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            data_loader.LoadWidebandPulsarData.get_processed_residuals(str(tmp_path))


class TestEnterpriseFeatherGolden:
    """Golden numerics: enterprise vs feather must be bit-identical.

    Skipped where enterprise (a data-prep-only dependency) is not installed, i.e.
    in CI. Run locally in the ``Argus`` conda env to guard the validated likelihood.
    """

    def test_enterprise_vs_feather_bit_identical(self, tmp_path):
        """A real par/tim load matches its feather round-trip bit-for-bit."""
        pytest.importorskip("enterprise")

        data_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
            "workflows",
            "data",
            "IPTA_MockDataChallenge2",
            "dataset_3b",
        )
        par = os.path.join(data_dir, "J0030+0451.par")
        tim = os.path.join(data_dir, "J0030+0451.tim")
        if not (os.path.exists(par) and os.path.exists(tim)):
            pytest.skip("IPTA MDC2 par/tim data not available")

        ent = data_loader.LoadWidebandPulsarData.read_par_tim(par, tim)
        f0 = data_loader.LoadWidebandPulsarData.get_par_value(par, "F0")

        path = str(tmp_path / f"{ent.name}.feather")
        ent.save_feather(path, F0=f0)
        fea = data_loader.LoadWidebandPulsarData.read_feather(path)

        for attr in ["toas", "toaerrs", "residuals", "M_matrix", "M_scaled", "P_eps"]:
            a = np.asarray(getattr(ent, attr))
            b = np.asarray(getattr(fea, attr))
            assert np.array_equal(
                a, b
            ), f"{attr} differs between enterprise and feather"
