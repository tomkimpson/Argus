"""Unit tests for data_loader module."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from argus import data_loader


class TestLoadWidebandPulsarData:
    """Tests for LoadWidebandPulsarData class."""

    def test_initialization(self, mock_enterprise_pulsar):
        """Test basic initialization."""
        psr_data = data_loader.LoadWidebandPulsarData(mock_enterprise_pulsar)

        assert psr_data.name == "PSR_J0030+0451"
        assert len(psr_data.toas) == 10
        assert len(psr_data.toaerrs) == 10
        assert len(psr_data.residuals) == 10

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
            psr_data.toaerrs[1:]**2 + psr_data.toaerrs[:-1]**2
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
        df1 = pd.DataFrame({
            'toas': [100.0, 200.0, 300.0],
            'residuals': [1e-6, 2e-6, 3e-6],
            'error': [1e-7, 1e-7, 1e-7]
        })
        df2 = pd.DataFrame({
            'toas': [101.0, 201.0, 301.0],
            'residuals': [1.5e-6, 2.5e-6, 3.5e-6],
            'error': [1.5e-7, 1.5e-7, 1.5e-7]
        })

        result = data_loader.LoadWidebandPulsarData.process_pulsar_residuals_by_epoch([df1, df2])

        assert 'toas' in result
        assert 'residuals' in result
        assert 'errors' in result

        # TOAs should be averaged
        expected_toas = np.array([100.5, 200.5, 300.5])
        assert np.allclose(result['toas'], expected_toas)

        # Residuals and errors should be matrices
        assert result['residuals'].shape == (3, 2)
        assert result['errors'].shape == (3, 2)

    def test_empty_list_error(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            data_loader.LoadWidebandPulsarData.process_pulsar_residuals_by_epoch([])

    def test_shape_mismatch_error(self):
        """Test that mismatched shapes raise ValueError."""
        df1 = pd.DataFrame({
            'toas': [100.0, 200.0],
            'residuals': [1e-6, 2e-6],
            'error': [1e-7, 1e-7]
        })
        df2 = pd.DataFrame({
            'toas': [101.0, 201.0, 301.0],
            'residuals': [1.5e-6, 2.5e-6, 3.5e-6],
            'error': [1.5e-7, 1.5e-7, 1.5e-7]
        })

        with pytest.raises(ValueError, match="shape"):
            data_loader.LoadWidebandPulsarData.process_pulsar_residuals_by_epoch([df1, df2])

    def test_missing_column_error(self):
        """Test that missing columns raise ValueError."""
        df1 = pd.DataFrame({
            'toas': [100.0, 200.0],
            'residuals': [1e-6, 2e-6]
            # Missing 'error' column
        })

        with pytest.raises(ValueError, match="missing required columns"):
            data_loader.LoadWidebandPulsarData.process_pulsar_residuals_by_epoch([df1])


class TestGetProcessedResiduals:
    """Tests for get_processed_residuals static method."""

    @patch('argus.data_loader.LoadWidebandPulsarData.read_multiple_par_tim')
    @patch('argus.gravitational_waves.pairwise_angular_separation')
    @patch('argus.gravitational_waves.hellings_downs')
    @patch('glob.glob')
    def test_basic_functionality(self, mock_glob, mock_hd, mock_sep, mock_read):
        """Test basic get_processed_residuals functionality."""
        # Setup mocks
        mock_glob.side_effect = [
            ['/data/psr1.par', '/data/psr2.par'],
            ['/data/psr1.tim', '/data/psr2.tim']
        ]

        df1 = pd.DataFrame({
            'toas': [100.0, 200.0],
            'residuals': [1e-6, 2e-6],
            'error': [1e-7, 1e-7]
        })
        df2 = pd.DataFrame({
            'toas': [100.0, 200.0],
            'residuals': [1.5e-6, 2.5e-6],
            'error': [1.5e-7, 1.5e-7]
        })

        metadata = pd.DataFrame({
            'name': ['PSR1', 'PSR2'],
            'RA': [0.5, 1.0],
            'DEC': [0.3, 0.6]
        })

        mock_read.return_value = ([df1, df2], metadata, [np.eye(5), np.eye(5)], [np.eye(5), np.eye(5)])
        mock_sep.return_value = np.array([[0, 1], [1, 0]])
        mock_hd.return_value = np.array([[1.0, 0.5], [0.5, 1.0]])

        result = data_loader.LoadWidebandPulsarData.get_processed_residuals('/data')

        assert 'processed_residuals' in result
        assert 'metadata' in result
        assert 'design_matrices' in result
        assert 'parameter_covariances' in result
        assert 'hd_correlation' in result

    def test_directory_validation_empty(self):
        """Test that empty directory raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            data_loader.LoadWidebandPulsarData.get_processed_residuals("")

    def test_directory_validation_nonexistent(self):
        """Test that nonexistent directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            data_loader.LoadWidebandPulsarData.get_processed_residuals("/nonexistent/path")

    @patch('glob.glob')
    def test_no_files_error(self, mock_glob):
        """Test error when no par/tim files found."""
        mock_glob.side_effect = [[], []]

        with pytest.raises(FileNotFoundError, match="No .par or .tim files found"):
            data_loader.LoadWidebandPulsarData.get_processed_residuals('/data')

    @patch('glob.glob')
    def test_file_count_mismatch_error(self, mock_glob):
        """Test error when par and tim file counts don't match."""
        mock_glob.side_effect = [
            ['/data/psr1.par'],
            ['/data/psr1.tim', '/data/psr2.tim']
        ]

        with pytest.raises(ValueError, match="Mismatch"):
            data_loader.LoadWidebandPulsarData.get_processed_residuals('/data')

    @patch('argus.data_loader.LoadWidebandPulsarData.read_multiple_par_tim')
    @patch('glob.glob')
    def test_pulsar_exclusion(self, mock_glob, mock_read):
        """Test that excluded pulsars are filtered out."""
        mock_glob.side_effect = [
            ['/data/psr1.par', '/data/J1640+2224.par'],
            ['/data/psr1.tim', '/data/J1640+2224.tim']
        ]

        # read_multiple_par_tim should only be called with non-excluded files
        mock_read.return_value = ([], pd.DataFrame(), [], [])

        with pytest.raises(Exception):  # Will fail due to empty return, but that's ok
            data_loader.LoadWidebandPulsarData.get_processed_residuals(
                '/data',
                excluded_psrs=['J1640+2224']
            )

        # Check that J1640+2224 was filtered out
        called_par_files = mock_read.call_args[0][0]
        assert not any('J1640+2224' in f for f in called_par_files)


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

        value = data_loader.LoadWidebandPulsarData.get_par_value(
            str(par_file), "F0"
        )

        assert value == 200.12345

    def test_get_nonexistent_parameter(self, tmp_path):
        """Test getting a parameter that doesn't exist."""
        par_file = tmp_path / "test.par"
        par_file.write_text("""PSR J0030+0451
F0 200.12345
""")

        value = data_loader.LoadWidebandPulsarData.get_par_value(
            str(par_file), "PBDOT"
        )

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

        value = data_loader.LoadWidebandPulsarData.get_par_value(
            str(par_file), "F0"
        )

        assert value == 200.0

    def test_invalid_value_error(self, tmp_path):
        """Test ValueError for invalid parameter value."""
        par_file = tmp_path / "test.par"
        par_file.write_text("""PSR J0030+0451
F0 not_a_number
""")

        with pytest.raises(ValueError, match="Invalid parameter value"):
            data_loader.LoadWidebandPulsarData.get_par_value(
                str(par_file), "F0"
            )


class TestReadMultipleParTim:
    """Tests for read_multiple_par_tim classmethod."""

    @patch('argus.data_loader.LoadWidebandPulsarData.read_par_tim')
    @patch('argus.data_loader.LoadWidebandPulsarData.get_par_value')
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

        par_files = ['/data/psr1.par']
        tim_files = ['/data/psr1.tim']

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
                ['/data/psr1.par'],
                ['/data/psr1.tim', '/data/psr2.tim']
            )

    @patch('argus.data_loader.LoadWidebandPulsarData.read_par_tim')
    @patch('argus.data_loader.LoadWidebandPulsarData.get_par_value')
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

        par_files = [f'/data/psr{i}.par' for i in range(5)]
        tim_files = [f'/data/psr{i}.tim' for i in range(5)]

        result = data_loader.LoadWidebandPulsarData.read_multiple_par_tim(
            par_files, tim_files, max_files=2
        )

        pulsar_dfs, metadata, _, _ = result

        # Should only process 2 files
        assert len(pulsar_dfs) == 2
