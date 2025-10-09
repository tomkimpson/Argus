"""Unit tests for io_manager module."""

import pytest
import os
import tempfile
import shutil
import logging
from unittest.mock import Mock, patch, MagicMock
from argus import io_manager


class TestSetupOutputDirectory:
    """Tests for setup_output_directory function."""

    def test_with_output_id(self, mock_config, tmp_path):
        """Test output directory setup with explicit output_id."""
        mock_config.set("Output", "output_id", "test_run_123")
        mock_config.set("Output", "base_dir", "outputs_{output_id}")

        # Mock the project root detection
        with patch('os.path.abspath', return_value=str(tmp_path / "project")):
            with patch('os.path.dirname', return_value=str(tmp_path)):
                output_dir = io_manager.setup_output_directory(
                    mock_config,
                    use_gw=True,
                    config_path=str(tmp_path / "workflows/test_workflow/configs/config.ini")
                )

        assert "test_run_123" in output_dir

    def test_without_output_id(self, mock_config, tmp_path):
        """Test output directory setup without output_id (uses timestamp)."""
        mock_config.set("Output", "output_id", "")
        mock_config.set("Output", "base_dir", "outputs_{timestamp}")

        with patch('os.path.abspath', return_value=str(tmp_path / "project")):
            with patch('os.path.dirname', return_value=str(tmp_path)):
                output_dir = io_manager.setup_output_directory(
                    mock_config,
                    use_gw=True,
                    timestamp="20240101_120000"
                )

        assert "20240101_120000" in output_dir

    def test_no_gw_nested_directory(self, mock_config, tmp_path):
        """Test that no-GW runs create nested directory structure."""
        mock_config.set("Output", "output_id", "test_run")
        mock_config.set("Output", "base_dir", "outputs_{output_id}")

        with patch('os.path.abspath', return_value=str(tmp_path / "project")):
            with patch('os.path.dirname', return_value=str(tmp_path)):
                output_dir = io_manager.setup_output_directory(
                    mock_config,
                    use_gw=False,
                    config_path=str(tmp_path / "workflows/test_workflow/configs/config.ini")
                )

        assert "no_gw" in output_dir

    def test_gw_run_base_directory(self, mock_config, tmp_path):
        """Test that GW runs use base directory."""
        mock_config.set("Output", "output_id", "test_run")
        mock_config.set("Output", "base_dir", "outputs_{output_id}")

        with patch('os.path.abspath', return_value=str(tmp_path / "project")):
            with patch('os.path.dirname', return_value=str(tmp_path)):
                output_dir = io_manager.setup_output_directory(
                    mock_config,
                    use_gw=True,
                    config_path=str(tmp_path / "workflows/test_workflow/configs/config.ini")
                )

        assert "no_gw" not in output_dir


class TestCopyConfigFile:
    """Tests for copy_config_file function."""

    def test_copy_config(self, tmp_path, mock_logger):
        """Test copying configuration file to output directory."""
        # Create a test config file
        config_file = tmp_path / "config.ini"
        config_file.write_text("[Test]\nkey = value\n")

        # Create output directory
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Copy config
        copied_path = io_manager.copy_config_file(
            str(config_file),
            str(output_dir),
            mock_logger
        )

        # Verify file was copied
        assert os.path.exists(copied_path)
        assert copied_path == str(output_dir / "config.ini")

        # Verify content is the same
        assert (output_dir / "config.ini").read_text() == "[Test]\nkey = value\n"

    def test_preserves_filename(self, tmp_path, mock_logger):
        """Test that the original filename is preserved."""
        config_file = tmp_path / "my_special_config.ini"
        config_file.write_text("[Test]\n")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        copied_path = io_manager.copy_config_file(
            str(config_file),
            str(output_dir),
            mock_logger
        )

        assert "my_special_config.ini" in copied_path


class TestSaveNumpyroResults:
    """Tests for save_numpyro_results function."""

    def test_save_results(self, tmp_path, mock_logger):
        """Test saving NumPyro results to NetCDF."""
        # Create mock InferenceData
        mock_inf_data = Mock()
        mock_inf_data.to_netcdf = Mock()

        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir, exist_ok=True)
        output_id = "test_run"

        results_path = io_manager.save_numpyro_results(
            mock_inf_data,
            output_dir,
            output_id,
            mock_logger
        )

        # Verify to_netcdf was called
        mock_inf_data.to_netcdf.assert_called_once()

        # Verify path format
        assert output_id in results_path
        assert results_path.endswith(".nc")
        assert results_path == os.path.join(output_dir, f"{output_id}_results.nc")


class TestSetupSingleLogger:
    """Tests for setup_single_logger function."""

    def test_logger_creation(self, mock_config, tmp_path):
        """Test basic logger creation."""
        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir, exist_ok=True)

        logger = io_manager.setup_single_logger(
            mock_config,
            output_dir=output_dir,
            enable_file_logging=False
        )

        assert logger.name == "argus"
        assert len(logger.handlers) > 0

    def test_file_logging_enabled(self, mock_config, tmp_path):
        """Test logger with file logging enabled."""
        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir, exist_ok=True)

        mock_config.set("Logging", "enable_file_logging", "True")

        logger = io_manager.setup_single_logger(
            mock_config,
            output_dir=output_dir,
            enable_file_logging=True
        )

        # Should have both console and file handlers
        assert len(logger.handlers) == 2

        # Verify log directory was created
        log_dir = os.path.join(output_dir, "logfiles")
        assert os.path.exists(log_dir)

    def test_file_logging_disabled(self, mock_config):
        """Test logger with file logging disabled."""
        logger = io_manager.setup_single_logger(
            mock_config,
            enable_file_logging=False
        )

        # Should have only console handler
        assert len(logger.handlers) == 1

    def test_file_logging_requires_output_dir(self, mock_config):
        """Test that file logging requires output_dir."""
        with pytest.raises(ValueError, match="output_dir must be provided"):
            io_manager.setup_single_logger(
                mock_config,
                output_dir=None,
                enable_file_logging=True
            )

    def test_logger_level_setting(self, mock_config, tmp_path):
        """Test that logger level is set correctly."""
        mock_config.set("Logging", "level", "DEBUG")
        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir, exist_ok=True)

        logger = io_manager.setup_single_logger(
            mock_config,
            output_dir=output_dir,
            enable_file_logging=False
        )

        assert logger.level == logging.DEBUG

    def test_no_propagation(self, mock_config):
        """Test that logger doesn't propagate to prevent duplicates."""
        logger = io_manager.setup_single_logger(
            mock_config,
            enable_file_logging=False
        )

        assert logger.propagate is False


class TestGetArgusLogger:
    """Tests for get_argus_logger function."""

    def test_get_initialized_logger(self, mock_config):
        """Test getting an initialized logger."""
        # First initialize the logger
        io_manager.setup_single_logger(mock_config, enable_file_logging=False)

        # Then retrieve it
        logger = io_manager.get_argus_logger()

        assert logger.name == "argus"
        assert len(logger.handlers) > 0

    def test_get_uninitialized_logger(self):
        """Test that getting uninitialized logger raises error."""
        # Clear any existing handlers
        logger = logging.getLogger("argus")
        logger.handlers = []

        with pytest.raises(RuntimeError, match="not initialized"):
            io_manager.get_argus_logger()


class TestGetOutputIdFromConfig:
    """Tests for get_output_id_from_config function."""

    def test_with_output_id(self, mock_config):
        """Test extracting output_id from config."""
        mock_config.set("Output", "output_id", "my_test_run")

        output_id = io_manager.get_output_id_from_config(mock_config)

        assert output_id == "my_test_run"

    def test_without_output_id_uses_timestamp(self, mock_config):
        """Test that timestamp is used when no output_id."""
        mock_config.set("Output", "output_id", "")

        output_id = io_manager.get_output_id_from_config(
            mock_config,
            timestamp="20240101_120000"
        )

        assert output_id == "20240101_120000"

    def test_without_output_id_generates_timestamp(self, mock_config):
        """Test that timestamp is generated when not provided."""
        mock_config.set("Output", "output_id", "")

        output_id = io_manager.get_output_id_from_config(mock_config)

        # Should be a timestamp format
        assert len(output_id) > 0
        # Should contain underscores (timestamp format)
        assert "_" in output_id

    def test_strips_whitespace(self, mock_config):
        """Test that whitespace is stripped from output_id."""
        mock_config.set("Output", "output_id", "  test_run  ")

        output_id = io_manager.get_output_id_from_config(mock_config)

        assert output_id == "test_run"
        assert not output_id.startswith(" ")
        assert not output_id.endswith(" ")
