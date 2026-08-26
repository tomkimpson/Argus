"""Unit tests for workflow module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from argus import workflow


class TestSetupDataAndKalmanFilter:
    """Tests for setup_data_and_kalman_filter function."""

    @patch("argus.jax_kalman_filter.JaxKalmanFilter")
    @patch("argus.data_loader.LoadWidebandPulsarData.get_processed_residuals")
    def test_basic_setup(
        self, mock_get_residuals, mock_kf_class, mock_config, mock_logger
    ):
        """Test basic data and Kalman filter setup."""
        # Setup mocks
        mock_data = {
            "processed_residuals": {},
            "metadata": Mock(),
            "design_matrices": [],
            "parameter_covariances": [],
            "hd_correlation": [],
        }
        mock_get_residuals.return_value = mock_data
        mock_kf = Mock()
        mock_kf_class.return_value = mock_kf

        mock_config.set("Data", "data_path", "/path/to/data")
        mock_config.set("Data", "excluded_psrs", "J1640+2224")

        pulsar_data, kf = workflow.setup_data_and_kalman_filter(
            mock_config, mock_logger, use_gw=True
        )

        # Should load data
        mock_get_residuals.assert_called_once()

        # Should create Kalman filter
        mock_kf_class.assert_called_once()
        assert kf == mock_kf

    @patch("argus.jax_kalman_filter.JaxKalmanFilter")
    @patch("argus.data_loader.LoadWidebandPulsarData.get_processed_residuals")
    def test_with_no_gw(
        self, mock_get_residuals, mock_kf_class, mock_config, mock_logger
    ):
        """Test setup without GW model."""
        mock_data = {"processed_residuals": {}, "metadata": Mock()}
        mock_get_residuals.return_value = mock_data
        mock_kf_class.return_value = Mock()

        mock_config.set("Data", "data_path", "/path/to/data")
        mock_config.set("Data", "excluded_psrs", "")

        pulsar_data, kf = workflow.setup_data_and_kalman_filter(
            mock_config, mock_logger, use_gw=False
        )

        # Should pass use_gw=False to Kalman filter
        call_kwargs = mock_kf_class.call_args[1]
        assert call_kwargs["use_gw"] is False

    @patch("argus.jax_kalman_filter.JaxKalmanFilter")
    @patch("argus.data_loader.LoadWidebandPulsarData.get_processed_residuals")
    def test_pulsar_exclusion(
        self, mock_get_residuals, mock_kf_class, mock_config, mock_logger
    ):
        """Test that excluded pulsars are passed correctly."""
        mock_data = {"processed_residuals": {}, "metadata": Mock()}
        mock_get_residuals.return_value = mock_data
        mock_kf_class.return_value = Mock()

        mock_config.set("Data", "data_path", "/path/to/data")
        mock_config.set("Data", "excluded_psrs", "J1640+2224, PSR_J1234")

        workflow.setup_data_and_kalman_filter(mock_config, mock_logger, use_gw=True)

        # Check excluded pulsars were passed
        call_kwargs = mock_get_residuals.call_args[1]
        excluded = call_kwargs["excluded_psrs"]
        assert "J1640+2224" in excluded
        assert "PSR_J1234" in excluded


class TestRunInference:
    """Tests for run_inference function."""

    @patch("argus.utils.diagnostics")
    @patch("argus.utils.corner_plot")
    @patch("argus.io_manager.save_numpyro_results")
    @patch("argus.bayesian_inference.run_nuts_sampling")
    @patch("argus.bayesian_inference.test_likelihood_performance")
    @patch("argus.bayesian_inference.display_prior_summary")
    @patch("argus.prior_models.get_prior_model_specs")
    @patch("argus.utils.get_noise_parameters")
    @patch("argus.workflow.setup_data_and_kalman_filter")
    @patch("argus.io_manager.copy_config_file")
    @patch("argus.io_manager.setup_single_logger")
    @patch("argus.io_manager.setup_output_directory")
    @patch("argus.io_manager.get_output_id_from_config")
    @patch("argus.utils.resolve_config_paths")
    @patch("argus.utils.load_config")
    def test_full_inference_workflow(
        self,
        mock_load_config,
        mock_resolve,
        mock_get_id,
        mock_setup_dir,
        mock_logger,
        mock_copy,
        mock_setup_data,
        mock_get_noise,
        mock_get_priors,
        mock_display,
        mock_test_ll,
        mock_nuts,
        mock_save,
        mock_corner,
        mock_diag,
        mock_config,
        tmp_path,
    ):
        """Test complete inference workflow."""
        # Setup all mocks
        mock_load_config.return_value = mock_config
        mock_resolve.return_value = mock_config
        mock_get_id.return_value = "test_run"
        output_dir = str(tmp_path / "output")
        mock_setup_dir.return_value = output_dir
        mock_logger_obj = Mock()
        mock_logger.return_value = mock_logger_obj
        mock_copy.return_value = str(tmp_path / "config.ini")

        # Mock data and KF
        import pandas as pd

        mock_data = {"metadata": pd.DataFrame({"name": ["PSR1", "PSR2"]})}
        mock_kf = Mock()
        mock_setup_data.return_value = (mock_data, mock_kf)

        # Mock noise parameters
        import jax.numpy as jnp

        mock_get_noise.return_value = (
            jnp.ones(2),
            jnp.full(2, 1e-7),
            jnp.full(2, 1e-15),
            jnp.full(2, 1e-8),
        )

        # Mock priors
        mock_get_priors.return_value = {}

        # Mock NUTS results
        mock_results = Mock()
        mock_nuts.return_value = mock_results
        mock_save.return_value = str(tmp_path / "results.nc")

        # Run inference
        config_path = str(tmp_path / "config.ini")
        result_dir = workflow.run_inference(config_path, use_gw=True)

        # Verify workflow steps
        mock_load_config.assert_called_once()
        mock_setup_dir.assert_called_once()
        mock_logger.assert_called_once()
        mock_setup_data.assert_called_once()
        mock_nuts.assert_called_once()
        mock_save.assert_called_once()

    @patch("argus.utils.corner_plot")
    @patch("argus.io_manager.save_numpyro_results")
    @patch("argus.bayesian_inference.run_nuts_sampling")
    @patch("argus.bayesian_inference.test_likelihood_performance")
    @patch("argus.bayesian_inference.display_prior_summary")
    @patch("argus.prior_models.get_prior_model_specs")
    @patch("argus.utils.get_noise_parameters")
    @patch("argus.workflow.setup_data_and_kalman_filter")
    @patch("argus.io_manager.copy_config_file")
    @patch("argus.io_manager.setup_single_logger")
    @patch("argus.io_manager.setup_output_directory")
    @patch("argus.io_manager.get_output_id_from_config")
    @patch("argus.utils.resolve_config_paths")
    @patch("argus.utils.load_config")
    def test_no_gw_workflow(
        self,
        mock_load_config,
        mock_resolve,
        mock_get_id,
        mock_setup_dir,
        mock_logger,
        mock_copy,
        mock_setup_data,
        mock_get_noise,
        mock_get_priors,
        mock_display,
        mock_test_ll,
        mock_nuts,
        mock_save,
        mock_corner,
        mock_config,
        tmp_path,
    ):
        """Test workflow without GW model."""
        # Setup mocks (similar to above but simpler)
        mock_load_config.return_value = mock_config
        mock_resolve.return_value = mock_config
        mock_get_id.return_value = "test_run"
        mock_setup_dir.return_value = str(tmp_path / "output")
        mock_logger.return_value = Mock()
        mock_copy.return_value = str(tmp_path / "config.ini")

        import pandas as pd
        import jax.numpy as jnp

        mock_data = {"metadata": pd.DataFrame({"name": ["PSR1"]})}
        mock_setup_data.return_value = (mock_data, Mock())
        mock_get_noise.return_value = (
            jnp.ones(1),
            jnp.full(1, 1e-7),
            jnp.full(1, 1e-15),
            jnp.full(1, 1e-8),
        )
        mock_get_priors.return_value = {}
        mock_nuts.return_value = Mock()
        mock_save.return_value = str(tmp_path / "results.nc")

        config_path = str(tmp_path / "config.ini")
        workflow.run_inference(config_path, use_gw=False)

        # Verify use_gw=False was passed
        call_args = mock_setup_data.call_args[0]
        assert call_args[2] is False  # use_gw parameter

    @patch("argus.utils.corner_plot")
    @patch("argus.io_manager.save_numpyro_results")
    @patch("argus.bayesian_inference.run_nuts_sampling")
    @patch("argus.bayesian_inference.test_likelihood_performance")
    @patch("argus.bayesian_inference.display_prior_summary")
    @patch("argus.prior_models.get_prior_model_specs")
    @patch("argus.utils.get_noise_parameters")
    @patch("argus.workflow.setup_data_and_kalman_filter")
    @patch("argus.io_manager.copy_config_file")
    @patch("argus.io_manager.setup_single_logger")
    @patch("argus.io_manager.setup_output_directory")
    @patch("argus.io_manager.get_output_id_from_config")
    @patch("argus.utils.resolve_config_paths")
    @patch("argus.utils.load_config")
    def test_corner_plot_error_handling(
        self,
        mock_load_config,
        mock_resolve,
        mock_get_id,
        mock_setup_dir,
        mock_logger,
        mock_copy,
        mock_setup_data,
        mock_get_noise,
        mock_get_priors,
        mock_display,
        mock_test_ll,
        mock_nuts,
        mock_save,
        mock_corner,
        mock_config,
        tmp_path,
    ):
        """Test that corner plot errors are handled gracefully."""
        # Setup mocks
        mock_load_config.return_value = mock_config
        mock_resolve.return_value = mock_config
        mock_get_id.return_value = "test_run"
        mock_setup_dir.return_value = str(tmp_path / "output")
        mock_logger_obj = Mock()
        mock_logger.return_value = mock_logger_obj
        mock_copy.return_value = str(tmp_path / "config.ini")

        import pandas as pd
        import jax.numpy as jnp

        mock_data = {"metadata": pd.DataFrame({"name": ["PSR1"]})}
        mock_setup_data.return_value = (mock_data, Mock())
        mock_get_noise.return_value = (
            jnp.ones(1),
            jnp.full(1, 1e-7),
            jnp.full(1, 1e-15),
            jnp.full(1, 1e-8),
        )
        mock_get_priors.return_value = {}
        mock_nuts.return_value = Mock()
        mock_save.return_value = str(tmp_path / "results.nc")

        # Make corner plot fail
        mock_corner.side_effect = Exception("Plot error")

        # Should not raise exception
        config_path = str(tmp_path / "config.ini")
        workflow.run_inference(config_path, use_gw=True)

        # Should log error
        assert any(
            "error" in str(call).lower()
            for call in mock_logger_obj.error.call_args_list
        )

    @patch("argus.utils.diagnostics")
    @patch("argus.utils.corner_plot")
    @patch("argus.io_manager.save_numpyro_results")
    @patch("argus.bayesian_inference.run_nuts_sampling")
    @patch("argus.bayesian_inference.test_likelihood_performance")
    @patch("argus.bayesian_inference.display_prior_summary")
    @patch("argus.prior_models.get_prior_model_specs")
    @patch("argus.utils.get_noise_parameters")
    @patch("argus.workflow.setup_data_and_kalman_filter")
    @patch("argus.io_manager.copy_config_file")
    @patch("argus.io_manager.setup_single_logger")
    @patch("argus.io_manager.setup_output_directory")
    @patch("argus.io_manager.get_output_id_from_config")
    @patch("argus.utils.resolve_config_paths")
    @patch("argus.utils.load_config")
    def test_diagnostics_error_handling(
        self,
        mock_load_config,
        mock_resolve,
        mock_get_id,
        mock_setup_dir,
        mock_logger,
        mock_copy,
        mock_setup_data,
        mock_get_noise,
        mock_get_priors,
        mock_display,
        mock_test_ll,
        mock_nuts,
        mock_save,
        mock_corner,
        mock_diag,
        mock_config,
        tmp_path,
    ):
        """Test that diagnostics errors are handled gracefully."""
        # Setup mocks
        mock_load_config.return_value = mock_config
        mock_resolve.return_value = mock_config
        mock_get_id.return_value = "test_run"
        mock_setup_dir.return_value = str(tmp_path / "output")
        mock_logger_obj = Mock()
        mock_logger.return_value = mock_logger_obj
        mock_copy.return_value = str(tmp_path / "config.ini")

        import pandas as pd
        import jax.numpy as jnp

        mock_data = {"metadata": pd.DataFrame({"name": ["PSR1"]})}
        mock_setup_data.return_value = (mock_data, Mock())
        mock_get_noise.return_value = (
            jnp.ones(1),
            jnp.full(1, 1e-7),
            jnp.full(1, 1e-15),
            jnp.full(1, 1e-8),
        )
        mock_get_priors.return_value = {}
        mock_nuts.return_value = Mock()
        results_path = str(tmp_path / "results.nc")
        mock_save.return_value = results_path
        mock_corner.return_value = None

        # Make diagnostics fail
        mock_diag.side_effect = Exception("Diagnostics error")

        # Should not raise exception
        config_path = str(tmp_path / "config.ini")
        workflow.run_inference(config_path, use_gw=True)

        # Should log error
        assert any(
            "error" in str(call).lower()
            for call in mock_logger_obj.error.call_args_list
        )


class TestStageAEndToEndSmoke:
    """End-to-end CPU smoke of the M1 Stage A single-pulsar run shape.

    Exercises the REAL pipeline (data load from a one-feather directory, flat
    red-noise priors, GW fixed negligible, EFAC/EQUAD fixed from a one-entry
    noise JSON, tiny NUTS) with only the output directory redirected to
    tmp_path. This validates the Stage A config shape before any GPU time.
    """

    def _write_stage_a_config(self, tmp_path, psr_dir, noise_json):
        config_text = f"""
[Data]
data_path = {psr_dir}
excluded_psrs = __NONE__

[NUTS]
num_samples = 10
num_warmup = 10
num_chains = 1
target_accept_prob = 0.9
max_tree_depth = 5
dense_mass = true

[PriorModel]
log10_ha_fixed = true
log10_ha_value = -20.0
log10_gamma_a_fixed = true
log10_gamma_a_value = -8.5

spin_injections_path =
red_noise_prior = flat

log10_gamma_p_min = -12.0
log10_gamma_p_max = -6.0
log10_sigma_p_min = -20.0
log10_sigma_p_max = -12.0

noise_params_path = {noise_json}

efac_min = 0.5
efac_max = 2.0
log10_equad_min = -8.0
log10_equad_max = -6.0

[Logging]
level = INFO
enable_file_logging = false

[Output]
output_id = stage_a_smoke
base_dir = {{output_id}}
"""
        config_path = tmp_path / "stage_a_smoke.ini"
        config_path.write_text(config_text)
        return str(config_path)

    def test_stage_a_single_pulsar_run(self, tmp_path):
        """Full run_inference pass on a staged one-pulsar directory."""
        import json
        import os

        import arviz as az

        # Stage the single-pulsar directory (mirrors scripts/stage_mdc2.py)
        psr_dir = tmp_path / "J9999+9999"
        psr_dir.mkdir()
        feather_src = os.path.abspath("test/data/test_pulsar.feather")
        os.symlink(feather_src, psr_dir / "J9999+9999.feather")

        noise_json = psr_dir / "psr_noise.json"
        noise_json.write_text(json.dumps({"J9999+9999": {"efac": 1.0, "equad": -7.0}}))

        config_path = self._write_stage_a_config(tmp_path, psr_dir, noise_json)

        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir, exist_ok=True)
        with patch("argus.io_manager.setup_output_directory", return_value=output_dir):
            result_dir = workflow.run_inference(config_path, use_gw=True)

        assert result_dir == output_dir
        results_path = os.path.join(output_dir, "stage_a_smoke_results.nc")
        assert os.path.exists(results_path)

        idata = az.from_netcdf(results_path)
        post = idata.posterior
        # The 2 sampled red-noise sites + derived physicals are present
        assert "log10_γp_standardized" in post
        assert "log10_σp_standardized" in post
        assert "log10_γp" in post
        assert "log10_σp" in post
        # GW is fixed: no sampled GW latents, deterministic at the fixed value
        assert "log10_ha_prime" not in post
        assert float(post["log10_ha"].values.reshape(-1)[0]) == -20.0
