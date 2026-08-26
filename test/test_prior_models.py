"""Unit tests for prior_models module."""

import pytest
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from argus import prior_models

tfpd = tfp.distributions


class TestGetGwParameterPriors:
    """Tests for get_gw_parameter_priors function."""

    def test_uniform_priors(self, mock_config):
        """Test creating uniform GW parameter priors."""
        priors = prior_models.get_gw_parameter_priors(mock_config)

        assert "log10_ha_spec" in priors
        assert "log10_ha_transform_params" in priors
        assert "log10_gamma_a_spec" in priors

        # Transform params should exist for ha (reparameterized)
        assert priors["log10_ha_transform_params"] is not None
        assert "mean" in priors["log10_ha_transform_params"]
        assert "std" in priors["log10_ha_transform_params"]

    def test_fixed_log10_ha(self, mock_config):
        """Test fixed log10_ha parameter."""
        mock_config.set("PriorModel", "log10_ha_fixed", "True")
        mock_config.set("PriorModel", "log10_ha_value", "-15.0")

        priors = prior_models.get_gw_parameter_priors(mock_config)

        # Should be fixed value, not distribution
        assert priors["log10_ha_spec"] == -15.0
        assert priors["log10_ha_transform_params"] is None

    def test_fixed_log10_gamma_a(self, mock_config):
        """Test fixed log10_gamma_a parameter."""
        mock_config.set("PriorModel", "log10_gamma_a_fixed", "True")
        mock_config.set("PriorModel", "log10_gamma_a_value", "-9.0")

        priors = prior_models.get_gw_parameter_priors(mock_config)

        assert priors["log10_gamma_a_spec"] == -9.0

    def test_reparameterization_parameters(self, mock_config):
        """Test that reparameterization parameters are correct."""
        mock_config.set("PriorModel", "log10_ha_min", "-16.0")
        mock_config.set("PriorModel", "log10_ha_max", "-14.0")

        priors = prior_models.get_gw_parameter_priors(mock_config)

        transform_params = priors["log10_ha_transform_params"]

        # Mean should be center of range
        expected_mean = (-16.0 + -14.0) / 2.0
        assert transform_params["mean"] == expected_mean

        # Std should be range/6 (3-sigma rule)
        expected_std = (-14.0 - -16.0) / 6.0
        assert transform_params["std"] == expected_std


class TestGetPulsarNoisePriors:
    """Tests for get_pulsar_noise_priors function."""

    def test_with_spin_injections(self, mock_config, tmp_path):
        """Test when spin injections are provided."""
        import pickle
        import pandas as pd

        # Create mock spin injection file
        spin_file = tmp_path / "spin.pkl"
        df = pd.DataFrame(
            {
                "psr": ["PSR1", "PSR2"],
                "optimal_sigma": [1e-15, 2e-15],
                "optimal_gamma": [1e-8, 2e-8],
            }
        )
        with open(spin_file, "wb") as f:
            pickle.dump(df, f)

        mock_config.set("PriorModel", "spin_injections_path", str(spin_file))

        sigma_p = jnp.array([1e-15, 2e-15])
        gamma_p = jnp.array([1e-8, 2e-8])

        priors = prior_models.get_pulsar_noise_priors(mock_config, 2, sigma_p, gamma_p)

        # Should use injected values (log10)
        assert priors["log10_gamma_p_spec"] is not None
        assert priors["log10_sigma_p_spec"] is not None

    def test_without_spin_injections(self, mock_config):
        """Test when no spin injections provided (hierarchical)."""
        mock_config.set("PriorModel", "spin_injections_path", "")

        priors = prior_models.get_pulsar_noise_priors(mock_config, 2, None, None)

        # Should be None (will use hierarchical)
        assert priors["log10_gamma_p_spec"] is None
        assert priors["log10_sigma_p_spec"] is None
        assert priors["hierarchical_specs"] is not None

    def test_hierarchical_specs_created(self, mock_config):
        """Test that hierarchical specs are always created."""
        mock_config.set("PriorModel", "spin_injections_path", "")

        priors = prior_models.get_pulsar_noise_priors(mock_config, 2, None, None)

        hier_specs = priors["hierarchical_specs"]
        assert hier_specs["hierarchical_noise"] is True
        assert hier_specs["log_ratio_parameterization"] is True
        assert "log10_gamma_p_mean_spec" in hier_specs
        assert "log10_gamma_p_std_spec" in hier_specs


class TestGetMeasurementNoisePriors:
    """Tests for get_measurement_noise_priors function."""

    def test_with_noise_params(self, mock_config, tmp_path):
        """Test when noise parameters file is provided."""
        import json

        noise_file = tmp_path / "noise.json"
        noise_data = {
            "PSR1": {"efac": 1.0, "equad": -7.0},
            "PSR2": {"efac": 1.2, "equad": -6.5},
        }
        noise_file.write_text(json.dumps(noise_data))

        mock_config.set("PriorModel", "noise_params_path", str(noise_file))

        efac = jnp.array([1.0, 1.2])
        equad = jnp.array([1e-7, 10 ** (-6.5)])

        priors = prior_models.get_measurement_noise_priors(mock_config, 2, efac, equad)

        # Should use injected values
        assert jnp.allclose(priors["efac_spec"], efac)
        assert jnp.allclose(priors["equad_spec"], equad)

    def test_without_noise_params(self, mock_config):
        """Test when no noise parameters provided (use priors)."""
        mock_config.set("PriorModel", "noise_params_path", "")

        priors = prior_models.get_measurement_noise_priors(mock_config, 2, None, None)

        # Should create distributions
        assert isinstance(priors["efac_spec"], tfpd.Distribution)
        assert isinstance(priors["equad_spec"], dict)
        assert priors["equad_spec"]["use_log10"] is True

    def test_log10_equad_parameterization(self, mock_config):
        """Test that EQUAD uses log10 parameterization."""
        mock_config.set("PriorModel", "noise_params_path", "")
        mock_config.set("PriorModel", "log10_equad_min", "-8.0")
        mock_config.set("PriorModel", "log10_equad_max", "-6.0")

        priors = prior_models.get_measurement_noise_priors(mock_config, 2, None, None)

        equad_spec = priors["equad_spec"]
        assert equad_spec["use_log10"] is True
        assert isinstance(equad_spec["log10_equad_spec"], tfpd.Distribution)


class TestCreateHierarchicalPriors:
    """Tests for create_hierarchical_priors function."""

    def test_creates_all_specs(self, mock_config):
        """Test that all hierarchical specs are created."""
        hier_specs = prior_models.create_hierarchical_priors(mock_config)

        # Check flags
        assert hier_specs["hierarchical_noise"] is True
        assert hier_specs["log_ratio_parameterization"] is True

        # Check gamma_p hierarchical specs
        assert "log10_gamma_p_mean_spec" in hier_specs
        assert "log10_gamma_p_std_spec" in hier_specs

        # Check ratio specs
        assert "log10_ratio_mean_spec" in hier_specs
        assert "log10_ratio_std_spec" in hier_specs

    def test_distribution_types(self, mock_config):
        """Test that specs are uniform distributions."""
        hier_specs = prior_models.create_hierarchical_priors(mock_config)

        assert isinstance(hier_specs["log10_gamma_p_mean_spec"], tfpd.Uniform)
        assert isinstance(hier_specs["log10_gamma_p_std_spec"], tfpd.Uniform)
        assert isinstance(hier_specs["log10_ratio_mean_spec"], tfpd.Uniform)
        assert isinstance(hier_specs["log10_ratio_std_spec"], tfpd.Uniform)

    def test_parameter_ranges(self, mock_config):
        """Test that parameter ranges are set correctly."""
        mock_config.set("PriorModel", "log10_gamma_p_mean_min", "-10.0")
        mock_config.set("PriorModel", "log10_gamma_p_mean_max", "-7.0")

        hier_specs = prior_models.create_hierarchical_priors(mock_config)

        gamma_mean_spec = hier_specs["log10_gamma_p_mean_spec"]
        assert float(gamma_mean_spec.low) == -10.0
        assert float(gamma_mean_spec.high) == -7.0


class TestGetPriorModelSpecs:
    """Tests for get_prior_model_specs function."""

    def test_returns_all_specs(self, mock_config):
        """Test that all required specs are returned."""
        n_pulsars = 2
        sigma_p = jnp.array([1e-15, 2e-15])
        gamma_p = jnp.array([1e-8, 2e-8])
        efac = jnp.ones(n_pulsars)
        equad = jnp.full(n_pulsars, 1e-7)

        specs = prior_models.get_prior_model_specs(
            mock_config, n_pulsars, sigma_p, gamma_p, efac, equad
        )

        # Check all required keys exist
        assert "log10_ha_spec" in specs
        assert "log10_ha_transform_params" in specs
        assert "log10_gamma_a_spec" in specs
        assert "log10_gamma_p_spec" in specs
        assert "log10_sigma_p_spec" in specs
        assert "efac_spec" in specs
        assert "equad_spec" in specs
        assert "hierarchical_specs" in specs

    def test_integration_with_submodules(self, mock_config):
        """Test that specs are correctly integrated from submodules."""
        n_pulsars = 2
        sigma_p = None
        gamma_p = None
        efac = None
        equad = None

        mock_config.set("PriorModel", "noise_params_path", "")
        mock_config.set("PriorModel", "spin_injections_path", "")

        specs = prior_models.get_prior_model_specs(
            mock_config, n_pulsars, sigma_p, gamma_p, efac, equad
        )

        # GW params should have transform params
        assert specs["log10_ha_transform_params"] is not None

        # Pulsar noise should be None (hierarchical)
        assert specs["log10_gamma_p_spec"] is None
        assert specs["log10_sigma_p_spec"] is None

        # Measurement noise should be distributions
        assert isinstance(specs["efac_spec"], tfpd.Distribution)

        # Hierarchical specs should exist
        assert specs["hierarchical_specs"]["hierarchical_noise"] is True


class TestFlatRedNoisePriors:
    """Tests for the flat (independent Uniform) red noise prior mode."""

    def _enable_flat(self, mock_config):
        mock_config.set("PriorModel", "red_noise_prior", "flat")
        mock_config.set("PriorModel", "log10_gamma_p_min", "-12.0")
        mock_config.set("PriorModel", "log10_gamma_p_max", "-6.0")
        mock_config.set("PriorModel", "log10_sigma_p_min", "-20.0")
        mock_config.set("PriorModel", "log10_sigma_p_max", "-12.0")

    def test_flat_mode_creates_uniform_specs(self, mock_config):
        """Flat mode returns per-pulsar Uniform specs and no hierarchical specs."""
        self._enable_flat(mock_config)
        n_pulsars = 3

        priors = prior_models.get_pulsar_noise_priors(
            mock_config, n_pulsars, None, None
        )

        assert isinstance(priors["log10_gamma_p_spec"], tfpd.Distribution)
        assert isinstance(priors["log10_sigma_p_spec"], tfpd.Distribution)
        assert priors["log10_gamma_p_spec"].low.shape == (n_pulsars,)
        assert float(priors["log10_gamma_p_spec"].low[0]) == -12.0
        assert float(priors["log10_sigma_p_spec"].high[0]) == -12.0
        assert priors["hierarchical_specs"] is None
        assert priors["empirical_specs"] is None

    def test_fixed_overrides_flat(self, mock_config):
        """spin_injections_path takes precedence over flat mode."""
        self._enable_flat(mock_config)
        mock_config.set("PriorModel", "spin_injections_path", "/some/path.pkl")
        gamma_p_array = jnp.array([1e-8, 1e-8])
        sigma_p_array = jnp.array([1e-15, 1e-15])

        priors = prior_models.get_pulsar_noise_priors(
            mock_config, 2, sigma_p_array, gamma_p_array
        )

        # Fixed arrays, not distributions
        assert not isinstance(priors["log10_gamma_p_spec"], tfpd.Distribution)
        assert priors["log10_gamma_p_spec"] is not None

    def test_default_remains_hierarchical(self, mock_config):
        """Without red_noise_prior the default hierarchical path is unchanged."""
        priors = prior_models.get_pulsar_noise_priors(mock_config, 3, None, None)

        assert priors["log10_gamma_p_spec"] is None
        assert priors["hierarchical_specs"] is not None
        assert priors["empirical_specs"] is None


class TestGetEmpiricalNoisePriors:
    """Tests for per-pulsar empirical red noise priors."""

    @pytest.fixture
    def empirical_json(self, tmp_path):
        import json

        priors = {
            "_meta": {"note": "test artifact"},
            "J1909-3744": {
                "log10_gamma_p": {"loc": -8.5, "scale": 0.30},
                "log10_ratio": {"loc": -6.5, "scale": 0.40},
            },
            "J0030+0451": {
                "log10_gamma_p": {"loc": -8.0, "scale": 0.20},
                "log10_ratio": {"loc": -6.0, "scale": 0.10},
            },
            "J1640+2224": {
                "log10_gamma_p": {"loc": -9.0, "scale": 0.50},
                "log10_ratio": {"loc": -7.0, "scale": 0.60},
            },
        }
        path = tmp_path / "empirical_priors.json"
        path.write_text(json.dumps(priors))
        return str(path)

    def test_loading_sorted_and_meta_ignored(self, empirical_json):
        """Pulsars are sorted by name; _meta keys ignored."""
        specs = prior_models.get_empirical_noise_priors(empirical_json)

        assert specs["psr_names"] == ["J0030+0451", "J1640+2224", "J1909-3744"]
        assert float(specs["gamma_loc"][0]) == pytest.approx(-8.0)
        assert float(specs["gamma_loc"][2]) == pytest.approx(-8.5)
        assert float(specs["ratio_scale"][1]) == pytest.approx(0.60)

    def test_exclusion_filtering(self, empirical_json):
        """Substring exclusion matches utils.get_efac_equad_injections semantics."""
        specs = prior_models.get_empirical_noise_priors(
            empirical_json, excluded_psrs=["J1640+2224"]
        )

        assert specs["psr_names"] == ["J0030+0451", "J1909-3744"]

    def test_inflation_applied_to_scales_only(self, empirical_json):
        """Inflation multiplies scales but not locs."""
        specs = prior_models.get_empirical_noise_priors(empirical_json, inflation=2.0)

        assert float(specs["gamma_scale"][0]) == pytest.approx(0.40)
        assert float(specs["gamma_loc"][0]) == pytest.approx(-8.0)
        assert float(specs["ratio_scale"][0]) == pytest.approx(0.20)

    def test_all_excluded_raises(self, empirical_json):
        """Excluding every pulsar raises a clear error."""
        with pytest.raises(ValueError, match="No pulsars left"):
            prior_models.get_empirical_noise_priors(
                empirical_json, excluded_psrs=["J0030", "J1640", "J1909"]
            )

    def test_pulsar_count_mismatch_raises(self, mock_config, empirical_json):
        """get_pulsar_noise_priors raises when the file and data disagree on n."""
        mock_config.set("PriorModel", "empirical_priors_path", empirical_json)
        mock_config.set("Data", "excluded_psrs", "")

        with pytest.raises(ValueError, match="must match exactly"):
            prior_models.get_pulsar_noise_priors(mock_config, 5, None, None)

    def test_empirical_mode_via_config(self, mock_config, empirical_json):
        """empirical_priors_path activates empirical mode with no hyperpriors."""
        mock_config.set("PriorModel", "empirical_priors_path", empirical_json)
        # mock_config excludes J1640+2224 -> 2 pulsars remain
        priors = prior_models.get_pulsar_noise_priors(mock_config, 2, None, None)

        assert priors["empirical_specs"] is not None
        assert priors["empirical_specs"]["psr_names"] == ["J0030+0451", "J1909-3744"]
        assert priors["log10_gamma_p_spec"] is None
        assert priors["log10_sigma_p_spec"] is None
        assert priors["hierarchical_specs"] is None

    def test_fixed_overrides_empirical(self, mock_config, empirical_json):
        """spin_injections_path takes precedence over empirical priors."""
        mock_config.set("PriorModel", "empirical_priors_path", empirical_json)
        mock_config.set("PriorModel", "spin_injections_path", "/some/path.pkl")
        gamma_p_array = jnp.array([1e-8, 1e-8])
        sigma_p_array = jnp.array([1e-15, 1e-15])

        priors = prior_models.get_pulsar_noise_priors(
            mock_config, 2, sigma_p_array, gamma_p_array
        )

        assert priors.get("empirical_specs") is None
        assert priors["log10_gamma_p_spec"] is not None


class TestRidgeParameterization:
    """Tests for the ridge GW parameterization (issue #109)."""

    def _enable_ridge(self, mock_config):
        mock_config.set("PriorModel", "gw_parameterization", "ridge")
        mock_config.set("PriorModel", "log10_pivot_psd_min", "-13.0")
        mock_config.set("PriorModel", "log10_pivot_psd_max", "-5.0")
        mock_config.set("PriorModel", "log10_gamma_a_min", "-11.0")
        mock_config.set("PriorModel", "log10_gamma_a_max", "-6.0")
        mock_config.set("PriorModel", "gw_pivot_freq_hz", "6.3376e-09")  # 1/(5 yr)

    def test_ridge_specs(self, mock_config):
        """Ridge mode returns pivot-PSD + gamma_a transforms and a pivot w."""
        self._enable_ridge(mock_config)
        specs = prior_models.get_gw_parameter_priors(mock_config)

        assert specs["gw_parameterization"] == "ridge"
        assert specs["log10_ha_spec"] is None
        assert specs["log10_ha_transform_params"] is None
        psd = specs["log10_pivot_psd_transform_params"]
        assert psd["mean"] == pytest.approx((-13.0 + -5.0) / 2.0)
        assert psd["std"] == pytest.approx((-5.0 - -13.0) / 6.0)
        ga = specs["log10_gamma_a_transform_params"]
        assert ga["min"] == -11.0 and ga["max"] == -6.0
        import math

        assert specs["gw_pivot_w"] == pytest.approx(2 * math.pi * 6.3376e-09)

    def test_default_is_direct(self, mock_config):
        """Absent gw_parameterization key -> direct mode, unchanged behavior."""
        specs = prior_models.get_gw_parameter_priors(mock_config)
        assert specs["gw_parameterization"] == "direct"
        assert specs["log10_ha_transform_params"] is not None

    def test_ridge_passes_through_prior_model_specs(self, mock_config):
        """get_prior_model_specs forwards the ridge keys in gwb mode."""
        self._enable_ridge(mock_config)
        specs = prior_models.get_prior_model_specs(
            mock_config, 2, None, None, None, None, mode="gwb"
        )
        assert specs["gw_parameterization"] == "ridge"
        assert "gw_pivot_w" in specs
        assert "log10_pivot_psd_transform_params" in specs
