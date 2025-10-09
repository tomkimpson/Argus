"""Unit tests for gravitational_waves module."""

import pytest
import numpy as np
from argus import gravitational_waves


class TestPairwiseAngularSeparation:
    """Tests for pairwise_angular_separation function."""

    def test_identical_coordinates(self):
        """Test that identical coordinates give zero separation."""
        ra = np.array([0.0, 1.0])
        dec = np.array([0.0, 0.5])
        sep = gravitational_waves.pairwise_angular_separation(ra, dec)

        # Diagonal should be zero
        assert np.allclose(sep[0, 0], 0.0)
        assert np.allclose(sep[1, 1], 0.0)

    def test_symmetry(self):
        """Test that separation matrix is symmetric."""
        ra = np.array([0.0, 1.0, 2.0])
        dec = np.array([0.0, 0.5, -0.3])
        sep = gravitational_waves.pairwise_angular_separation(ra, dec)

        assert np.allclose(sep, sep.T)

    def test_antipodal_points(self):
        """Test separation of antipodal points (should be π)."""
        # Points at opposite poles
        ra = np.array([0.0, 0.0])
        dec = np.array([np.pi/2, -np.pi/2])
        sep = gravitational_waves.pairwise_angular_separation(ra, dec)

        assert np.allclose(sep[0, 1], np.pi, atol=1e-10)

    def test_right_angle_separation(self):
        """Test separation of points at right angles."""
        # Point at equator and point at pole
        ra = np.array([0.0, 0.0])
        dec = np.array([0.0, np.pi/2])
        sep = gravitational_waves.pairwise_angular_separation(ra, dec)

        assert np.allclose(sep[0, 1], np.pi/2, atol=1e-10)

    def test_single_pulsar(self):
        """Test with a single pulsar."""
        ra = np.array([1.5])
        dec = np.array([0.3])
        sep = gravitational_waves.pairwise_angular_separation(ra, dec)

        assert sep.shape == (1, 1)
        assert np.allclose(sep[0, 0], 0.0)

    def test_known_separation(self):
        """Test with known separation values."""
        # Two points on equator separated by 60 degrees in RA
        ra = np.array([0.0, np.pi/3])
        dec = np.array([0.0, 0.0])
        sep = gravitational_waves.pairwise_angular_separation(ra, dec)

        # Separation should be 60 degrees = π/3 radians
        assert np.allclose(sep[0, 1], np.pi/3, atol=1e-10)


class TestHellingsDowns:
    """Tests for hellings_downs function."""

    def test_autocorrelation(self):
        """Test that HD correlation is 1 for zero angle."""
        theta = 0.0
        hd = gravitational_waves.hellings_downs(theta)
        assert np.isclose(hd, 1.0)

    def test_autocorrelation_array(self):
        """Test autocorrelation in array with zero angles."""
        theta = np.array([0.0, 0.0, 0.0])
        hd = gravitational_waves.hellings_downs(theta)
        assert np.allclose(hd, 1.0)

    def test_cross_correlation_value(self):
        """Test HD correlation for known angle."""
        # For θ = π/2, HD(π/2) should be a specific value
        theta = np.pi / 2
        x = (1 - np.cos(theta)) / 2  # x = 0.5
        expected = (3/2) * x * np.log(x) - x/4 + 0.5
        hd = gravitational_waves.hellings_downs(theta)
        assert np.isclose(hd, expected)

    def test_array_input(self):
        """Test with array input."""
        theta = np.array([0.0, np.pi/4, np.pi/2, np.pi])
        hd = gravitational_waves.hellings_downs(theta)

        # First element should be 1 (autocorrelation)
        assert np.isclose(hd[0], 1.0)
        # All values should be between -0.5 and 1.0
        assert np.all(hd >= -0.5)
        assert np.all(hd <= 1.0)

    def test_pi_angle(self):
        """Test HD correlation at π (antipodal points)."""
        theta = np.pi
        x = (1 - np.cos(theta)) / 2  # x = 1
        expected = (3/2) * x * np.log(x) - x/4 + 0.5  # = 0 - 0.25 + 0.5 = 0.25
        hd = gravitational_waves.hellings_downs(theta)
        assert np.isclose(hd, expected, atol=1e-10)

    def test_small_angle_approximation(self):
        """Test HD correlation for small angles."""
        theta = 0.01  # Small angle in radians
        hd = gravitational_waves.hellings_downs(theta)
        # For small angles, HD should be close to 1
        assert hd < 1.0
        assert hd > 0.9

    def test_mixed_zero_nonzero_array(self):
        """Test array with both zero and non-zero angles."""
        theta = np.array([0.0, np.pi/2, 0.0, np.pi/4])
        hd = gravitational_waves.hellings_downs(theta)

        # Zero angles should give 1.0
        assert np.isclose(hd[0], 1.0)
        assert np.isclose(hd[2], 1.0)
        # Non-zero angles should give values < 1.0
        assert hd[1] < 1.0
        assert hd[3] < 1.0

    def test_negative_correlation(self):
        """Test that HD can be negative for some angles."""
        # HD function can be negative for certain angle ranges
        theta = np.linspace(0, np.pi, 100)
        hd = gravitational_waves.hellings_downs(theta)
        # Check that some values are negative
        assert np.any(hd < 0)
