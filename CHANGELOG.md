# Changelog

All notable changes to the Argus project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Command-line interface with `argus` command
- Package installation support via PyPI as `argus-pta`
- Configuration file template generation with `argus init`
- Enhanced API with main functions exposed at package level
- Comprehensive README with installation and usage instructions
- Package metadata and classifiers for better discoverability

### Changed
- Restructured package to support proper installation
- Updated dependencies to include all required packages
- Enhanced `__init__.py` to expose key functions and modules
- Package name changed to `argus-pta` for PyPI distribution

### Fixed
- Added missing dependencies (JAX, NumPyro, TensorFlow Probability, etc.)
- Import structure for installable package

## [0.0.0-dev] - 2025-01-XX

### Added
- Initial package structure with Poetry configuration
- Bayesian inference capabilities using NumPyro
- JAX-based Kalman filtering implementation
- Pulsar timing data analysis tools
- Gravitational wave detection algorithms
- Comprehensive test suite
- Documentation with MkDocs
- GitHub Actions CI/CD pipeline

### Features
- GPU acceleration support with JAX
- Model comparison and Bayes factors
- Enterprise pulsar data integration
- Flexible configuration system
- Analysis workflow management