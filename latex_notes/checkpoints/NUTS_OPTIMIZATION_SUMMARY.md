# NUTS Optimization for Hierarchical Pulsar Timing Array Analysis

## Overview

This document summarizes the optimization work performed on the Argus Bayesian inference pipeline to improve NUTS (No U-Turn Sampler) performance for high-dimensional hierarchical models in pulsar timing array gravitational wave detection.

## Problem Statement

The original NUTS implementation was too slow for practical use with hierarchical Bayesian models containing ~35-70 parameters. Initial runs were taking hours to complete warmup phases, with some runs exhibiting exponential performance degradation.

## Key Parameter Structure

The Argus model includes:
- **GW background**: `log10_ha` (amplitude), `γa` (spectral index, typically fixed)  
- **Pulsar red noise**: `log10_γp`, `log10_σp` (per-pulsar parameters)
- **Measurement noise**: `EFAC`, `EQUAD` (per-pulsar error scaling)

For 32 pulsars, this results in 65-70 parameters when using hierarchical priors.

## Summary of Optimization Attempts

### Run 012: Baseline Performance Issues
- **Configuration**: Individual priors for all parameters
- **Parameters**: 65 free parameters  
- **Result**: ❌ Very slow (~43 seconds per iteration), only 90/1000 samples completed
- **Issues**: Large gradient L2 norm (1.76e+03), poor parameter conditioning

### Run 013: First Hierarchical Implementation
- **Configuration**: Hierarchical priors for `log10_γp` only
- **Parameters**: 35 free parameters (50% reduction from Run 012)
- **Improvements**: 
  - Parameter reparameterization: `log10_ha` → N(0,1) transform
  - Hierarchical modeling: Population-level hyperparameters for `log10_γp`
  - Better NUTS configuration: target_accept_prob=0.85, max_tree_depth=12
- **Result**: ✅ Much improved, better gradient conditioning (L2 norm: 1.27e+02)
- **Issues**: Still slower than desired, σp parameters fixed (not learned from data)

### Run 014: Full Hierarchical Model (Failed)
- **Configuration**: Hierarchical priors for both `log10_γp` AND `log10_σp`  
- **Parameters**: 69 free parameters
- **Improvements**:
  - Extended hierarchical modeling to σp parameters
  - Advanced gradient balancing across all parameter types
- **Result**: ❌ SEVERE performance degradation
  - Iterations taking 200-350 seconds each (vs ~20s in Run 013)
  - NUTS hitting max tree depth (4095 leapfrog steps per iteration)
  - Exponential slowdown - killed after 2+ hours at only 12% warmup completion
- **Analysis**: 69 parameters exceeded NUTS practical limits despite optimization

### Run 015: Conservative Hierarchical Success
- **Configuration**: Hierarchical `log10_γp`, FIXED `log10_σp` (revert from Run 014)
- **Parameters**: 35 free parameters (same as Run 013)
- **Improvements**:
  - Kept gradient balancing optimizations from Run 014
  - Reduced max_tree_depth from 12→10 for safety
  - Improved runtime estimation logging (removed inaccurate estimates)
- **Result**: ✅ COMPLETE SUCCESS
  - Completed in 5h 51m with good convergence
  - R-hat ≤ 1.01, no divergent transitions
  - Validated that gradient balancing improvements work effectively
- **Limitation**: σp parameters still fixed, not learned from data

### Run 016: Log-Ratio Parameterization (In Progress)
- **Configuration**: Hierarchical `log10_γp` + log-ratio parameterization for σp
- **Parameters**: 68 free parameters
- **Innovation**: `log10_σp = log10_γp + log10_ratio`
  - Sample γp and ratio hierarchically
  - Derive σp deterministically
  - Potentially reduces parameter correlations vs direct σp sampling
- **Goal**: Achieve 68-parameter sampling with better geometry than Run 014
- **Status**: Currently running

## Key Technical Improvements Implemented

### 1. Parameter Reparameterization
```python
# Transform uniform priors to N(0,1) for better NUTS geometry
log10_ha_prime ~ N(0,1)
log10_ha = mean + log10_ha_prime * std
```

### 2. Hierarchical Modeling
```python
# Population-level hyperparameters reduce effective dimensionality
log10_gamma_p_mean ~ Uniform(-9, -7)
log10_gamma_p_std ~ Uniform(0.1, 1.0)  
log10_γp[i] ~ Normal(population_mean, population_std)
```

### 3. Gradient Magnitude Balancing
```python
# Scale parameters to have similar gradient magnitudes
log10_γp_raw ~ N(0, 1/√n_pulsars)  # Dimensional scaling
# All transforms use 3-sigma rule for consistent scaling
```

### 4. Log-Ratio Parameterization (New)
```python
# Reduce correlations between γp and σp
log10_γp[i] ~ Hierarchical(...)
log10_ratio[i] ~ Hierarchical(...)  
log10_σp[i] = log10_γp[i] + log10_ratio[i]  # Deterministic
```

### 5. NUTS Configuration Optimization
- **Target acceptance probability**: 0.85 (conservative for high-dimensional spaces)
- **Max tree depth**: 10 (reduced from default 12 to prevent runaway trajectories)
- **Dense mass matrix**: Enabled for better correlation handling
- **Step size adaptation**: Automatic with parameter-specific scaling

## Performance Results Summary

| Run | Parameters | Status | Runtime | Key Innovation | Notes |
|-----|------------|--------|---------|----------------|-------|
| 012 | 65 | ❌ Failed | 1+ hours (incomplete) | Baseline | Poor conditioning |
| 013 | 35 | ✅ Success | ~1.4 hours | Hierarchical γp | Good baseline |
| 014 | 69 | ❌ Failed | 2+ hours (incomplete) | Full hierarchical | Exponential degradation |
| 015 | 35 | ✅ Success | 5h 51m | Gradient balancing | Validated improvements |
| 016 | 68 | 🔄 Running | TBD | Log-ratio parameterization | Testing new approach |

## Key Findings

### What Works ✅
1. **Hierarchical modeling** dramatically improves NUTS performance when parameter count ≤ 35
2. **Gradient balancing** through standardized parameterization improves convergence  
3. **Parameter reparameterization** (uniform → normal) helps NUTS navigation
4. **Conservative NUTS settings** (target_accept_prob=0.85, max_tree_depth=10) prevent failures
5. **Dense mass matrix** essential for correlated parameters

### What Doesn't Work ❌
1. **Traditional hierarchical modeling breaks down** around 69 parameters
2. **Direct independent sampling** of correlated parameters (γp, σp) creates challenging geometry
3. **Overly optimistic NUTS settings** (high tree depth) can lead to runaway trajectories
4. **Simple runtime estimation** wildly inaccurate for hierarchical models

### Parameter Count Limits
- **≤ 35 parameters**: NUTS works well with proper tuning
- **35-50 parameters**: Potentially workable with advanced techniques  
- **≥ 70 parameters**: Likely requires alternative approaches or parameterization innovations

## Scientific Limitations

### Run 015 (Current Best)
- **Pros**: Stable, convergent, practical runtime
- **Cons**: σp parameters fixed, not learned from data
- **Impact**: May bias GW detection by assuming incorrect noise levels

### The σp Problem
- **Physical reality**: Each pulsar has different intrinsic timing noise (σp)
- **Statistical requirement**: Need to infer σp from data for unbiased GW detection
- **Computational challenge**: Adding σp as free parameters doubles parameter count (35 → 68+)

## Future Directions

### Immediate Testing
1. **Analyze Run 016 results** to assess log-ratio parameterization effectiveness
2. **Compare Run 016 vs Run 014** to isolate impact of parameterization choice

### Alternative Approaches if Log-Ratio Fails
1. **Principal component parameterization**: Reduce (γp, σp) to dominant modes
2. **Physical constraint models**: Use theoretical relationships between γp and σp  
3. **Block sampling strategies**: Coordinate updates of related parameters
4. **Alternative inference methods**: Variational inference, specialized MCMC variants

### Advanced Parameterizations
1. **Correlated noise modeling**: Multivariate priors for (γp, σp) pairs
2. **Pulsar grouping**: Share parameters across similar pulsars
3. **Time-varying parameters**: Allow noise properties to evolve

## Code Structure

### Key Files Modified
- `bayesian_inference.py`: Core model definitions, parameterizations, NUTS setup
- `inference_runners.py`: Runtime estimation, logging improvements
- Configuration files: `config_numpyro_test_01[3-6].ini`

### Key Functions
- `numpyro_model()`: NumPyro probabilistic model with parameterizations
- `get_prior_model_specs()`: Configuration parsing and prior setup
- `count_free_parameters()`: Parameter counting for diagnostics
- `run_numpyro_inference()`: NUTS execution with optimizations

## Recommendations for Continuation

### Immediate Actions
1. **Monitor Run 016** for signs of trajectory length explosion (like Run 014)
2. **If Run 016 succeeds**: Validate scientific results, compare parameter estimates
3. **If Run 016 fails**: Consider principal component or alternative approaches

### Parameter Count Strategy
- **Target**: Find reliable approach for ~50-60 parameters
- **Fallback**: Accept some level of σp constraint if full freedom impossible
- **Innovation**: Continue exploring parameterization improvements

### Scientific Validation
- **Compare GW detection sensitivity** across different σp treatments
- **Quantify bias** from fixed vs inferred σp parameters
- **Validate hierarchical assumptions** against simulated data

## Lessons Learned

1. **Incremental optimization beats revolutionary changes** - Run 015's success built on Run 013's foundation
2. **Parameter count limits are real** - NUTS performance degrades sharply beyond ~35-50 parameters
3. **Parameterization matters as much as algorithm tuning** - How you represent the problem affects sampling efficiency
4. **Conservative settings prevent catastrophic failures** - Better to sample slowly than not at all
5. **Gradient analysis is crucial** - Parameter scaling must ensure similar gradient magnitudes

---

*Last updated: 2025-06-23*  
*Status: Run 016 (log-ratio parameterization) in progress*