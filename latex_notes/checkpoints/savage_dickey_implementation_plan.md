# Savage-Dickey Bayes Factor Implementation Plan

**Date**: 2025-07-22  
**Status**: Planning Phase  
**Goal**: Implement Savage-Dickey method for Bayes factor computation in Argus codebase

## Background

The Savage-Dickey density ratio provides a way to compute Bayes factors for nested models when one model (M₀) is a special case of another (M₁) where a parameter is set to a specific value. For gravitational wave detection:

- **M₁ (GW model)**: ha ≠ 0 (gravitational waves present)
- **M₀ (no-GW model)**: ha = 0 (no gravitational waves)

Since M₀ is nested within M₁, we can use:
```
BF₁₀ = π(ha=0|M₁) / posterior(ha=0|D,M₁)
```

This avoids the need to run separate nested sampling for both models and can be more numerically stable.

## Current System Analysis

### Existing ha Parameter Handling
The current code uses a reparameterization for better NUTS sampling:

```python
# In bayesian_inference.py lines 208-220
log10_ha_prime ~ N(0,1)
log10_ha = mean + log10_ha_prime * std
ha = 10^log10_ha
```

**Why this reparameterization?**
1. **Standardizes parameter scale** - all transformed parameters are O(1)
2. **Improves NUTS step size adaptation** - similar scales across parameters
3. **Reduces geometric pathologies** - avoids funnel geometry that traps NUTS

### Challenge for Savage-Dickey
- Need meaningful prior mass at ha = 0
- But ha = 0 corresponds to log10_ha = -∞  
- Current transform maps finite N(0,1) to finite log10 range
- Cannot directly evaluate prior at ha = 0 in log space

## Savage-Dickey Implementation Strategy

### 1. Implement Spike-and-Slab Prior for ha

Create mixed prior in **linear ha space** to handle ha = 0 properly:

```python
# New spike-and-slab prior structure
spike_indicator ~ Bernoulli(spike_prob)
if spike_indicator == 1:
    ha = 0.0  # Spike component at ha = 0
else:
    log10_ha_prime ~ N(0,1)  # Keep existing reparameterization
    log10_ha = mean + log10_ha_prime * std  
    ha = 10^log10_ha  # Slab component
```

**Benefits:**
- Preserves NUTS efficiency for continuous component
- Allows exact prior evaluation at ha = 0
- Maintains existing parameter scaling advantages

### 2. Modify Bayesian Inference Functions

**New functions in `bayesian_inference.py`:**
- `savage_dickey_configurable_prior_model()`: Spike-and-slab prior implementation
- Extend `numpyro_model()` to handle discrete/continuous mixture
- Add config parameters: `spike_prob`, `savage_dickey_method`

**NUTS Compatibility:**
- Use discrete sampling for spike indicator
- Apply existing reparameterization only to continuous component
- Ensure proper gradient flow for continuous parameters

### 3. Implement Savage-Dickey Calculation

**New functions in `analysis.py`:**

```python
def calculate_savage_dickey_bayes_factor(mcmc_samples, prior_config):
    """
    Calculate Savage-Dickey Bayes factor from MCMC samples.
    
    BF₁₀ = [π(ha=0)/π(ha≠0)] / [posterior(ha=0)/posterior(ha≠0)]
    """
    # Extract prior components
    spike_prob = prior_config['spike_prob']
    
    # Prior ratio: π(ha=0)/π(ha≠0) = spike_prob/(1-spike_prob)
    prior_ratio = spike_prob / (1 - spike_prob)
    
    # Posterior evaluation
    ha_samples = mcmc_samples['ha']
    continuous_samples = ha_samples[ha_samples > 0]  # Exclude spike samples
    
    # Estimate posterior density at ha=0 using KDE on continuous samples
    posterior_density_at_zero = estimate_posterior_density_at_zero(continuous_samples)
    
    # Posterior ratio calculation
    n_spike = jnp.sum(ha_samples == 0.0)
    n_continuous = len(ha_samples) - n_spike
    posterior_ratio = (n_spike/len(ha_samples)) / (posterior_density_at_zero * (n_continuous/len(ha_samples)))
    
    # Savage-Dickey Bayes factor
    bf_10 = prior_ratio / posterior_ratio
    
    return bf_10
```

**Supporting functions:**
- `estimate_posterior_density_at_zero()`: KDE-based density estimation
- Handle edge cases and provide uncertainty estimates

### 4. Create Test Configuration

**File: `configs/savage_dickey_test_001.ini`**

```ini
[PriorModel]
# Savage-Dickey specific settings
savage_dickey_method = true
spike_prob = 0.5  # Equal prior probability for ha=0 and ha≠0

# Fix all other parameters at injected values for one-parameter test
log10_gamma_a_fixed = true
log10_gamma_a_value = -9.0

psr_noise_fixed = true
efac_equad_fixed = true

# Slab component uses existing reparameterization
log10_ha_min = -18.0  
log10_ha_max = -14.0

[Inference] 
method = numpyro

[NUTS]
# Optimized for discrete/continuous mixture
num_samples = 4000
num_warmup = 2000
num_chains = 4
target_accept_prob = 0.85
```

### 5. Integration with Workflow

**Extend `workflow.py`:**

```python
def run_savage_dickey_inference(config_path, timestamp=None):
    """Run Savage-Dickey Bayes factor calculation."""
    # Load config and setup
    config = utils.load_config(config_path)
    # ... existing setup code ...
    
    # Run MCMC with spike-and-slab prior
    mcmc_results = run_numpyro_inference(...)
    
    # Calculate Savage-Dickey Bayes factor
    bf_results = calculate_savage_dickey_bayes_factor(
        mcmc_results.mcmc.get_samples(), 
        config['PriorModel']
    )
    
    return bf_results

def run_model_comparison(config_path, method='standard', timestamp=None):
    """
    Run model comparison using specified method.
    
    Args:
        method: 'standard' (nested sampling) or 'savage_dickey'
    """
    if method == 'savage_dickey':
        return run_savage_dickey_inference(config_path, timestamp)
    else:
        # Existing nested sampling approach
        return run_standard_model_comparison(config_path, timestamp)
```

### 6. Technical Challenges and Solutions

**Challenge 1: NUTS with Discrete Parameters**
- *Solution*: Use `numpyro.sample()` with `Bernoulli` distribution for spike indicator
- *Implementation*: Condition likelihood on indicator value

**Challenge 2: Posterior Density Estimation at ha=0**  
- *Solution*: Use kernel density estimation on continuous samples only
- *Implementation*: Gaussian KDE with automatic bandwidth selection
- *Validation*: Cross-validate density estimates with known distributions

**Challenge 3: Numerical Stability**
- *Solution*: Monitor effective sample sizes and convergence diagnostics
- *Implementation*: Add warnings for insufficient continuous samples
- *Fallback*: Graceful degradation to nested sampling method

**Challenge 4: Prior Specification**
- *Solution*: Config-driven spike probability and slab parameters
- *Implementation*: Validate config parameters and provide sensible defaults
- *Documentation*: Clear guidance on spike_prob selection

## Implementation Timeline

### Phase 1: Core Implementation (Days 1-3)
1. Implement spike-and-slab prior in `bayesian_inference.py`
2. Create Savage-Dickey calculation functions in `analysis.py`
3. Extend workflow integration

### Phase 2: Testing and Validation (Days 4-5)
1. Create test configuration with one free parameter
2. Validate against known analytical cases
3. Compare with existing nested sampling results
4. Test numerical stability and convergence

### Phase 3: Integration and Documentation (Day 6)
1. Integrate with main workflow
2. Add proper error handling and diagnostics
3. Update documentation and examples
4. Final testing on realistic parameter spaces

## Expected Benefits

1. **Theoretical Appropriateness**: Proper nested model comparison methodology
2. **Numerical Stability**: Avoid potential evidence ratio numerical issues
3. **Computational Efficiency**: Single MCMC run instead of two nested sampling runs
4. **NUTS Compatibility**: Preserve existing optimization for continuous parameters
5. **Flexibility**: Config-driven approach allows easy parameter adjustment

## Validation Strategy

1. **Analytical Test Cases**: Compare against problems with known BF solutions
2. **Consistency Checks**: Verify agreement with nested sampling on simple models
3. **Parameter Recovery**: Ensure method correctly identifies injected signals
4. **Convergence Diagnostics**: Monitor effective sample size and R-hat statistics
5. **Sensitivity Analysis**: Test robustness to spike_prob and slab range choices

## Files to Modify/Create

**Modifications:**
- `python/argus/bayesian_inference.py`: Add spike-and-slab prior functions
- `python/argus/analysis.py`: Add Savage-Dickey calculation functions  
- `python/argus/workflow.py`: Integration with existing workflow

**New files:**
- `python/argus/configs/savage_dickey_test_001.ini`: Test configuration
- Unit tests for validation
- Documentation updates

## Notes and Considerations

- The spike-and-slab approach is essential due to the log-space parameterization challenge
- Careful attention needed for NUTS tuning with discrete/continuous mixtures  
- Posterior density estimation quality is critical for reliable BF calculation
- Method should gracefully degrade to nested sampling if issues arise
- Consider implementing both parametric and non-parametric density estimation options