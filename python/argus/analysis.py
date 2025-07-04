"""Analysis and comparison utilities for the argus package."""

import os
import json
import logging
from datetime import datetime
import jax.numpy as jnp

from argus import utils


def calculate_and_save_bayes_factor(gw_output_dir, no_gw_output_dir, logger=None):
    """Calculate and save the Bayes factor between GW and no-GW models.
    
    Args:
        gw_output_dir (str): Directory containing GW model results
        no_gw_output_dir (str): Directory containing no-GW model results
        logger: Logger object (optional)
    
    Returns
    -------
        dict: Dictionary containing log evidences and Bayes factor, or None if calculation failed
    """
    if logger is None:
        from argus.io_manager import get_argus_logger
        logger = get_argus_logger()
        
    try:
        logger.info("=== Calculating Bayes Factor ===")
        
        # Determine inference methods used
        gw_method = utils.get_inference_method_from_files(gw_output_dir)
        no_gw_method = utils.get_inference_method_from_files(no_gw_output_dir)
        
        if gw_method is None:
            logger.error(f"No inference results found in {gw_output_dir}")
            return None
            
        if no_gw_method is None:
            logger.error(f"No inference results found in {no_gw_output_dir}")
            return None
        
        # Only calculate Bayes factor for nested sampling (JAXNS) since it provides evidence
        if gw_method != 'jaxns' or no_gw_method != 'jaxns':
            logger.warning("Bayes factor calculation requires nested sampling (JAXNS) for both models")
            logger.warning(f"GW method: {gw_method}, no-GW method: {no_gw_method}")
            logger.warning("MCMC methods (NumPyro) do not directly provide model evidence")
            return None
        
        # Find and load JAXNS results
        gw_results_file = utils.find_results_file(gw_output_dir, '*_results.json')
        no_gw_results_file = utils.find_results_file(no_gw_output_dir, '*_results.json')
        
        if gw_results_file is None:
            logger.error(f"No JAXNS results file found in {gw_output_dir}")
            return None
            
        if no_gw_results_file is None:
            logger.error(f"No JAXNS results file found in {no_gw_output_dir}")
            return None
        
        logger.info(f"Loading GW results from: {gw_results_file}")
        logger.info(f"Loading no-GW results from: {no_gw_results_file}")
        
        # Load the results using utility functions
        gw_results = utils.load_jaxns_results(gw_results_file)
        no_gw_results = utils.load_jaxns_results(no_gw_results_file)
        
        # Extract log evidences
        log_Z_gw, log_Z_gw_uncert = utils.extract_log_evidence(gw_results, 'jaxns')
        log_Z_no_gw, log_Z_no_gw_uncert = utils.extract_log_evidence(no_gw_results, 'jaxns')
        
        # Calculate log Bayes factor (GW vs no-GW)
        log_bayes_factor = log_Z_gw - log_Z_no_gw
        
        # Calculate uncertainty in Bayes factor (assuming independent uncertainties)
        log_bayes_factor_uncert = (log_Z_gw_uncert**2 + log_Z_no_gw_uncert**2)**0.5
        
        logger.info(f"Log evidence (GW model): {log_Z_gw:.3f} ± {log_Z_gw_uncert:.3f}")
        logger.info(f"Log evidence (no-GW model): {log_Z_no_gw:.3f} ± {log_Z_no_gw_uncert:.3f}")
        logger.info(f"Log Bayes factor (GW vs no-GW): {log_bayes_factor:.3f} ± {log_bayes_factor_uncert:.3f}")
        
        # Create results dictionary
        bayes_factor_results = {
            "log_evidence_gw": log_Z_gw,
            "log_evidence_gw_uncertainty": log_Z_gw_uncert,
            "log_evidence_no_gw": log_Z_no_gw,
            "log_evidence_no_gw_uncertainty": log_Z_no_gw_uncert,
            "log_bayes_factor": log_bayes_factor,
            "log_bayes_factor_uncertainty": log_bayes_factor_uncert,
            "bayes_factor": float(jnp.exp(log_bayes_factor)),
            "calculation_timestamp": datetime.now().isoformat(),
            "gw_results_file": gw_results_file,
            "no_gw_results_file": no_gw_results_file,
            "inference_method": "jaxns"
        }
        
        # Save results to the main GW directory
        bayes_factor_file = os.path.join(gw_output_dir, 'bayes_factor_results.json')
        with open(bayes_factor_file, 'w') as f:
            json.dump(bayes_factor_results, f, indent=2)
        
        logger.info(f"Bayes factor results saved to: {bayes_factor_file}")
        logger.info("=== Bayes Factor Calculation Complete ===")
        
        return bayes_factor_results
        
    except Exception as e:
        logger.error(f"Error calculating Bayes factor: {e}")
        logger.error("Bayes factor calculation failed")
        return None


def interpret_bayes_factor(bayes_factor_results, logger=None):
    """Interpret Bayes factor results using Jeffreys' scale.
    
    Args:
        bayes_factor_results (dict): Results dictionary from calculate_and_save_bayes_factor
        logger: Logger object (optional)
        
    Returns
    -------
        str: Interpretation string
    """
    if logger is None:
        from argus.io_manager import get_argus_logger
        logger = get_argus_logger()
        
    if bayes_factor_results is None:
        return "No Bayes factor results available"
    
    log_bf = bayes_factor_results["log_bayes_factor"]
    bf = bayes_factor_results["bayes_factor"]
    
    # Jeffreys' scale interpretation
    if log_bf < -2.3:  # BF < 0.1
        interpretation = "Strong evidence against GW model"
    elif log_bf < -1.0:  # BF < 0.37
        interpretation = "Moderate evidence against GW model"
    elif log_bf < 1.0:  # BF < 2.7
        interpretation = "Inconclusive evidence"
    elif log_bf < 2.3:  # BF < 10
        interpretation = "Moderate evidence for GW model"
    else:  # BF >= 10
        interpretation = "Strong evidence for GW model"
    
    result_str = f"Bayes Factor = {bf:.2f}, Interpretation: {interpretation}"
    logger.info(result_str)
    
    return result_str


def compare_inference_methods(results_paths, method_names, logger=None):
    """Compare results from different inference methods.
    
    Args:
        results_paths (list): List of paths to results files
        method_names (list): List of method names corresponding to results
        logger: Logger object (optional)
        
    Returns
    -------
        dict: Comparison summary
    """
    if logger is None:
        from argus.io_manager import get_argus_logger
        logger = get_argus_logger()
        
    logger.info("=== Comparing Inference Methods ===")
    
    comparison = {
        "methods": method_names,
        "results_files": results_paths,
        "comparison_timestamp": datetime.now().isoformat()
    }
    
    # This is a placeholder for future method comparison functionality
    # Could include parameter estimate comparisons, convergence metrics, etc.
    
    logger.info("Method comparison functionality is a placeholder for future development")
    
    return comparison