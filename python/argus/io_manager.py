"""I/O and file management utilities for the argus package."""

import os
import shutil
import logging
from datetime import datetime


def setup_output_directory(config, use_gw, timestamp=None, config_path=None):
    """Set up output directory for the inference run.
    
    Args:
        config: Configuration object
        use_gw (bool): Whether to include gravitational wave model
        timestamp (str): Optional timestamp to use for output directory
        config_path (str): Optional path to the configuration file to determine workflow directory
    
    Returns
    -------
        str: output_dir path
    """
    # Get output_id from config file
    output_id = config.get('Output', 'output_id', fallback='').strip()
    
    # Determine directory name: use ID if provided, otherwise use timestamp
    if output_id:
        base_dir = config.get('Output', 'base_dir').format(output_id=output_id)
    else:
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = config.get('Output', 'base_dir').format(timestamp=timestamp)
    
    # Determine which workflow directory to use based on config path
    workflow_name = 'example_workflow'  # default fallback
    if config_path:
        # Extract workflow name from config path (e.g., "workflows/dev_workflow/configs/dev_config.ini")
        config_path_normalized = os.path.normpath(config_path)
        path_parts = config_path_normalized.split(os.sep)
        if 'workflows' in path_parts:
            workflows_index = path_parts.index('workflows')
            if workflows_index + 1 < len(path_parts):
                workflow_name = path_parts[workflows_index + 1]
    
    # Create base output directory in the appropriate workflow directory
    # Get to project root (3 levels up from argus/io_manager.py)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    base_output_dir = os.path.join(project_root, 'workflows', workflow_name, 'outputs', base_dir)
    
    if not use_gw:
        # For no-GW runs, nest under the GW directory
        # First ensure the parent (GW) directory exists
        os.makedirs(base_output_dir, exist_ok=True)
        output_dir = os.path.join(base_output_dir, "no_gw")
    else:
        # For GW runs, use the base directory
        output_dir = base_output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting Bayesian inference {'with' if use_gw else 'without'} GW model...")
    
    return output_dir


def copy_config_file(config_path, output_dir, logger):
    """Copy configuration file to output directory.
    
    Args:
        config_path (str): Path to the configuration file
        output_dir (str): Output directory path
        logger: Logger object
        
    Returns
    -------
        str: Path to copied config file
    """
    config_filename = os.path.basename(config_path)
    output_config_path = os.path.join(output_dir, config_filename)
    shutil.copy2(config_path, output_config_path)
    logger.info(f"Copied config file to {output_config_path}")
    return output_config_path


def save_numpyro_results(inf_data, output_dir, output_id, logger):
    """Save NumPyro/ArviZ results.
    
    Args:
        inf_data: ArviZ InferenceData object
        output_dir: Output directory path
        output_id: Output identifier for naming
        logger: Logger object
    
    Returns
    -------
        str: Path to saved results file
    """
    logger.info("Saving NumPyro results...")
    results_path = os.path.join(output_dir, f'{output_id}_results.nc')
    
    # Save to NetCDF format
    inf_data.to_netcdf(results_path)
    logger.info(f"NumPyro results saved to {results_path}")
    
    return results_path


def setup_single_logger(config, output_dir=None, enable_file_logging=True):
    """Set up a single, properly configured logger for the entire application.
    
    This function creates a centralized logger that eliminates duplicate messages
    and provides consistent logging throughout the application. It supports both
    console and file logging.
    
    Args:
        config: Configuration object
        output_dir (str, optional): Directory for log files. Required if enable_file_logging=True
        enable_file_logging (bool): Whether to enable file logging. Default True.
        
    Returns
    -------
        logging.Logger: Configured logger instance for the entire application
        
    Raises
    ------
        ValueError: If enable_file_logging=True but output_dir is None
    """
    # Clear any existing handlers from the root logger and all argus loggers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Clear handlers from argus package loggers
    argus_logger = logging.getLogger('argus')
    for handler in argus_logger.handlers[:]:
        argus_logger.removeHandler(handler)
    
    # Create the main application logger
    logger = logging.getLogger('argus')
    logger.setLevel(getattr(logging, config.get('Logging', 'level', fallback='INFO')))
    
    # Prevent propagation to avoid duplicate messages
    logger.propagate = False
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if requested
    if enable_file_logging:
        if output_dir is None:
            raise ValueError("output_dir must be provided when enable_file_logging=True")
        
        # Create timestamp for log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(output_dir, 'logfiles', f'inference_output_{timestamp}.txt')
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    logger.info("Centralized logging initialized")
    return logger


def get_argus_logger():
    """Get the centralized argus logger.
    
    This function provides consistent access to the main application logger
    throughout the codebase. The logger should be initialized first using
    setup_single_logger().
    
    Returns
    -------
        logging.Logger: The centralized argus logger
        
    Raises
    ------
        RuntimeError: If the logger hasn't been initialized with setup_single_logger()
    """
    logger = logging.getLogger('argus')
    if not logger.handlers:
        raise RuntimeError(
            "Argus logger not initialized. Call setup_single_logger() first."
        )
    return logger


def setup_console_logging(config):
    """Set up console-only logging.
    
    DEPRECATED: Use setup_single_logger() instead for centralized logging.
    This function is kept for backward compatibility.
    
    Args:
        config: Configuration object
        
    Returns
    -------
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, config.get('Logging', 'level')))
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
    return logger


def get_output_id_from_config(config, timestamp=None):
    """Extract output_id from config with fallback to timestamp.
    
    Args:
        config: Configuration object
        timestamp (str): Optional timestamp to use as fallback
        
    Returns
    -------
        str: Output identifier
    """
    output_id = config.get('Output', 'output_id', fallback='').strip()
    if not output_id:
        # Fallback to timestamp if no output_id
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_id = timestamp
    return output_id