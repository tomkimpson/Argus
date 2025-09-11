"""Command-line interface for Argus package."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import jax

from . import utils, workflow, __version__


def setup_jax():
    """Set up JAX configuration."""
    jax.config.update("jax_enable_x64", True)


def print_system_info():
    """Print system and JAX configuration info."""
    print(f"=== ARGUS VERSION INFO ===")
    print(f"Argus version: {__version__}")
    print(f"JAX version: {jax.__version__}")
    print(f"Default device: {jax.default_backend()}")
    print()


def run_analysis(args):
    """Run Bayesian inference analysis."""
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file '{config_path}' not found.", file=sys.stderr)
        sys.exit(1)
    
    setup_jax()
    print_system_info()
    
    # Check GPU availability
    utils.check_gpu_availability()
    
    # Create timestamp for run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Starting Bayesian inference with config: {config_path}")
    
    # Run the analysis workflow
    gw_output_dir, no_gw_output_dir, bayes_factor_results = workflow.run_model_comparison(
        config_path=str(config_path),
        timestamp=timestamp
    )
    
    print(f"\n✓ Analysis complete!")
    print(f"Results saved to: {gw_output_dir}")
    if no_gw_output_dir:
        print(f"No-GW results saved to: {no_gw_output_dir}")


def create_config_template(args):
    """Create a configuration file template."""
    template_path = Path(args.output) if args.output else Path("argus_config.ini")
    
    # Basic configuration template
    template_content = """# Argus Configuration Template
# Modify these parameters for your analysis

[Data]
# Path to pulsar timing data
data_path = /path/to/your/data.pkl
noise_params_path = /path/to/noise_params.json

[Analysis]
# Analysis parameters
nsamples = 2000
nwarmup = 1000
nchains = 4

[Output]
# Output directory for results
output_dir = ./outputs/
"""
    
    template_path.write_text(template_content)
    print(f"Configuration template created: {template_path}")


def main():
    """Main entry point for Argus CLI."""
    parser = argparse.ArgumentParser(
        description='Argus: Bayesian inference for pulsar timing data analysis',
        prog='argus'
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version=f'Argus {__version__}'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run analysis command
    run_parser = subparsers.add_parser(
        'run',
        help='Run Bayesian inference analysis'
    )
    run_parser.add_argument(
        'config',
        type=str,
        help='Path to the configuration file'
    )
    run_parser.set_defaults(func=run_analysis)
    
    # Create config template command  
    config_parser = subparsers.add_parser(
        'init',
        help='Create a configuration file template'
    )
    config_parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output path for configuration file (default: argus_config.ini)'
    )
    config_parser.set_defaults(func=create_config_template)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)
    
    # Execute the appropriate function
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()