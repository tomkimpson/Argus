"""Generate example documentation from Python scripts."""

import os
from pathlib import Path
import mkdocs_gen_files


def convert_script_to_docs(script_path: Path, output_path: str):
    """Convert a Python script to documentation format."""

    if not script_path.exists():
        return

    with open(script_path) as f:
        content = f.read()

    # Extract title from filename
    title = script_path.stem.replace("_", " ").title()

    # Create markdown content
    doc_content = f"""# {title}

This example demonstrates: {title.lower()}

## Source Code

```python
{content}
```

## Running this Example

To run this example:

```bash
cd examples/legacy/
python {script_path.name}
```

!!! note "Requirements"
    Make sure you have installed all dependencies:
    ```bash
    poetry install
    ```

## Expected Output

The script will output parameter estimation results and save them to the `outputs/` directory.

## Next Steps

- Modify the configuration parameters in the script
- Try different prior distributions
- Experiment with different samplers (NUTS vs nested sampling)
"""

    with mkdocs_gen_files.open(output_path, "w") as f:
        f.write(doc_content)


# Define example scripts to convert
EXAMPLE_SCRIPTS = [
    (
        "examples/legacy/parameter_estimation_with_NUTS.py",
        "examples/nuts_estimation.md",
    ),
    (
        "examples/legacy/parameter_estimation_with_nested_sampling.py",
        "examples/nested_sampling_estimation.md",
    ),
    (
        "examples/legacy/multi_parameter_estimation_with_NUTS.py",
        "examples/multi_parameter_nuts.md",
    ),
    (
        "examples/legacy/multi_parameter_estimation_with_nested_sampling.py",
        "examples/multi_parameter_nested.md",
    ),
    (
        "examples/legacy/null_parameter_estimation_with_nested_sampling.py",
        "examples/null_hypothesis_test.md",
    ),
]

# Convert each script
for script_path, output_path in EXAMPLE_SCRIPTS:
    full_script_path = Path(script_path)
    convert_script_to_docs(full_script_path, output_path)

# Create placeholder example files for referenced examples
PLACEHOLDER_EXAMPLES = [
    (
        "examples/multi_parameter_estimation.md",
        "Multi-parameter Estimation",
        "Advanced parameter estimation with multiple GW sources",
    ),
    (
        "examples/custom_noise_models.md",
        "Custom Noise Models",
        "Building and using custom noise models",
    ),
    (
        "examples/gw_detection.md",
        "Gravitational Wave Detection",
        "Detecting and characterizing GW signals",
    ),
    (
        "examples/nanograv_analysis.md",
        "NANOGrav Analysis",
        "Analysis using NANOGrav 15-year data",
    ),
    (
        "examples/mock_data_challenges.md",
        "Mock Data Challenges",
        "IPTA mock data challenge analysis",
    ),
    (
        "examples/custom_datasets.md",
        "Custom Datasets",
        "Working with your own timing data",
    ),
    ("examples/loading_data.md", "Loading Data", "Data loading and preprocessing"),
    ("examples/hpc_usage.md", "HPC Usage", "High-performance computing with Argus"),
    (
        "examples/optimization.md",
        "Optimization Tips",
        "Performance optimization techniques",
    ),
]

for file_path, title, description in PLACEHOLDER_EXAMPLES:
    with mkdocs_gen_files.open(file_path, "w") as f:
        f.write(
            f"""# {title}

{description}

!!! warning "Under Development"
    This example is currently under development. Please check back later for complete documentation.

## Placeholder Content

This section will contain:

- Step-by-step tutorial
- Complete working code examples  
- Expected outputs and interpretation
- Tips and best practices

## Related Examples

- [Basic Parameter Estimation](basic_parameter_estimation.md)
- [NUTS Estimation](nuts_estimation.md)
- [Nested Sampling Estimation](nested_sampling_estimation.md)

## Contributing

If you'd like to contribute to this example, please see our [Contributing Guide](../contributing.md).
"""
        )

print("Example documentation generated successfully!")
