"""Generate API documentation pages."""

import os
from pathlib import Path
import mkdocs_gen_files

# Define the module structure for Argus
MODULES = [
    "argus.data_loader",
    "argus.model",
    "argus.bayesian_inference",
    "argus.jax_kalman_filter",
    "argus.gravitational_waves",
    "argus.io_manager",
    "argus.utils",
    "argus.workflow",
]

# Create API documentation structure
nav_content = []
nav_content.append("# API Reference\n")

for module in MODULES:
    module_name = module.split(".")[-1]
    filename = f"api/{module_name}.md"

    # Create module documentation page
    with mkdocs_gen_files.open(filename, "w") as f:
        f.write(f"# {module_name}\n\n")
        f.write(f"::: {module}\n")

    # Add to navigation
    nav_content.append(f"- [{module_name.replace('_', ' ').title()}]({module_name}.md)")

# Create the main API index page
with mkdocs_gen_files.open("api/index.md", "w") as f:
    f.write("# API Reference\n\n")
    f.write("This section contains the complete API documentation for Argus.\n\n")
    f.write("## Modules\n\n")

    for module in MODULES:
        module_name = module.split(".")[-1]
        f.write(f"- [{module_name.replace('_', ' ').title()}]({module_name}.md) - ")

        # Add brief descriptions for each module
        descriptions = {
            "data_loader": "Load and preprocess pulsar timing data",
            "model": "Core Argus model implementation",
            "bayesian_inference": "Bayesian parameter estimation routines",
            "jax_kalman_filter": "JAX-based Kalman filtering for state-space analysis",
            "gravitational_waves": "Gravitational wave signal modeling",
            "io_manager": "Input/output and configuration management",
            "utils": "Utility functions and helpers",
            "workflow": "End-to-end analysis workflows",
        }

        f.write(descriptions.get(module_name, "Module functionality") + "\n")

# Create navigation file for literate-nav plugin
with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.write("\n".join(nav_content))

print("API documentation generated successfully!")
