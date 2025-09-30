# Contributing to Argus

Thank you for your interest in contributing to **Argus**!

This project is an ongoing open-source effort to support cutting-edge research in gravitational wave detection with pulsar timing arrays. Contributions from the community are very welcome.

## How to Contribute

Whether you're fixing bugs, improving documentation, adding features, or suggesting ideas — all contributions are appreciated!

### 🛠️ Steps to Contribute

1. **Fork the repository**  
   Click the "Fork" button on GitHub to create your own copy of the repository.

2. **Clone your fork and set up the development environment**

   ```bash
   git clone https://github.com/your-username/Argus.git
   cd Argus

   # Create conda environment with Python 3.11
   conda create -n argus-dev python=3.11
   conda activate argus-dev

   # Install enterprise-pulsar via conda
   conda install -c conda-forge enterprise-pulsar

   # Install Argus in editable mode with dev dependencies
   pip install -e ".[dev]"
   ```

3. **Create a new branch for your change**

   ```bash
   git checkout -b my-feature-branch
   ```

4. **Make your changes**

   Write clean, well-documented code. Include tests where appropriate.

5. **Commit and push**

   ```bash
   git add .
   git commit -m "Add feature X"
   git push origin my-feature-branch
   ```

6. **Open a Pull Request (PR)**
   Go to your fork on GitHub and open a PR against the `main` branch of the original repository.

   Please describe your changes clearly and tag any relevant issues.

---

## Code Style and Guidelines

- Follow **PEP8** for Python code style.
- Use **type hints** where helpful.
- Include **docstrings** for all public functions and classes.
- If you're modifying core algorithms, include a test or example.

## Reporting Issues

If you find a bug or have a feature request, please [open an issue](https://github.com/tomkimpson/Argus/issues) on GitHub with as much detail as possible.

---

We’re grateful for your support and contributions!

– The Argus Team
