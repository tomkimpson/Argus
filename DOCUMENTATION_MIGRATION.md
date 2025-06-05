# Documentation Migration: Sphinx → Material for MkDocs

This document outlines the migration from Sphinx to Material for MkDocs for the Argus project documentation.

## 🔄 Migration Status

✅ **Complete**: The new MkDocs documentation system is ready for production use.

## 🌐 Deployment

### GitHub Pages Setup

The documentation is automatically deployed to GitHub Pages when changes are pushed to the `main` branch.

**Live URL**: https://tomkimpson.github.io/Argus/

### Repository Settings Required

To enable GitHub Pages deployment, ensure the following settings in your GitHub repository:

1. **Go to Settings → Pages**
2. **Source**: Deploy from a branch
3. **Branch**: `gh-pages` (will be created automatically by the workflow)
4. **Folder**: `/ (root)`

### Workflow Files

- **`.github/workflows/docs.yml`**: Builds and deploys documentation on pushes to `main`
- **`.github/workflows/pull_request.yml`**: Tests both Sphinx and MkDocs builds on PRs

## 🛠️ Local Development

### Prerequisites

```bash
# Activate conda environment
conda activate Argus

# Install MkDocs dependencies (if not already installed)
pip install mkdocs-material "mkdocstrings[python]" mkdocs-gen-files mkdocs-literate-nav markdown-katex
```

### Commands

```bash
# Serve locally with live reload
mkdocs serve
# or: make serve-new

# Build static site
mkdocs build 
# or: make docs-new

# Deploy to GitHub Pages (manual)
mkdocs gh-deploy
# or: make mkdocs-deploy
```

## 📁 New Structure

```
docs/
├── mkdocs.yml                 # Main configuration
├── index.md                   # Landing page  
├── getting_started.md         # Installation guide
├── contributing.md            # Contributing guidelines
├── state_space.md            # Technical content
├── notes_for_developers.md   # Developer documentation
├── examples/                 # Tutorial examples (placeholder)
├── notebooks/                # Jupyter notebook integration (placeholder)
├── api/                      # Auto-generated API docs
├── scripts/                  # Build automation scripts
├── assets/                   # Images and icons
├── stylesheets/              # Custom CSS
└── javascripts/              # Custom JavaScript
```

## 🆚 Comparison: Sphinx vs MkDocs

| Feature | Sphinx (Legacy) | MkDocs Material (New) |
|---------|----------------|----------------------|
| **Theme** | Read the Docs | Material Design |
| **Configuration** | Python (`conf.py`) | YAML (`mkdocs.yml`) |
| **Content Format** | RST + Markdown | Pure Markdown |
| **Build Speed** | Slower | Faster |
| **Live Reload** | Limited | Excellent |
| **Mobile Support** | Basic | Excellent |
| **Search** | Basic | Enhanced |
| **API Docs** | sphinx-autodoc | mkdocstrings |
| **Math Support** | MathJax | KaTeX |

## 🔮 Future Plans

1. **Remove Sphinx dependencies** once migration is confirmed stable
2. **Add custom logo/favicon** to replace placeholder icons
3. **Expand examples** with real tutorials and code samples
4. **Integrate Jupyter notebooks** for interactive documentation
5. **Add automated API documentation** testing and validation

## 🐛 Known Issues

- Some notebook links are placeholders (intentional)
- API documentation generates warnings for missing type annotations
- Favicon is a placeholder Material icon

## 📞 Support

For issues with the new documentation system:

1. Check the [GitHub Actions logs](https://github.com/tomkimpson/Argus/actions) for build failures
2. Open an issue in the repository
3. Refer to [MkDocs Material documentation](https://squidfunk.github.io/mkdocs-material/) for advanced configuration