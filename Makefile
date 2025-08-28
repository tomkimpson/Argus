# Makefile for Argus project
#

# MkDocs variables  
MKDOCS_CONFIG = mkdocs.yml
MKDOCS_SITE   = site

# Set project variables
PKG_PROJECT := argus
PKG_VERSION := $(shell git describe --tags --always --dirty)

# Put help first so that "make" without an argument displays available targets
help:
	@echo "Available targets:"
	@echo "  docs         - Build documentation with MkDocs"
	@echo "  serve        - Start MkDocs development server"
	@echo "  docs-deploy  - Deploy documentation to GitHub Pages"
	@echo "  clean        - Clean build artifacts and cache"
	@echo "  clean-docs   - Clean documentation build artifacts"

.PHONY: help clean clean-docs docs serve docs-deploy

# Project targets

clean: clean-docs
	@echo "Cleaning project artifacts..."
	@rm -rf dist
	@rm -rf .tests
	@rm -rf .pytest_cache
	@rm -rf .cache
	@rm -rf .ropeproject
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete"

# Documentation targets

clean-docs:
	@echo "Cleaning documentation artifacts..."
	@rm -rf site
	@rm -rf build
	@echo "Documentation clean complete"

docs:
	@echo "Building documentation with MkDocs..."
	@mkdocs build
	@echo "Documentation built successfully in ./site/"

serve:
	@echo "Starting MkDocs development server..."
	@echo "Documentation will be available at http://127.0.0.1:8000"
	@mkdocs serve

docs-deploy:
	@echo "Deploying documentation to GitHub Pages..."
	@mkdocs gh-deploy
	@echo "Documentation deployed successfully"

