.PHONY: help install test fmt lint type-check run clean setup-dev

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install package and dependencies
	pip install -e .

install-dev: ## Install package with development dependencies
	pip install -e ".[dev]"

setup-dev: install-dev ## Setup development environment
	pre-commit install

test: ## Run tests
	pytest

fmt: ## Format code with ruff and black
	ruff format .
	ruff check --fix .

lint: ## Run linting
	ruff check .

type-check: ## Run type checking
	mypy invoice_pdf/ tests/

run: ## Run the legacy script (example with test input)
	python main_2step_enhanced.py --input input/test --output output/test

run-new: ## Run the new modular version (when available)
	python -m invoice_pdf --input input/test --output output/test

clean: ## Clean up generated files
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

ci: lint type-check test ## Run all CI checks locally