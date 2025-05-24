# Frai Development Makefile
# Provides convenient commands for development, testing, and deployment

.PHONY: help install install-dev test lint format clean run

# Help target
help: ## Show help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package
	pip install -e .

install-dev: ## Install with dev dependencies
	pip install -e .[dev,test]

test: ## Run tests
	pytest

lint: ## Run linting
	ruff check . --fix
	black .
	isort .

format: lint ## Format code

clean: ## Clean cache files
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ htmlcov/

run: ## Run application
	python main.py
