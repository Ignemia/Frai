# Frai Development Makefile
# Provides convenient commands for development, testing, and deployment

.PHONY: help install install-dev test test-unit test-integration test-performance test-all
.PHONY: lint format type-check security-check quality-check
.PHONY: clean clean-cache clean-build clean-test clean-all
.PHONY: run run-dev run-docker build-docker
.PHONY: docs docs-serve release pre-commit setup

# Default Python version
PYTHON := python3
PIP := pip3

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Help target
help: ## Show this help message
	@echo "$(BLUE)Personal Chatter Development Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Setup Commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / && /Setup/ {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Development Commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / && /Development|Run/ {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Testing Commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / && /Test/ {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Quality Commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / && /Quality|Format|Lint|Type|Security/ {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Utility Commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / && /Clean|Build|Deploy|Docs/ {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Setup commands
setup: ## Setup: Complete development environment setup
	@echo "$(BLUE)Setting up Personal Chatter development environment...$(NC)"
	$(PYTHON) -m venv venv
	@echo "$(YELLOW)Activate virtual environment with: source venv/bin/activate (or venv\\Scripts\\activate on Windows)$(NC)"
	@echo "$(YELLOW)Then run: make install-dev$(NC)"

install: ## Setup: Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: ## Setup: Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -r requirements-test.txt
	$(PIP) install -e .

pre-commit: ## Setup: Install and run pre-commit hooks
	@echo "$(BLUE)Setting up pre-commit hooks...$(NC)"
	pre-commit install
	pre-commit install --hook-type commit-msg
	@echo "$(GREEN)Pre-commit hooks installed!$(NC)"

# Development commands
run: ## Development: Run the application in production mode
	@echo "$(BLUE)Starting Personal Chatter...$(NC)"
	$(PYTHON) main.py

run-dev: ## Development: Run the application in development mode
	@echo "$(BLUE)Starting Personal Chatter in development mode...$(NC)"
	uvicorn api.api:app --reload --host 0.0.0.0 --port 8000

run-docker: ## Development: Run the application using Docker
	@echo "$(BLUE)Starting Personal Chatter with Docker...$(NC)"
	docker-compose up --build

# Testing commands
test: ## Testing: Run all tests with coverage
	@echo "$(BLUE)Running all tests with coverage...$(NC)"
	$(PYTHON) -m pytest --cov=api --cov=services --cov-report=html --cov-report=term-missing --cov-report=xml

test-unit: ## Testing: Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	$(PYTHON) -m pytest tests/unit/ -v

test-integration: ## Testing: Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(PYTHON) -m pytest tests/integration/ -v

test-implementation: ## Testing: Run implementation tests only
	@echo "$(BLUE)Running implementation tests...$(NC)"
	$(PYTHON) -m pytest tests/implementation/ -v

test-blackbox: ## Testing: Run black box tests only
	@echo "$(BLUE)Running black box tests...$(NC)"
	$(PYTHON) -m pytest tests/blackbox/ -v

test-performance: ## Testing: Run performance tests only
	@echo "$(BLUE)Running performance tests...$(NC)"
	$(PYTHON) -m pytest tests/performance/ -v --benchmark-only

test-quick: ## Testing: Run quick tests (unit + integration)
	@echo "$(BLUE)Running quick tests...$(NC)"
	$(PYTHON) -m pytest tests/unit/ tests/integration/ -x --ff

test-watch: ## Testing: Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	$(PYTHON) -m pytest-watch

# Code quality commands
lint: ## Quality: Run linting with ruff
	@echo "$(BLUE)Running linting...$(NC)"
	ruff check . --fix

format: ## Quality: Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	black .
	isort .

type-check: ## Quality: Run type checking with mypy
	@echo "$(BLUE)Running type checking...$(NC)"
	mypy api/ services/ --ignore-missing-imports --show-error-codes

security-check: ## Quality: Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	bandit -r api/ services/ -ll
	safety check

quality-check: lint format type-check security-check ## Quality: Run all quality checks
	@echo "$(GREEN)All quality checks completed!$(NC)"

# Build and deployment
build-docker: ## Build: Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t personal-chatter:latest .

clean-cache: ## Clean: Remove Python cache files
	@echo "$(BLUE)Cleaning Python cache...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +

clean-build: ## Clean: Remove build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/

clean-test: ## Clean: Remove test artifacts
	@echo "$(BLUE)Cleaning test artifacts...$(NC)"
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf tests/coverage_html/
	rm -rf tests/coverage.xml
	rm -rf .mypy_cache/

clean-all: clean-cache clean-build clean-test ## Clean: Remove all generated files
	@echo "$(GREEN)All clean commands completed!$(NC)"

# Documentation
docs: ## Docs: Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	sphinx-build -b html docs/ docs/_build/

docs-serve: ## Docs: Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8080$(NC)"
	cd docs/_build && $(PYTHON) -m http.server 8080

# Release management
release-patch: ## Deploy: Create a patch release
	@echo "$(BLUE)Creating patch release...$(NC)"
	bump2version patch

release-minor: ## Deploy: Create a minor release
	@echo "$(BLUE)Creating minor release...$(NC)"
	bump2version minor

release-major: ## Deploy: Create a major release
	@echo "$(BLUE)Creating major release...$(NC)"
	bump2version major

# Utility commands
check-deps: ## Utility: Check for outdated dependencies
	@echo "$(BLUE)Checking for outdated dependencies...$(NC)"
	$(PIP) list --outdated

update-deps: ## Utility: Update dependencies (be careful!)
	@echo "$(YELLOW)Updating dependencies - review changes carefully!$(NC)"
	$(PIP) list --outdated --format=freeze | grep -v '^-e' | cut -d = -f 1 | xargs -n1 $(PIP) install -U

profile: ## Utility: Run application with profiling
	@echo "$(BLUE)Running with profiling...$(NC)"
	$(PYTHON) -m cProfile -o profile.stats main.py

# Database management (if applicable)
db-upgrade: ## Database: Upgrade database schema
	@echo "$(BLUE)Upgrading database...$(NC)"
	alembic upgrade head

db-downgrade: ## Database: Downgrade database schema
	@echo "$(BLUE)Downgrading database...$(NC)"
	alembic downgrade -1

db-reset: ## Database: Reset database (DANGER!)
	@echo "$(RED)Resetting database - this will destroy all data!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		alembic downgrade base && alembic upgrade head; \
	fi
