# Git Hooks Setup Guide

This document explains how to set up and use git hooks for automated code quality checks in the vla-metrics project.

## Overview

The project uses [pre-commit](https://pre-commit.com/) to manage git hooks that automatically check code quality before commits. The hooks include:

- **Black**: Automatic code formatting
- **Ruff**: Fast Python linting
- **Pylint**: Additional code quality checks
- **MyPy**: Static type checking
- **Interrogate**: Docstring coverage checking
- **Standard checks**: Trailing whitespace, file endings, YAML/TOML validation, etc.

## Installation

### 1. Install Development Dependencies

First, install the package with development dependencies:

```bash
pip install -e ".[dev]"
```

This installs `pre-commit`, `pylint`, `black`, `ruff`, `mypy`, and other dev tools.

### 2. Install Git Hooks

Install the pre-commit hooks into your git repository:

```bash
pre-commit install
```

This creates a `.git/hooks/pre-commit` script that runs automatically before each commit.

### 3. (Optional) Install Commit Message Hook

To also check commit messages:

```bash
pre-commit install --hook-type commit-msg
```

## Usage

### Automatic Running on Commit

Once installed, the hooks run automatically when you commit:

```bash
git add src/vla_metrics/new_file.py
git commit -m "feat: add new metric"
```

The hooks will:
1. Check for common issues (trailing whitespace, file endings, etc.)
2. Format code with Black
3. Lint code with Ruff (and auto-fix issues)
4. Check code quality with Pylint
5. Type-check with MyPy
6. Check docstring coverage with Interrogate

If any hook fails, the commit is aborted and you'll see error messages.

### Manual Running

Run hooks manually on all files:

```bash
# Run all hooks on all files
pre-commit run --all-files

# Run specific hook on all files
pre-commit run black --all-files
pre-commit run pylint --all-files
pre-commit run mypy --all-files

# Run hooks on specific files
pre-commit run --files src/vla_metrics/task_performance/success_rate.py
```

### Skipping Hooks (Not Recommended)

In rare cases, you may need to skip hooks:

```bash
# Skip all hooks (use sparingly!)
git commit --no-verify -m "message"

# Set SKIP environment variable to skip specific hooks
SKIP=pylint git commit -m "message"
```

**Note**: Only skip hooks when absolutely necessary. Code should pass all checks before merging.

## Hook Details

### Black - Code Formatter

**What it does**: Automatically formats Python code to ensure consistent style.

**Configuration**: `pyproject.toml` → `[tool.black]`
- Line length: 100 characters
- Target versions: Python 3.8+

**Example**:
```bash
# Run manually
black src/ tests/ examples/

# Check without modifying
black --check src/
```

### Ruff - Fast Linter

**What it does**: Checks for code quality issues and common bugs. Much faster than flake8/pylint for basic checks.

**Configuration**: `pyproject.toml` → `[tool.ruff]`
- Enabled rules: E (errors), F (pyflakes), I (imports), N (naming), W (warnings), B (bugbear), C4 (comprehensions), UP (pyupgrade)

**Example**:
```bash
# Run manually
ruff src/ tests/ examples/

# Auto-fix issues
ruff --fix src/
```

### Pylint - Code Quality Checker

**What it does**: Performs thorough code quality analysis, checking for bugs, code smells, and style issues.

**Configuration**: `pyproject.toml` → `[tool.pylint.*]`
- Max line length: 100
- Disabled checks: Missing docstrings (handled by interrogate), too-few-public-methods, protected-access

**Example**:
```bash
# Run manually
pylint src/vla_metrics/

# Run on specific file
pylint src/vla_metrics/task_performance/success_rate.py
```

### MyPy - Type Checker

**What it does**: Verifies type hints and catches type-related bugs.

**Configuration**: `pyproject.toml` → `[tool.mypy]`
- Target: Python 3.8
- Requires type hints for all functions

**Example**:
```bash
# Run manually
mypy src/

# Run on specific module
mypy src/vla_metrics/task_performance/
```

### Interrogate - Docstring Coverage

**What it does**: Checks that code has adequate docstring coverage.

**Configuration**: `pyproject.toml` → `[tool.interrogate]`
- Minimum coverage: 80%
- Excludes: tests/, examples/

**Example**:
```bash
# Run manually
interrogate src/

# Detailed report
interrogate -v src/
```

## Updating Hooks

Pre-commit hooks are pinned to specific versions in `.pre-commit-config.yaml`. To update:

```bash
# Update to latest versions
pre-commit autoupdate

# Test updated hooks
pre-commit run --all-files
```

## Troubleshooting

### Hooks Not Running

If hooks don't run on commit:

```bash
# Reinstall hooks
pre-commit uninstall
pre-commit install

# Verify installation
pre-commit run --all-files
```

### Hook Failures

If a hook fails:

1. **Read the error message**: It usually tells you exactly what's wrong
2. **Run the hook manually**: `pre-commit run <hook-name> --all-files`
3. **Fix the issue**: Make the required changes
4. **Re-run hooks**: `pre-commit run --all-files` to verify
5. **Commit again**: Try your commit again

### Slow Hook Performance

Some hooks (especially pylint and mypy) can be slow on large changes:

- **Run incrementally**: Commit smaller changes more frequently
- **Cache**: Pre-commit caches results; subsequent runs are faster
- **Skip temporarily**: Use `SKIP=pylint,mypy git commit -m "message"` for work-in-progress commits, but fix before pushing

### Type Checking Errors

If mypy reports errors about missing imports:

```bash
# Install type stubs
pip install types-setuptools types-requests

# Or ignore specific imports in pyproject.toml
# [tool.mypy]
# ignore_missing_imports = true  # Not recommended for production
```

## CI/CD Integration

The same hooks run in CI/CD pipelines to ensure code quality. Local hooks help catch issues before pushing:

```yaml
# Example GitHub Actions workflow
- name: Run pre-commit hooks
  run: |
    pip install pre-commit
    pre-commit run --all-files
```

## Configuration Files

- **`.pre-commit-config.yaml`**: Hook configuration and versions
- **`pyproject.toml`**: Tool-specific configurations (black, ruff, pylint, mypy, interrogate)
- **`.git/hooks/pre-commit`**: Auto-generated hook script (don't edit manually)

## Best Practices

1. **Install hooks immediately** after cloning the repository
2. **Run `pre-commit run --all-files`** periodically to catch issues
3. **Don't skip hooks** unless absolutely necessary
4. **Keep hooks updated** with `pre-commit autoupdate`
5. **Commit frequently** to minimize hook run time
6. **Fix issues immediately** rather than accumulating technical debt

## Additional Resources

- [Pre-commit documentation](https://pre-commit.com/)
- [Black documentation](https://black.readthedocs.io/)
- [Ruff documentation](https://docs.astral.sh/ruff/)
- [Pylint documentation](https://pylint.pycqa.org/)
- [MyPy documentation](https://mypy.readthedocs.io/)
