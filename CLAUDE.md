# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutoTuner is a "no-human-in-loop" parameter tuning framework for commercial and academic RTL-to-GDS flows. It provides automatic hyperparameter tuning and parametric sweeping for OpenROAD-flow-scripts (ORFS).

Key features:
- Multiple search algorithms (HyperOpt, Ax, Optuna, PBT, Random, Nevergrad)
- Ray-based distributed execution
- Support for both `tune` and `sweep` modes
- Integration with METRICS2.1 for PPA capture
- Customizable evaluation functions

## Common Development Commands

### Installation and Setup
```bash
# Install prerequisites and setup virtual environment
./installer.sh
./setup.sh

# Install in development mode
pip install -e .
```

### Code Quality and Testing
```bash
# Format code and run linting
make format        # Runs ruff format and ruff check --fix
make check         # Runs ruff check and pre-commit hooks

# Run tests
python test/smoke_test_tune.py
python test/smoke_test_sweep.py
python test/smoke_test_sample_iteration.py
```

### Running AutoTuner
```bash
# Main CLI entry point
openroad_autotuner --design <design> --platform <platform> --config <config.json> <mode>

# Example tune command
openroad_autotuner --design gcd --platform sky130hd \
  --config config.json \
  tune --samples 5 --algorithm hyperopt

# Example sweep command
openroad_autotuner --design gcd --platform sky130hd \
  --config config.json \
  sweep
```

## Architecture Overview

### Core Components

1. **CLI Interface** (`src/autotuner/cli.py`):
   - Argument parsing and validation using Pydantic
   - Configuration processing and path management
   - Entry point for all AutoTuner operations

2. **Configuration Management** (`src/autotuner/core/config.py`):
   - Pydantic-based settings with validation
   - Separate configs for tune and sweep modes
   - Environment variable support via `.env` files

3. **Distributed Execution** (`src/autotuner/distributed.py`):
   - Ray-based distributed computing
   - `AutoTunerBase` and `PPAImprov` trainable classes
   - Algorithm setup and configuration
   - Search space management

4. **Utilities** (`src/autotuner/utils.py`):
   - Configuration parsing and validation
   - OpenROAD flow execution
   - Metrics processing
   - Ray server setup

### Key Classes

- `AutoTunerBase`: Base trainable class for Ray Tune experiments
- `PPAImprov`: Specialized evaluator for PPA (Performance, Power, Area) improvement
- `Settings`: Main configuration class with validation
- `TuneConfig`/`SweepConfig`: Mode-specific configuration

### Search Algorithms

Supported algorithms configured via `--algorithm` parameter:
- `hyperopt`: Tree Parzen Estimator (default)
- `ax`: Bayesian + Multi-Armed Bandit
- `optuna`: Tree Parzen Estimator + CMA-ES
- `pbt`: Population Based Training
- `random`: Random/Grid Search

### Evaluation Functions

- `default`: Optimizes effective clock period
- `ppa-improv`: Optimizes PPA improvement against reference metrics

## Development Patterns

### Configuration Structure

JSON configuration files define search spaces:
```json
{
  "_SDC_CLK_PERIOD": {
    "type": "float",
    "minmax": [1.0, 3.7],
    "step": 0
  },
  "CORE_MARGIN": {
    "type": "int",
    "minmax": [2, 10],
    "step": 1
  }
}
```

### Adding New Algorithms

1. Implement algorithm setup in `set_algorithm()` function
2. Add algorithm choice to CLI argument parser
3. Update configuration validation if needed
4. Add corresponding imports and dependencies

### Adding New Evaluation Functions

1. Create new trainable class inheriting from `AutoTunerBase`
2. Override `evaluate()` method with custom scoring logic
3. Add to `set_training_class()` function
4. Update CLI choices and validation

### Testing Strategy

- Smoke tests for each platform/design combination
- Tests cover both tune and sweep modes
- Use `accepted_rc` for acceptable return codes
- Test utilities in `test/autotuner_test_utils.py`

## File Structure

```
src/autotuner/
├── __init__.py
├── cli.py                    # CLI interface and argument parsing
├── distributed.py            # Ray-based distributed execution
├── utils.py                  # Utility functions
├── core/
│   └── config.py            # Pydantic configuration models
├── algorithms/              # Algorithm implementations (empty)
├── evaluators/              # Evaluation functions (empty)
├── runners/                 # Execution runners (empty)
└── io/                      # I/O utilities (empty)
```

## Dependencies

Key dependencies managed in `pyproject.toml`:
- `ray[tune]`: Distributed execution and hyperparameter tuning
- `ax-platform`, `optuna`, `hyperopt`: Search algorithms
- `pydantic`: Configuration validation
- `ruff`: Code formatting and linting
- `pytest`: Testing framework

## Environment Setup

The project uses Python 3.12+ with specific dependency versions. Development dependencies include testing, linting, and type checking tools.

Pre-commit hooks ensure code quality via ruff formatting and checking.
