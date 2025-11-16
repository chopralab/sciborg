# Tests Directory

This directory contains all testing code for SciBORG, organized into:

## Structure

```
tests/
├── test_dependencies.py      # Dependency verification tests
├── test_workflow_chains.py    # Workflow chains tests
├── notebooks/                 # Testing notebooks
│   ├── experimental.ipynb
│   ├── experiment2.ipynb
│   └── experiment3.ipynb
└── scripts/                   # Testing scripts
    └── raw_testall.py
```

## Test Files

### `test_dependencies.py`
Verifies that all required packages can be imported correctly after updating to LangChain v1.0+.

**Usage:**
```bash
python tests/test_dependencies.py
```

### `test_workflow_chains.py`
Verifies that workflow chains work correctly after updating to LCEL.

**Usage:**
```bash
python tests/test_workflow_chains.py
```

## Notebooks

The `notebooks/` subdirectory contains experimental and testing notebooks for development and validation.

## Scripts

The `scripts/` subdirectory contains utility scripts for testing various components.

