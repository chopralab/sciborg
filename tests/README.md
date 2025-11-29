# Tests Directory

This directory contains all testing code for SciBORG.

## Structure

```
tests/
├── test_dependencies.py         # Dependency verification tests
├── test_workflow_chains.py      # Workflow chains tests
└── test_agent_integration.py    # Integration tests for agents
```

## Test Files

### `test_dependencies.py`
Verifies that all required packages can be imported correctly after updating to LangChain v1.0+.

**Usage:**
```bash
pytest tests/test_dependencies.py
```

### `test_workflow_chains.py`
Verifies that workflow chains work correctly after updating to LCEL.

**Usage:**
```bash
pytest tests/test_workflow_chains.py
```

### `test_agent_integration.py`
Integration tests for SciBORG agents using real API calls. Requires `OPENAI_API_KEY` environment variable.

**Usage:**
```bash
pytest tests/test_agent_integration.py
```

## Running All Tests

```bash
pytest tests/ -v
```

