"""
Workflow chains tests for SciBORG.

This test suite verifies that workflow chains work correctly after updating to LCEL.
"""
import sys
import os
from dotenv import load_dotenv
from pathlib import Path

# Get the project root directory (sciborg root)
current_dir = Path(__file__).parent
project_root = current_dir.parent

# Look for sciborg root by going up until we find .env or sciborg directory
while project_root != project_root.parent:
    if (project_root / '.env').exists() or project_root.name == 'sciborg':
        break
    project_root = project_root.parent
    if project_root == project_root.parent:  # Reached filesystem root
        # Fallback: assume we're in tests/, go up one level
        project_root = current_dir.parent
        break

# Load .env file from project root
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=str(env_path))
    print(f"Loaded .env from: {env_path}")
else:
    # Try loading from current directory as fallback
    load_dotenv()
    print("Warning: .env not found in project root, trying current directory")

# Add project root to Python path
sys.path.insert(0, str(project_root))

def test_workflow_chains_imports():
    """Test workflow chains imports"""
    from sciborg.ai.chains.workflow import (
        create_workflow_planner_chain,
        create_workflow_constructor_chain
    )
    
    assert create_workflow_planner_chain is not None
    assert create_workflow_constructor_chain is not None

def test_return_types():
    """Test that functions return RunnableSequence"""
    from sciborg.ai.chains.workflow import (
        create_workflow_planner_chain,
        create_workflow_constructor_chain
    )
    from langchain_core.runnables import RunnableSequence
    from langchain_openai import ChatOpenAI
    from sciborg.core.library.base import BaseCommandLibrary
    
    # Create a minimal library for testing
    library = BaseCommandLibrary(name="test", microservices={})
    llm = ChatOpenAI(model='gpt-4', temperature=0.1)
    
    # Test planner chain
    planner_chain = create_workflow_planner_chain(library, llm)
    assert isinstance(planner_chain, RunnableSequence), \
        f"Expected RunnableSequence, got {type(planner_chain)}"
    
    # Test constructor chain
    constructor_chain = create_workflow_constructor_chain(library, llm)
    assert isinstance(constructor_chain, RunnableSequence), \
        f"Expected RunnableSequence, got {type(constructor_chain)}"

def test_lcel_compatibility():
    """Test that LCEL chains can be invoked"""
    from sciborg.ai.chains.workflow import create_workflow_planner_chain
    from langchain_openai import ChatOpenAI
    from langchain_core.runnables import Runnable
    from sciborg.core.library.base import BaseCommandLibrary
    
    # Create a minimal library for testing
    library = BaseCommandLibrary(name="test", microservices={})
    llm = ChatOpenAI(model='gpt-4', temperature=0.1)
    
    # Create chain
    chain = create_workflow_planner_chain(library, llm)
    
    # Check if chain has invoke method (RunnableSequence should have this)
    assert hasattr(chain, 'invoke'), "Chain should have invoke method (LCEL compatible)"
    
    # Check if it's a Runnable type
    assert isinstance(chain, Runnable), \
        f"Chain should be a Runnable, got {type(chain)}"


