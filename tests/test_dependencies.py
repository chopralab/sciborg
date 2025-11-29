"""
Dependency verification tests for SciBORG.

This test suite verifies that all required packages can be imported correctly
after updating to LangChain v1.0+.
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

def test_langchain_imports():
    """Test LangChain v1.0+ imports"""
    import langchain
    import langchain_core
    import langchain_openai
    import langchain_community
    import langchain_classic
    
    # Verify they can be imported
    assert langchain is not None
    assert langchain_core is not None
    assert langchain_openai is not None
    assert langchain_community is not None
    assert langchain_classic is not None

def test_langchain_classic_imports():
    """Test langchain-classic imports for legacy components"""
    from langchain_classic.chains import LLMChain
    from langchain_classic import hub
    
    assert LLMChain is not None
    assert hub is not None

def test_agent_executor_import():
    """Test AgentExecutor import (moved to langchain_classic.agents in v1.0)"""
    from langchain_classic.agents import AgentExecutor, create_structured_chat_agent
    
    assert AgentExecutor is not None
    assert create_structured_chat_agent is not None

def test_pydantic_v2():
    """Test Pydantic v2"""
    import pydantic
    from pydantic import model_validator
    
    version = getattr(pydantic, '__version__', 'unknown')
    if version != 'unknown':
        major_version = int(version.split('.')[0])
        assert major_version >= 2, f"Expected Pydantic v2, got {version}"
    
    # Verify v2-specific features are available
    assert model_validator is not None

def test_core_imports():
    """Test core LangChain components"""
    from langchain_core.language_models import BaseLanguageModel
    from langchain_core.runnables import RunnableSequence
    from langchain_core.output_parsers import JsonOutputParser
    
    assert BaseLanguageModel is not None
    assert RunnableSequence is not None
    assert JsonOutputParser is not None


