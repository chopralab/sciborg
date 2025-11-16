"""
Temporary test script for Component 1: Dependencies & Requirements migration verification.
Delete after successful migration.

This script verifies that all required packages can be imported correctly
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
    print("Testing LangChain v1.0+ imports...")
    
    try:
        import langchain
        version = getattr(langchain, '__version__', 'installed')
        print(f"  ✓ langchain {version}")
    except ImportError as e:
        print(f"  ✗ langchain import failed: {e}")
        return False
    
    try:
        import langchain_core
        version = getattr(langchain_core, '__version__', 'installed')
        print(f"  ✓ langchain_core {version}")
    except ImportError as e:
        print(f"  ✗ langchain_core import failed: {e}")
        return False
    
    try:
        import langchain_openai
        version = getattr(langchain_openai, '__version__', 'installed')
        print(f"  ✓ langchain_openai {version}")
    except ImportError as e:
        print(f"  ✗ langchain_openai import failed: {e}")
        return False
    
    try:
        import langchain_community
        version = getattr(langchain_community, '__version__', 'installed')
        print(f"  ✓ langchain_community {version}")
    except ImportError as e:
        print(f"  ✗ langchain_community import failed: {e}")
        return False
    
    try:
        import langchain_classic
        version = getattr(langchain_classic, '__version__', 'installed')
        print(f"  ✓ langchain_classic {version}")
    except ImportError as e:
        print(f"  ✗ langchain_classic import failed: {e}")
        return False
    
    return True

def test_langchain_classic_imports():
    """Test langchain-classic imports for legacy components"""
    print("\nTesting langchain-classic imports...")
    
    try:
        from langchain_classic.chains import LLMChain
        print("  ✓ langchain_classic.chains.LLMChain")
    except ImportError as e:
        print(f"  ✗ langchain_classic.chains.LLMChain import failed: {e}")
        return False
    
    try:
        from langchain_classic import hub
        print("  ✓ langchain_classic.hub")
    except ImportError as e:
        print(f"  ✗ langchain_classic.hub import failed: {e}")
        return False
    
    return True

def test_agent_executor_import():
    """Test AgentExecutor import (moved to langchain_classic.agents in v1.0)"""
    print("\nTesting AgentExecutor import...")
    
    try:
        from langchain_classic.agents import AgentExecutor, create_structured_chat_agent
        print("  ✓ langchain_classic.agents.AgentExecutor")
        print("  ✓ langchain_classic.agents.create_structured_chat_agent")
    except ImportError as e:
        print(f"  ✗ AgentExecutor import failed: {e}")
        # Try langchain.agents as fallback
        try:
            from langchain.agents import AgentExecutor, create_structured_chat_agent
            print("  ✓ langchain.agents.AgentExecutor (fallback)")
            print("  ✓ langchain.agents.create_structured_chat_agent (fallback)")
        except ImportError as e2:
            print(f"  ✗ Fallback import also failed: {e2}")
            return False
    
    return True

def test_pydantic_v2():
    """Test Pydantic v2"""
    print("\nTesting Pydantic v2...")
    
    try:
        import pydantic
        version = getattr(pydantic, '__version__', 'unknown')
        if version != 'unknown':
            major_version = int(version.split('.')[0])
            if major_version >= 2:
                print(f"  ✓ pydantic {version} (v2)")
            else:
                print(f"  ✗ pydantic {version} (expected v2)")
                return False
        else:
            # Check if it's v2 by trying to import v2-specific features
            try:
                from pydantic import model_validator
                print(f"  ✓ pydantic installed (v2)")
            except ImportError:
                print(f"  ✗ pydantic v2 features not available")
                return False
    except ImportError as e:
        print(f"  ✗ pydantic import failed: {e}")
        return False
    
    return True

def test_core_imports():
    """Test core LangChain components"""
    print("\nTesting core LangChain components...")
    
    try:
        from langchain_core.language_models import BaseLanguageModel
        print("  ✓ langchain_core.language_models.BaseLanguageModel")
    except ImportError as e:
        print(f"  ✗ BaseLanguageModel import failed: {e}")
        return False
    
    try:
        from langchain_core.runnables import RunnableSequence
        print("  ✓ langchain_core.runnables.RunnableSequence")
    except ImportError as e:
        print(f"  ✗ RunnableSequence import failed: {e}")
        return False
    
    try:
        from langchain_core.output_parsers import JsonOutputParser
        print("  ✓ langchain_core.output_parsers.JsonOutputParser")
    except ImportError as e:
        print(f"  ✗ JsonOutputParser import failed: {e}")
        return False
    
    return True

def main():
    """Run all dependency tests"""
    print("=" * 60)
    print("Component 1: Dependencies & Requirements Test")
    print("=" * 60)
    
    results = []
    
    results.append(("LangChain v1.0+ imports", test_langchain_imports()))
    results.append(("langchain-classic imports", test_langchain_classic_imports()))
    results.append(("AgentExecutor import", test_agent_executor_import()))
    results.append(("Pydantic v2", test_pydantic_v2()))
    results.append(("Core components", test_core_imports()))
    
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All dependency tests passed!")
        print("Component 1 migration verified successfully.")
    else:
        print("✗ Some tests failed. Please review the errors above.")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())

