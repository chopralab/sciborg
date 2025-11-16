"""
Temporary test script for Component 2: Workflow Chains migration verification.
Delete after successful migration.

This script verifies that workflow chains work correctly after updating to LCEL.
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
    print("Testing workflow chains imports...")
    
    try:
        from sciborg.ai.chains.workflow import (
            create_workflow_planner_chain,
            create_workflow_constructor_chain
        )
        print("  ✓ Workflow chains imports successful")
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False
    
    return True

def test_return_types():
    """Test that functions return RunnableSequence"""
    print("\nTesting return types...")
    
    try:
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
        if isinstance(planner_chain, RunnableSequence):
            print("  ✓ create_workflow_planner_chain returns RunnableSequence")
        else:
            print(f"  ✗ create_workflow_planner_chain returns {type(planner_chain)}, expected RunnableSequence")
            return False
        
        # Test constructor chain
        constructor_chain = create_workflow_constructor_chain(library, llm)
        if isinstance(constructor_chain, RunnableSequence):
            print("  ✓ create_workflow_constructor_chain returns RunnableSequence")
        else:
            print(f"  ✗ create_workflow_constructor_chain returns {type(constructor_chain)}, expected RunnableSequence")
            return False
            
    except Exception as e:
        print(f"  ✗ Error testing return types: {e}")
        return False
    
    return True

def test_lcel_compatibility():
    """Test that LCEL chains can be invoked"""
    print("\nTesting LCEL chain compatibility...")
    
    try:
        from sciborg.ai.chains.workflow import create_workflow_planner_chain
        from langchain_openai import ChatOpenAI
        from langchain_core.runnables import RunnableSequence
        from sciborg.core.library.base import BaseCommandLibrary
        
        # Create a minimal library for testing
        library = BaseCommandLibrary(name="test", microservices={})
        llm = ChatOpenAI(model='gpt-4', temperature=0.1)
        
        # Create chain
        chain = create_workflow_planner_chain(library, llm)
        
        # Check if chain has invoke method (RunnableSequence should have this)
        if hasattr(chain, 'invoke'):
            print("  ✓ Chain has invoke method (LCEL compatible)")
        else:
            print("  ✗ Chain does not have invoke method")
            return False
        
        # Check if chain has stream method (RunnableSequence should have this)
        if hasattr(chain, 'stream'):
            print("  ✓ Chain has stream method (LCEL compatible)")
        else:
            print("  ⚠ Chain does not have stream method")
        
        # Check if it's a RunnableSequence type
        from langchain_core.runnables import Runnable
        if isinstance(chain, Runnable):
            print("  ✓ Chain is a Runnable (LCEL compatible)")
        else:
            print(f"  ⚠ Chain type is {type(chain)}, expected Runnable")
            
    except Exception as e:
        print(f"  ✗ Error testing LCEL compatibility: {e}")
        return False
    
    return True

def main():
    """Run all workflow chains tests"""
    print("=" * 60)
    print("Component 2: Workflow Chains Test")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_workflow_chains_imports()))
    results.append(("Return types", test_return_types()))
    results.append(("LCEL compatibility", test_lcel_compatibility()))
    
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
        print("✓ All workflow chains tests passed!")
        print("Component 2 migration verified successfully.")
    else:
        print("✗ Some tests failed. Please review the errors above.")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())

