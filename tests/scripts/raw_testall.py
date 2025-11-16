#============================================================
# Imports and environment variables
#============================================================

import os
import sys
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
import traceback

# Unicode symbols for test status
PASS_SYMBOL = "✅"
FAIL_SYMBOL = "❌"

def log_status(message, status=True):
    """Log the status of a test with a checkmark or cross."""
    symbol = PASS_SYMBOL if status else FAIL_SYMBOL
    print(f"{symbol} {message}")

# Get the project root directory (sciborg root)
current_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
project_root = current_dir

# Look for sciborg root by going up until we find .env or sciborg directory
while project_root != project_root.parent:
    if (project_root / '.env').exists() or project_root.name == 'sciborg':
        break
    project_root = project_root.parent
    if project_root == project_root.parent:  # Reached filesystem root
        # Fallback: assume we're in notebooks/testing, go up two levels
        project_root = current_dir
        if 'notebooks' in str(project_root):
            project_root = project_root.parent.parent
        break

# Load .env file from project root
try:
    env_path = project_root / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path))
        log_status(f"Environment variables loaded from: {env_path}")
    else:
        # Try loading from current directory as fallback
        load_dotenv()
        log_status("Warning: .env not found in project root, trying current directory")
except Exception as e:
    log_status(f"Error loading environment variables: {e}", status=False)
    sys.exit(1)

try:
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    os.environ["OPENAI_API_KEY"] = openai_key
    log_status("OpenAI API key set successfully.")
except Exception as e:
    log_status(f"Error setting OpenAI API key: {e}", status=False)
    sys.exit(1)

# Add project root to Python path
sys.path.insert(0, str(project_root))

try:
    from langchain_openai import ChatOpenAI
    from sciborg.ai.agents.core import create_sciborg_chat_agent
    from sciborg.ai.chains.microservice import module_to_microservice
    from sciborg.ai.chains.workflow import create_workflow_planner_chain
    from sciborg.utils.drivers import PubChemCaller
    from sciborg.core.library.base import BaseDriverMicroservice
    log_status("Modules imported successfully.")
except ImportError as e:
    log_status(f"Error importing required modules: {e}", status=False)
    sys.exit(1)

#============================================================
# Building the Pubchem Microservice
#============================================================

file_path = project_root / 'ai' / 'agents' / 'driver_pubchem.json'

try:
    pubchem_command_microservice = module_to_microservice(PubChemCaller)
    with open(file_path, 'w') as outfile:
        outfile.write(pubchem_command_microservice.model_dump_json(indent=2))
    log_status("PubChem microservice built and saved successfully.")
except Exception as e:
    log_status(f"Error creating PubChem microservice: {e}", status=False)
    sys.exit(1)

#================================================================
# Planner Chain
#================================================================

try:
    planner = create_workflow_planner_chain(
        llm=ChatOpenAI(model='gpt-4'),
        library=pubchem_command_microservice
    )
    log_status("Workflow planner chain initialized successfully.")
except Exception as e:
    log_status(f"Error initializing workflow planner chain: {e}", status=False)
    sys.exit(1)

try:
    planner_response = planner.invoke(
        {
            "query": "What is the IC50 of 1-[(2S)-2-(dimethylamino)-3-(4-hydroxyphenyl)propyl]-3-[(2S)-1-thiophen-3-ylpropan-2-yl]urea to the Mu opioid receptor, cite a specific assay in your response?"
        }
    )
    log_status("Planner chain query invoked successfully.")
    print(planner_response)
except Exception as e:
    log_status(f"Error invoking planner chain: {e}", status=False)

#============================================================
# Pubchem Agent
#============================================================

try:
    pubchem_agent = create_sciborg_chat_agent(
        microservice=pubchem_command_microservice,
        llm=ChatOpenAI(model='gpt-4'),
        human_interaction=False,
        verbose=True
    )
    log_status("PubChem agent created successfully.")
except Exception as e:
    log_status(f"Error creating PubChem agent: {e}", status=False)
    sys.exit(1)

try:
    agent_response = pubchem_agent.invoke(
        {
            "input": "What is the Ki of pzm21 to the Mu opioid receptor, cite a specific assay in your response?"
        }
    )
    log_status("PubChem agent query invoked successfully.")
    print(agent_response)
except Exception as e:
    log_status(f"Error invoking PubChem agent: {e}", status=False)

#============================================================
# RAG Agent + Pubchem Microservice
#============================================================

try:
    rag_agent = create_sciborg_chat_agent(
        microservice=pubchem_command_microservice,
        rag_vectordb_path=str(project_root / 'embeddings' / 'NIH_docs_embeddings'),
        llm=ChatOpenAI(model='gpt-4'),
        human_interaction=False,
        verbose=True
    )
    log_status("RAG agent created successfully.")
except Exception as e:
    log_status(f"Error creating RAG agent: {e}", status=False)
    sys.exit(1)

try:
    rag_response = rag_agent.invoke(
        {
            "input": "How does microwave irradiation influence reaction mechanisms differently compared to conventional heating methods?"
        }
    )
    log_status("RAG agent query invoked successfully.")
    print(rag_response)
except Exception as e:
    log_status(f"Error invoking RAG agent: {e}", status=False)

#============================================================
# Final Output
#============================================================

print("\n🎉 All tests completed! 🎉")