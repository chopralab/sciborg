"""
SciBORG Agents Package

This package provides agent implementations for SciBORG:
- Functional API (core.py): Simple function-based agent creation
- Class-based API (newcore.py): Advanced agent with memory persistence, workflow planning
- Specialized agents: RAG and PubChem agents
"""

# Main agent implementations
from sciborg.ai.agents.core import (
    create_sciborg_chat_agent,
    command_to_tool
)

from sciborg.ai.agents.newcore import (
    SciborgAgent,
    SciborgAgentExecutor
)

# Specialized agents
from sciborg.ai.agents.rag_agent import rag_agent
from sciborg.ai.agents.pubchem_agent import pubchem_agent

__all__ = [
    # Functional API
    'create_sciborg_chat_agent',
    'command_to_tool',
    
    # Class-based API
    'SciborgAgent',
    'SciborgAgentExecutor',
    
    # Specialized agents
    'rag_agent',
    'pubchem_agent',
]

