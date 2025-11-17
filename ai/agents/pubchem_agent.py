"""
PubChem Agent

A specialized agent for querying PubChem API to answer chemistry-related questions.
This agent can look up chemical compounds, substances, assays, and related information.
"""

from langchain_openai import ChatOpenAI
from langchain_classic.agents import tool, AgentExecutor, create_structured_chat_agent
from langchain_classic.agents.format_scratchpad import format_to_openai_function_messages
from langchain_classic.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.language_models import BaseLanguageModel
from sciborg.ai.chains.microservice import extract_docstring

from sciborg.utils.drivers.PubChemCaller import (
    get_synonym,
    get_description,
    get_assay_description_from_AID as get_assay_description,
    get_assay_name_from_aid,
    get_compound_property_table,
    get_cid_from_name,
    get_smiles_from_name
)


def pubchem_agent(
    question: str,
    llm: BaseLanguageModel = ChatOpenAI(temperature=0.1)
) -> AgentExecutor:
    """
    Create a PubChem agent with specialized chemistry tools.
    
    This agent can answer questions related to chemistry using the PubChem API.
    Use this agent when you need to look up chemical information that's not in
    your knowledge base.
    
    Args:
        question: The question to answer (used for initialization)
        llm: Language model to use (defaults to GPT-4 with temperature=0.1)
        
    Returns:
        AgentExecutor: Configured agent ready to answer chemistry questions
        
    Example:
        ```python
        agent = pubchem_agent("What is the molecular weight of caffeine?")
        result = agent.invoke({"question": "What is the molecular weight of caffeine?"})
        ```
    """
    # Wrap functions with @tool decorator
    # extract_docstring handles f-strings and edge cases that tool() can't read
    # It optimizes internally by checking func.__doc__ first, so overhead is minimal
    tools = [
        tool(func, description=extract_docstring(func))
        for func in [
            get_synonym,
            get_description,
            get_assay_description,
            get_assay_name_from_aid,
            get_compound_property_table,
            get_cid_from_name,
            get_smiles_from_name
        ]
    ]

    llm_with_tools = llm.bind(functions=[convert_to_openai_function(t) for t in tools])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a powerful chemistry assistant with access to PubChem API.
                
                Your task is to use the appropriate tool to answer what the user is asking for.
                
                Process:
                    1. Rephrase the question in a query in the most explanatory way possible.
                    2. Break the task down into smaller parts depending on the tools that are available to you.
                    3. Call the tools one by one that are available to you to get the result.
                    4. Provide the answer in a clear format to the user.

                Valid input types:
                    - sid: substance id
                    - cid: compound id
                    - aid: assay id
                    - smiles: The simplified molecular-input line-entry system (SMILES) is a specification 
                              in the form of a line notation for describing the structure of chemical 
                              species using short ASCII strings. 
                              Examples: C=C, Na.Cl, C#N, CBr, CC
                    - names: Common names of chemical compounds

                Important: In your final response, do not add any URLs or hyperlinks.
                """,
            ),
            ("user", "{question}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    chain = (
        {
            "question": lambda x: x["question"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | OpenAIFunctionsAgentOutputParser()
    )

    pubchem_agent_executor = AgentExecutor(agent=chain, tools=tools, verbose=True)

    return pubchem_agent_executor
