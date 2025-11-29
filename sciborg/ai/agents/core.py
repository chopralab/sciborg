"""
SciBORG Agent Core - Functional API

This module provides a simple functional interface for creating SciBORG agents.
For more advanced features (memory persistence, workflow planning, etc.), 
use the class-based SciborgAgent from newcore.py.

This module is kept for backward compatibility and simple use cases.
"""

from typing import Type, Literal
from inspect import signature, getdoc
from pydantic import BaseModel as BaseModelV2

# LangChain Core Imports
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import StructuredTool

# LangChain Classic Imports
from langchain_classic.agents import AgentExecutor, create_structured_chat_agent, tool
from langchain_classic.base_memory import BaseMemory
from langchain_classic.memory import (
    ConversationBufferWindowMemory,
    CombinedMemory
)

# LangChain Community Imports
from langchain_community.tools import HumanInputRun

# LangChain OpenAI
from langchain_openai import ChatOpenAI

# SciBORG Internal Imports
from sciborg.core.library.base import BaseDriverMicroservice
from sciborg.core.command.base import BaseDriverCommand
from sciborg.ai.memory.internal_logging import CustomActionLogSummaryMemory, FSAMemory
from sciborg.ai.memory.embedding import EmbeddingSummaryMemory
from sciborg.ai.prompts.agent import (
    HUMAN_TOOL_INSTRUCTIONS,
    ASSUME_DEFAULTS_INSTRUCTIONS,
    BASE_SCIBORG_CHAT_PROMPT_TEMPLATE,
    RAG_AS_A_TOOL_INSTRUCTIONS
)
from sciborg.ai.tools.core import LinqxTool
from sciborg.ai.agents.rag_agent import rag_agent


def command_to_tool(
    command: BaseDriverCommand,
    schema: Type[BaseModelV2] | None = None
) -> StructuredTool:
    """
    Convert a BaseDriverCommand to a LangChain StructuredTool.
    
    Note: This disables SciBORG validation but still allows for internal validation.
    
    Args:
        command: The command to convert
        schema: Optional Pydantic schema for validation
        
    Returns:
        StructuredTool instance
    """
    return StructuredTool.from_function(
        name=command.name,
        args_schema=schema,
        description=f"Function Signature:\n{signature(command._function)}\nFunction Docstring:\n{getdoc(command._function)}",
        func=command._function,
        handle_tool_error=True,
    )


def create_sciborg_chat_agent(
    microservice: BaseDriverMicroservice,
    llm: BaseLanguageModel | None = None,
    prompt_template: str = BASE_SCIBORG_CHAT_PROMPT_TEMPLATE,
    use_memory: Literal['chat', 'action', 'embedding', 'all'] | None = None,
    memory: BaseMemory | None = None,
    intermediate_memory_buffer: str = "",
    past_action_log: str = "",
    human_interaction: bool = True,
    assume_defaults: bool = False,
    rag_vectordb_path: str | None = None,
    agent_description: str | None = None,
    agent_as_a_tool: AgentExecutor | None = None,
    agent_as_a_fsa: bool = False,
    fsa_object: Type[BaseModelV2] | None = None,
    start_state: BaseModelV2 | None = None,
    use_sciborg_tools: bool = True,
    handle_tool_error: bool = True,
    verbose: bool = False,
    return_intermediate_steps: bool = False,
    **agent_executor_kwargs
) -> AgentExecutor:
    """
    Create a SciBORG chat agent using a functional API.
    
    This is a simple functional interface for creating agents. For advanced features
    like memory persistence, workflow planning, and graph visualization, use the
    class-based SciborgAgent from newcore.py.
    
    Args:
        microservice: The microservice (driver side) that the agent will operate
        llm: The LLM to build the agent with (defaults to GPT-4)
        prompt_template: The prompt template string. Must include placeholders:
            ['tools', 'tool_names', 'agent_scratchpad', 'chat_history', 
             'microservice', 'microservice_description']
        use_memory: Memory type to use ('chat', 'action', 'embedding', 'all', or None)
        memory: Pre-initialized memory object (optional)
        intermediate_memory_buffer: Initial buffer for action memory
        past_action_log: Initial action log string
        human_interaction: Enable human interaction during operation
        assume_defaults: Agent assumes defaults when used as a tool by other agents
        rag_vectordb_path: Path to RAG vector database for document retrieval
        agent_description: Description of the agent
        agent_as_a_tool: Another agent to use as a tool
        agent_as_a_fsa: Enable FSA (Finite State Automaton) mode
        fsa_object: Pydantic model class for FSA state
        start_state: Initial FSA state object
        use_sciborg_tools: Use SciBORG tools (LinqxTool) vs standard tools
        handle_tool_error: Enable error handling in tools
        verbose: Enable verbose logging
        return_intermediate_steps: Return intermediate steps in response
        **agent_executor_kwargs: Additional kwargs for AgentExecutor
        
    Returns:
        AgentExecutor: An AI agent which operates the provided microservice
        
    Raises:
        ValueError: If human_interaction and assume_defaults are both True
        
    Example:
        ```python
        # Build driver side microservice
        my_microservice = module_to_microservice(my_module)
        
        # Build agent executor
        sciborg_agent = create_sciborg_chat_agent(microservice=my_microservice)
        
        # Query the agent
        result = sciborg_agent.invoke({'input': 'Please help me with ...'})
        ```
    """
    if llm is None:
        llm = ChatOpenAI(model='gpt-4')
    
    # Validate incompatible options
    if human_interaction and assume_defaults:
        raise ValueError(
            "Agent is not designed to communicate with a human and be used as a tool at the same time. "
            "If you intend to use the agent as a tool and need human interaction, have a managing agent "
            "set up for human interaction and use this agent as a tool."
        )

    # Define input and partial variables
    input_variables = ['tools', 'tool_names', 'agent_scratchpad']
    partial_variables = {
        'microservice': microservice.name,
        'microservice_description': microservice.desc,
    }

    # Add memory-related input variables
    if use_memory == 'all':
        input_variables.append('chat_history')
        input_variables.append('past_action_log')
        input_variables.append('embedding_log')
    elif use_memory == 'chat':
        input_variables.append('chat_history')
        partial_variables['past_action_log'] = past_action_log
    elif use_memory == 'action':
        input_variables.append('past_action_log')
        partial_variables['chat_history'] = ""
    elif use_memory == 'embedding' and rag_vectordb_path:
        input_variables.append('embedding_log')
    else:
        partial_variables['chat_history'] = ""
        partial_variables['past_action_log'] = past_action_log

    # Add instructions based on configuration
    TOTAL_INSTRUCTIONS = ""
    if human_interaction:
        TOTAL_INSTRUCTIONS += HUMAN_TOOL_INSTRUCTIONS
    if assume_defaults:
        TOTAL_INSTRUCTIONS += ASSUME_DEFAULTS_INSTRUCTIONS
    if rag_vectordb_path:
        TOTAL_INSTRUCTIONS += RAG_AS_A_TOOL_INSTRUCTIONS
    
    partial_variables['additional_instructions'] = TOTAL_INSTRUCTIONS

    # Build prompt from template
    prompt = PromptTemplate(
        input_variables=input_variables,
        partial_variables=partial_variables,
        template=prompt_template,
    )

    # Build tools from microservice commands
    if use_sciborg_tools:
        tools = [
            LinqxTool(sciborg_command=command, handle_tool_error=handle_tool_error) 
            for command in microservice.commands.values()
        ]
    else:
        tools = [
            command_to_tool(command) 
            for command in microservice.commands.values()
        ]

    action_tool_names = [x.name for x in tools]

    # Add human interaction tool if enabled
    if human_interaction:
        tools.append(HumanInputRun())
    
    # Add RAG agent as tool if path provided
    if rag_vectordb_path:
        @tool
        def call_RAG_agent(question: str) -> str:
            """
            Query relevant sources of information to answer a question.
            
            Use this when you need to search for information in documents or when
            the user specifies you should refer to "information" or "documents".
            
            TASK: Frame the best possible descriptive question and use it to query
            the relevant sources. Return citations if present.
            """
            RAG_agent = rag_agent(question, rag_vectordb_path)
            store_output = RAG_agent.invoke({"question": question})
            return store_output['output']
        
        tools.append(call_RAG_agent)

    # Add another agent as a tool if provided
    if agent_as_a_tool is not None:
        @tool
        def call_provided_Agent(question: str) -> str:
            """Call the provided agent as a tool."""
            output = agent_as_a_tool.invoke({"input": question})
            return output['output']
        tools.append(call_provided_Agent)

    # Initialize memory based on configuration
    context_tool_names = list(set([x.name for x in tools]) - set(action_tool_names))

    if use_memory == 'all' and memory is None and not agent_as_a_fsa:
        chat_memory = ConversationBufferWindowMemory(
            memory_key='chat_history', 
            input_key='input', 
            output_key='output'
        )
        intermediate_memory = CustomActionLogSummaryMemory(
            llm=ChatOpenAI(temperature=0, model='gpt-4'),
            memory_key='past_action_log',
            input_key='input',
            output_key='intermediate_steps',
            filtered_tool_list=action_tool_names,
            buffer=intermediate_memory_buffer
        )
        if rag_vectordb_path:
            embedding_memory = EmbeddingSummaryMemory(
                llm=ChatOpenAI(temperature=0),
                memory_key='rag_log',
                input_key='input',
                output_key='intermediate_steps',
                filtered_tool_list=context_tool_names
            )
            memory = CombinedMemory(memories=[chat_memory, intermediate_memory, embedding_memory])
        else:
            memory = CombinedMemory(memories=[chat_memory, intermediate_memory])
        return_intermediate_steps = True
        
    elif use_memory == 'all' and memory is None and agent_as_a_fsa:
        chat_memory = ConversationBufferWindowMemory(
            memory_key='chat_history',
            input_key='input',
            output_key='output'
        )
        intermediate_memory = FSAMemory(
            llm=ChatOpenAI(temperature=0, model='gpt-4'),
            fsa_object=fsa_object,
            memory_key='past_action_log',
            input_key='input',
            output_key='intermediate_steps',
            buffer=intermediate_memory_buffer
        )
        if rag_vectordb_path:
            embedding_memory = EmbeddingSummaryMemory(
                llm=ChatOpenAI(temperature=0),
                memory_key='rag_log',
                input_key='input',
                output_key='intermediate_steps',
                filtered_tool_list=context_tool_names
            )
            memory = CombinedMemory(memories=[chat_memory, intermediate_memory, embedding_memory])
        else:
            memory = CombinedMemory(memories=[chat_memory, intermediate_memory])
        
        # Initialize FSA buffer if needed
        if fsa_object is not None and start_state is None and intermediate_memory_buffer == "":
            intermediate_memory.buffer = fsa_object().model_dump_json(indent=2)
        elif fsa_object is None and start_state is not None and intermediate_memory_buffer == "":
            intermediate_memory.buffer = start_state.model_dump_json(indent=2)
        
        return_intermediate_steps = True
        
    elif use_memory == 'chat' and memory is None:
        memory = ConversationBufferWindowMemory(
            memory_key='chat_history',
            input_key='input',
            output_key='output'
        )
        return_intermediate_steps = False
        
    elif use_memory == 'action' and memory is None and not agent_as_a_fsa:
        memory = CustomActionLogSummaryMemory(
            llm=ChatOpenAI(temperature=0, model='gpt-4'),
            memory_key='past_action_log',
            input_key='input',
            output_key='intermediate_steps',
            filtered_tool_list=action_tool_names,
            buffer=intermediate_memory_buffer
        )
        return_intermediate_steps = True
        
    elif use_memory == 'action' and memory is None and agent_as_a_fsa:
        memory = FSAMemory(
            llm=ChatOpenAI(temperature=0, model='gpt-4'),
            fsa_object=fsa_object,
            memory_key='past_action_log',
            input_key='input',
            output_key='intermediate_steps',
            buffer=intermediate_memory_buffer
        )
        return_intermediate_steps = True
        
        # Initialize FSA buffer if needed
        if fsa_object is not None and start_state is None and intermediate_memory_buffer == "":
            memory.buffer = fsa_object().model_dump_json(indent=2)
        elif fsa_object is None and start_state is not None and intermediate_memory_buffer == "":
            memory.buffer = start_state.model_dump_json(indent=2)
    else:
        memory = None

    # Create agent from prompt and tools
    agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=verbose,
        handle_parsing_errors=True,
        return_intermediate_steps=return_intermediate_steps,
        **agent_executor_kwargs
    )
