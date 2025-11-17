"""
SciBORG Agent Core Implementation

This module provides the main class-based agent implementation for SciBORG.
It includes support for:
- Multiple memory types (chat, action, embedding, FSA)
- Memory persistence (save/load)
- Workflow planning
- RAG integration
- FSA (Finite State Automaton) support
- Agent-to-agent communication

The SciborgAgent class provides a comprehensive, extensible interface for building
and managing AI agents that operate microservices.
"""

import json
import time
from typing import Any, Literal, Type, Optional
from inspect import signature, getdoc
from pydantic import BaseModel

# LangChain Core Imports
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import StructuredTool
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.utils.input import get_color_mapping

# LangChain Classic Imports (for legacy AgentExecutor support)
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
from sciborg.ai.prompts.memory import ACTION_LOG_FSA_TEMPLATE



class SciborgAgentExecutor(AgentExecutor):
    """
    Custom AgentExecutor with FSA (Finite State Automaton) support.
    
    Extends the base AgentExecutor to provide:
    - FSA state management
    - Custom action hooks for state updates
    - Node highlighting based on state checks
    """
    
    check_dict: dict[str, Any] = {}
    node_highlighters: dict[str, Any] = {}
    fsa_schema: Type[BaseModel] = None
    current_fsa: BaseModel = None
    llm_chain: Optional[Runnable] = None

    def __init__(self, **data: Any):
        """Initialize the executor with FSA support."""
        super().__init__(**data)
        
        # Initialize LLM chain for FSA state updates
        if self.fsa_schema is not None:
            parser = JsonOutputParser(pydantic_object=self.fsa_schema)
            prompt = PromptTemplate(
                input_variables=['new_lines', 'summary'],
                partial_variables={'formatting_instructions': parser.get_format_instructions()},
                template=ACTION_LOG_FSA_TEMPLATE
            )
            llm = ChatOpenAI(temperature=0, model='gpt-4')
            self.llm_chain = prompt | llm

    def _call(
        self,
        inputs: dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> dict[str, Any]:
        """Run text through and get agent response with FSA state updates."""
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools], excluded_colors=["green", "red"]
        )
        intermediate_steps: list[tuple[AgentAction, str]] = []

        # Let's start tracking the number of iterations and time elapsed
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
        # We now enter the agent loop (until it returns something).
        while self._should_continue(iterations, time_elapsed):
            next_step_output = self._take_next_step(
                name_to_tool_map,
                color_mapping,
                inputs,
                intermediate_steps,
                run_manager=run_manager,
            )
            if isinstance(next_step_output, AgentFinish):
                return self._return(
                    next_step_output, intermediate_steps, run_manager=run_manager
                )

            intermediate_steps.extend(next_step_output)
            
            # Update FSA state based on agent actions
            self.custom_update_on_agent_action(intermediate_steps)
            
            if len(next_step_output) == 1:
                next_step_action = next_step_output[0]
                # See if tool should return directly
                tool_return = self._get_tool_return(next_step_action)
                if tool_return is not None:
                    return self._return(
                        tool_return, intermediate_steps, run_manager=run_manager
                    )
            iterations += 1
            time_elapsed = time.time() - start_time
        output = self._action_agent.return_stopped_response(
            self.early_stopping_method, intermediate_steps, **inputs
        )
        return self._return(output, intermediate_steps, run_manager=run_manager)

    def custom_update_on_agent_action(
        self, intermediate_steps: list[tuple[AgentAction, str]]
    ) -> None:
        """
        Custom update hook called when an AgentAction is produced.
        
        Updates FSA state based on agent actions and checks state conditions
        for node highlighting.
        
        Args:
            intermediate_steps: List of (AgentAction, observation) tuples
        """
        if self.current_fsa is None or self.llm_chain is None:
            return
            
        print(f"Intermediate steps so far:\n {intermediate_steps}")

        # Convert FSA object to a proper JSON-serializable dictionary
        fsa_dict = self.current_fsa.model_dump() if hasattr(self.current_fsa, 'model_dump') else vars(self.current_fsa)
        
        # Convert None values to null for JSON compatibility
        fsa_json = json.dumps(fsa_dict)

        result = self.llm_chain.invoke({
            "summary": json.loads(fsa_json),  # Ensure we're passing proper JSON
            "new_lines": str(intermediate_steps)
        })
        result = result.content
        print(f"Result from LLM chain: {result}")
        
        # Parse the result if it's a string
        if isinstance(result, str):
            try:
                result = json.loads(result.replace('\'',"\"").replace('None', 'null'))
            except json.JSONDecodeError:
                # If it's not valid JSON, try evaluating it as a Python literal
                import ast
                try:
                    result = ast.literal_eval(result)
                except (SyntaxError, ValueError):
                    # If all else fails, just use an empty dict to avoid errors
                    print(f"Warning: Could not parse result as JSON or Python literal: {result}")
                    result = {}
        
        # Instead of creating a new object, update the existing one
        for key, value in result.items():
            setattr(self.current_fsa, key, value)
            
        print(f"Updated FSA: {self.current_fsa}")

        print(f"Check dict: {self.check_dict}")

        # Check state conditions and trigger node highlighters
        for key, val in self.check_dict.items():
            print(f"Checking {key} with value: {val(self.current_fsa)}")
            if val(self.current_fsa):
                print(f"Check {key} passed with value: {val(self.current_fsa)}")
                if key in self.node_highlighters:
                    self.node_highlighters[key](self.current_fsa)


class SciborgAgent:
    """
    Main SciBORG Agent class for building and managing AI agents.
    
    This class provides a comprehensive interface for creating agents that can:
    - Operate microservices through commands
    - Use multiple memory types (chat, action, embedding, FSA)
    - Save and load memory state
    - Plan and execute workflows
    - Integrate with RAG for document retrieval
    - Support agent-to-agent communication
    
    Example:
        ```python
        agent = SciborgAgent(
            microservice=my_microservice,
            use_memory='all',
            memory_file='session_1.json'
        )
        result = agent.invoke("Please help me with...")
        agent.save_memory()
        ```
    """
    
    def __init__(
        self,
        microservice: BaseDriverMicroservice,
        llm: BaseLanguageModel = ChatOpenAI(temperature=0, model='gpt-4'),
        prompt_template: str = BASE_SCIBORG_CHAT_PROMPT_TEMPLATE,
        use_memory: Literal['chat', 'action', 'embedding', 'all'] | None = None,
        memory: BaseMemory | None = None,
        intermediate_memory_buffer: str = "",
        past_action_log: str = "",
        human_interaction: bool = False,
        assume_defaults: bool = False,
        rag_vectordb_path: str | None = None,
        agent_description: str | None = None,
        agent_as_a_tool: AgentExecutor | None = None,
        agent_as_a_fsa: bool = False,
        fsa_schema: Type[BaseModel] | None = None,
        fsa_object: BaseModel | None = None,
        use_sciborg_tools: bool = True,
        handle_tool_error: bool = True,
        verbose: bool = False,
        return_intermediate_steps: bool = False,
        memory_file: str | None = None,
        **agent_executor_kwargs
    ):
        """
        Initialize a SciBORG agent.
        
        Args:
            microservice: The microservice the agent will operate
            llm: Language model to use (defaults to GPT-4)
            prompt_template: Prompt template for the agent
            use_memory: Memory type(s) to use ('chat', 'action', 'embedding', 'all', or list)
            memory: Pre-initialized memory object (optional)
            intermediate_memory_buffer: Initial buffer for action memory
            past_action_log: Initial action log
            human_interaction: Enable human interaction during execution
            assume_defaults: Agent assumes defaults when used as a tool
            rag_vectordb_path: Path to RAG vector database
            agent_description: Description of the agent
            agent_as_a_tool: Another agent to use as a tool
            agent_as_a_fsa: Enable FSA (Finite State Automaton) mode
            fsa_schema: Pydantic model class for FSA state
            fsa_object: Initial FSA state object
            use_sciborg_tools: Use SciBORG tools (LinqxTool) vs standard tools
            handle_tool_error: Enable error handling in tools
            verbose: Enable verbose logging
            return_intermediate_steps: Return intermediate steps in response
            memory_file: Path to memory file for persistence
            **agent_executor_kwargs: Additional kwargs for AgentExecutor
        """
        # Store configuration
        self.microservice = microservice
        self.rag_vectordb_path = rag_vectordb_path
        self.llm = llm
        self.use_memory = use_memory
        self.memory = memory
        self.human_interaction = human_interaction
        self.assume_defaults = assume_defaults
        self.verbose = verbose
        self.return_intermediate_steps = return_intermediate_steps
        self.agent_executor_kwargs = agent_executor_kwargs
        self.memory_file = memory_file
        self.fsa_schema = fsa_schema
        
        # Initialize FSA object if schema provided
        if fsa_object is not None:
            self._fsa_object = fsa_object
        elif fsa_schema is not None:
            self._fsa_object = fsa_schema.model_construct()
        else:
            self._fsa_object = None
        
        # Stateflow functionality removed - FSA support no longer uses stateflow chains
        
        # Build agent components
        self.tools = self._build_tools(use_sciborg_tools, handle_tool_error, agent_as_a_tool, agent_description)
        self.memory = self._initialize_memory(intermediate_memory_buffer, past_action_log, agent_as_a_fsa, fsa_schema, fsa_object)
        self.prompt = self._build_prompt(prompt_template, rag_vectordb_path, past_action_log)
        self.agent_executor = self._create_agent_executor()

    @property
    def fsa_object(self) -> BaseModel | None:
        """
        Get the current FSA object.
        
        If fsa_schema is provided but _fsa_object is not initialized,
        automatically creates a default instance of the schema.
        """
        if hasattr(self, 'fsa_schema') and self.fsa_schema and not hasattr(self, '_fsa_object'):
            self._fsa_object = self.fsa_schema.model_construct()
        elif not hasattr(self, '_fsa_object'):
            self._fsa_object = None
        return self._fsa_object
        
    @fsa_object.setter
    def fsa_object(self, value: BaseModel | None):
        """Set the FSA object."""
        self._fsa_object = value

    def _command_to_tool(
        self, 
        command: BaseDriverCommand,
        schema: Type[BaseModel] | None = None
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

    def _build_tools(
        self, 
        use_sciborg_tools: bool, 
        handle_tool_error: bool, 
        agent_as_a_tool: AgentExecutor | None, 
        agent_description: str | None
    ) -> list[StructuredTool]:
        """
        Build the list of tools available to the agent.
        
        Args:
            use_sciborg_tools: Whether to use SciBORG tools (LinqxTool)
            handle_tool_error: Enable error handling
            agent_as_a_tool: Another agent to use as a tool
            agent_description: Description of the agent tool
            
        Returns:
            List of tools
        """
        tools = []
        
        # Add microservice commands as tools
        if use_sciborg_tools:
            tools = [
                LinqxTool(sciborg_command=command, handle_tool_error=handle_tool_error) 
                for command in self.microservice.commands.values()
            ]
        else:
            tools = [
                self._command_to_tool(command) 
                for command in self.microservice.commands.values()
            ]
        
        # Add human interaction tool if enabled
        if self.human_interaction:
            tools.append(HumanInputRun())
        
        # Add RAG agent as tool if path provided
        if self.rag_vectordb_path:
            @tool
            def call_RAG_agent(question: str) -> str:
                """
                Query relevant sources of information to answer a question.
                
                Use this when you need to search for information in documents or
                when the user specifies you should refer to "information" or "documents".
                
                TASK: Frame the best possible descriptive question and use it to query
                the relevant sources. Return citations if present.
                """
                RAG_agent = rag_agent(question, self.rag_vectordb_path)
                return RAG_agent.invoke({"question": question})['output']
            tools.append(call_RAG_agent)
        
        # Add another agent as a tool if provided
        if agent_as_a_tool:
            @tool
            def call_provided_Agent(question: str) -> str:
                """Call the provided agent as a tool."""
                output = agent_as_a_tool.invoke({"input": question})
                return output['output']
            tools.append(call_provided_Agent)
        
        return tools

    def _initialize_memory(
        self, 
        intermediate_memory_buffer: str, 
        past_action_log: str, 
        agent_as_a_fsa: bool, 
        fsa_schema: Type[BaseModel] | None, 
        fsa_object: BaseModel | None
    ) -> BaseMemory | None:
        """
        Initialize memory based on configuration.
        
        Args:
            intermediate_memory_buffer: Initial buffer for action memory
            past_action_log: Initial action log
            agent_as_a_fsa: Whether agent is in FSA mode
            fsa_schema: FSA schema class
            fsa_object: Initial FSA object
            
        Returns:
            Memory object or None
        """
        action_tool_names = [tool.name for tool in self.tools]
        use_memory = self.use_memory
        memories = []
        rag_vectordb_path = self.rag_vectordb_path

        # Handle list of memory types
        if isinstance(use_memory, list):
            for memory_type in use_memory:
                self.return_intermediate_steps = True
                if memory_type == 'chat':
                    memories.append(ConversationBufferWindowMemory(
                        memory_key='chat_history',
                        input_key='input',
                        output_key='output'
                    ))
                if memory_type == 'action':
                    memories.append(CustomActionLogSummaryMemory(
                        llm=ChatOpenAI(temperature=0, model='gpt-4'),
                        memory_key='past_action_log',
                        input_key='input',
                        output_key='intermediate_steps',
                        buffer=intermediate_memory_buffer,
                        filtered_tool_list=action_tool_names
                    ))
                if memory_type == 'embedding' and rag_vectordb_path:
                    memories.append(EmbeddingSummaryMemory(
                        llm=ChatOpenAI(temperature=0),
                        memory_key='rag_log',
                        input_key='input',
                        output_key='intermediate_steps',
                        filtered_tool_list=action_tool_names
                    ))
                if memory_type == 'fsa' and agent_as_a_fsa:
                    memories.append(FSAMemory(
                        llm=ChatOpenAI(temperature=0, model='gpt-4'),
                        memory_key='fsa_log',
                        input_key='input',
                        output_key='intermediate_steps',
                        fsa_object=fsa_object,
                        buffer=intermediate_memory_buffer,
                    ))
            return CombinedMemory(memories=memories) if memories else None

        # Handle single memory type
        elif use_memory:
            self.return_intermediate_steps = True
            
            if use_memory == 'chat':
                return ConversationBufferWindowMemory(
                    memory_key='chat_history',
                    input_key='input',
                    output_key='output'
                )
            if use_memory == 'action':
                return CustomActionLogSummaryMemory(
                    llm=ChatOpenAI(temperature=0, model='gpt-4'),
                    memory_key='past_action_log',
                    input_key='input',
                    output_key='intermediate_steps',
                    buffer=intermediate_memory_buffer,
                    filtered_tool_list=action_tool_names
                )
            if use_memory == 'embedding' and rag_vectordb_path:
                return EmbeddingSummaryMemory(
                    llm=ChatOpenAI(temperature=0),
                    memory_key='rag_log',
                    input_key='input',
                    output_key='intermediate_steps'
                )
            if use_memory == 'fsa' and agent_as_a_fsa:
                return FSAMemory(
                    llm=ChatOpenAI(temperature=0, model='gpt-4'),
                    memory_key='fsa_log',
                    input_key='input',
                    output_key='intermediate_steps',
                    fsa_schema=fsa_schema,
                    buffer=intermediate_memory_buffer,
                )
            if use_memory == 'all':
                memories = [
                    ConversationBufferWindowMemory(
                        memory_key='chat_history',
                        input_key='input',
                        output_key='output'
                    ),
                    CustomActionLogSummaryMemory(
                        llm=self.llm,
                        memory_key='past_action_log',
                        input_key='input',
                        output_key='intermediate_steps',
                        buffer=intermediate_memory_buffer,
                        filtered_tool_list=action_tool_names
                    )
                ]
                if self.rag_vectordb_path:
                    memories.append(EmbeddingSummaryMemory(
                        llm=self.llm,
                        memory_key='rag_log',
                        input_key='input',
                        output_key='intermediate_steps'
                    ))
                if agent_as_a_fsa:
                    memories.append(FSAMemory(
                        llm=self.llm,
                        memory_key='fsa_log',
                        input_key='input',
                        output_key='intermediate_steps',
                        fsa_schema=fsa_schema,
                        buffer=intermediate_memory_buffer,
                    ))
                return CombinedMemory(memories=memories)
        
        return None

    def _build_prompt(
        self, 
        prompt_template: str, 
        rag_vectordb_path: str | None, 
        past_action_log: str
    ) -> PromptTemplate:
        """
        Build the prompt template for the agent.
        
        Args:
            prompt_template: Base prompt template string
            rag_vectordb_path: Path to RAG vector database
            past_action_log: Initial action log
            
        Returns:
            Configured PromptTemplate
        """
        input_variables = ['tools', 'tool_names', 'agent_scratchpad']
        partial_variables = {
            'microservice': self.microservice.name,
            'microservice_description': self.microservice.desc,
        }

        # Add memory-related input variables based on configuration
        use_memory = self.use_memory
        if isinstance(use_memory, list):
            for memory_type in use_memory:
                self.return_intermediate_steps = True
                if memory_type == 'chat':
                    input_variables.append('chat_history')
                if memory_type == 'action':
                    input_variables.append('past_action_log')
                if memory_type == 'embedding' and rag_vectordb_path:
                    input_variables.append('embedding_log')
        elif use_memory:
            self.return_intermediate_steps = True
            if use_memory == 'chat':
                input_variables.append('chat_history')
            if use_memory == 'action':
                input_variables.append('past_action_log')
            if use_memory == 'embedding' and rag_vectordb_path:
                input_variables.append('embedding_log')
            if use_memory == 'all':
                input_variables.append('chat_history')
                input_variables.append('past_action_log')
                if rag_vectordb_path:
                    input_variables.append('embedding_log')
        else:
            partial_variables['past_action_log'] = 'Logging not used'
            partial_variables['chat_history'] = 'Chat history not used'

        # Add instructions based on configuration
        TOTAL_INSTRUCTIONS = ""
        if self.human_interaction:
            TOTAL_INSTRUCTIONS += HUMAN_TOOL_INSTRUCTIONS
        if self.assume_defaults:
            TOTAL_INSTRUCTIONS += ASSUME_DEFAULTS_INSTRUCTIONS
        if self.rag_vectordb_path:
            TOTAL_INSTRUCTIONS += RAG_AS_A_TOOL_INSTRUCTIONS
        
        partial_variables['additional_instructions'] = TOTAL_INSTRUCTIONS

        return PromptTemplate(
            input_variables=input_variables,
            partial_variables=partial_variables,
            template=prompt_template
        )
    
    def _create_agent_executor(self) -> SciborgAgentExecutor:
        """Create and configure the agent executor."""
        agent = create_structured_chat_agent(llm=self.llm, tools=self.tools, prompt=self.prompt)

        agext_exec = SciborgAgentExecutor(
            agent=agent, 
            tools=self.tools, 
            memory=self.memory, 
            verbose=self.verbose, 
            return_intermediate_steps=self.return_intermediate_steps,
            current_fsa=self.fsa_object, 
            fsa_schema=self.fsa_schema,
            **self.agent_executor_kwargs
        )
        
        # Load memory from file if specified
        if self.memory_file:
            memories = self._load_memory()
            if agext_exec.memory and hasattr(agext_exec.memory, 'memories'):
                agext_exec.memory.memories = memories
        
        return agext_exec
    
    def save_memory(self, memory_file: str | None = None) -> None:
        """
        Save agent memory to a JSON file.
        
        Args:
            memory_file: Path to save memory file (defaults to self.memory_file or 'memory.json')
        """
        if not memory_file:
            if not self.memory_file:
                self.memory_file = "memory.json"
            memory_file = self.memory_file
            print("Memory file name not provided, saving memory to default file called memory.json")
        else:
            self.memory_file = memory_file
            print(f"Memory file name provided, saving memory to {memory_file}")

        if not self.agent_executor.memory or not hasattr(self.agent_executor.memory, 'memories'):
            print("Warning: No memory to save")
            return

        memories_data = {}

        for memory in self.agent_executor.memory.memories:
            if memory:
                if isinstance(memory, ConversationBufferWindowMemory):
                    data_to_add = []
                    for message in memory.buffer_as_messages:
                        if isinstance(message, HumanMessage):
                            message_type = "human"
                        elif isinstance(message, AIMessage):
                            message_type = "ai"
                        else:
                            message_type = "unknown"
                        message_data = {
                            "message_type": message_type,
                            "content": message.content,
                        }
                        data_to_add.append(message_data)

                    memories_data['ConversationBufferWindowMemory'] = {
                        "memory_key": memory.memory_key,
                        "input_key": memory.input_key,
                        "output_key": memory.output_key,
                        "messages": data_to_add
                    }

                if isinstance(memory, CustomActionLogSummaryMemory):
                    memories_data['CustomActionLogSummaryMemory'] = {
                        "memory_key": memory.memory_key,
                        "input_key": memory.input_key,
                        "output_key": memory.output_key,
                        "buffer": memory.buffer,
                        "filtered_tool_list": memory.filtered_tool_list
                    }

                if isinstance(memory, EmbeddingSummaryMemory):
                    memories_data['EmbeddingSummaryMemory'] = {
                        "memory_key": memory.memory_key,
                        "input_key": memory.input_key,
                        "output_key": memory.output_key,
                        "buffer": memory.buffer,
                        "filtered_tool_list": memory.filtered_tool_list
                    }

        with open(memory_file, "w") as f:
            json.dump(memories_data, f, indent=4)

        print("Memory saved successfully")
        print(f"Memory file saved at {memory_file}")

    def _load_memory(self) -> list[BaseMemory]:
        """
        Load memory from a JSON file.
        
        Returns:
            List of memory objects
        """
        memory_from_file = {}
        try:
            with open(self.memory_file, "r") as f:
                memory_from_file = json.load(f)
        except FileNotFoundError:
            return []

        memories = []

        for memory_name, memory_data in memory_from_file.items():
            if memory_name == 'ConversationBufferWindowMemory':
                add_messages = []
                for message_data in memory_data['messages']:
                    if message_data['message_type'] == "human":
                        message = HumanMessage(content=message_data['content'])
                    elif message_data['message_type'] == "ai":
                        message = AIMessage(content=message_data['content'])
                    else:
                        message = None
                    if message is not None:
                        add_messages.append(message)

                to_add = ConversationBufferWindowMemory(
                    memory_key=memory_data['memory_key'],
                    input_key=memory_data['input_key'],
                    output_key=memory_data['output_key']
                )
                to_add.chat_memory.messages = add_messages
                memories.append(to_add)
            
            if memory_name == 'CustomActionLogSummaryMemory':
                to_add = CustomActionLogSummaryMemory(
                    llm=ChatOpenAI(temperature=0, model='gpt-4'),
                    memory_key=memory_data['memory_key'],
                    input_key=memory_data['input_key'],
                    output_key=memory_data['output_key'],
                    buffer=memory_data['buffer'],
                    filtered_tool_list=memory_data['filtered_tool_list']
                )
                memories.append(to_add)

            if memory_name == 'EmbeddingSummaryMemory':
                to_add = EmbeddingSummaryMemory(
                    llm=ChatOpenAI(temperature=0, model='gpt-4'),
                    memory_key=memory_data['memory_key'],
                    input_key=memory_data['input_key'],
                    output_key=memory_data['output_key'],
                    buffer=memory_data['buffer'],
                    filtered_tool_list=memory_data['filtered_tool_list']
                )
                memories.append(to_add)
        
        return memories
    
    def invoke(self, input_text: str | dict) -> Any:
        """
        Invoke the agent with input text.
        
        Args:
            input_text: Input text (string or dict with 'input' key)
            
        Returns:
            Agent response
        """
        if isinstance(input_text, dict):
            input_text = input_text.get("input", "")
        
        return self.agent_executor.invoke({"input": input_text})
    
    def prime(self, input_content: str) -> None:
        """
        Prime the agent with a workflow description.
        
        NOTE: This method has been disabled as stateflow functionality has been removed.
        For workflow visualization, consider using alternative methods or re-implementing
        without stateflow dependencies.
        
        Args:
            input_content: Detailed workflow description
            
        Raises:
            NotImplementedError: Stateflow functionality has been removed
        """
        raise NotImplementedError(
            "The prime() method has been disabled as stateflow functionality has been removed. "
            "This method previously used stateflow chains for workflow visualization. "
            "Consider using alternative visualization methods or re-implementing without stateflow."
        )

